"""Веб-интерфейс юридического ассистента (FastAPI).

GET  /                     — чат
POST /api/chat             — ответ на последнюю реплику с учётом истории
GET  /api/documents        — реестр документов базы знаний
GET  /api/documents/{id}   — полный текст документа, разбитый на фрагменты
GET  /api/health           — состояние сервиса
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import legal_core as lc  # noqa: E402

ROOT = Path(__file__).resolve().parent
KB_PATH = Path(os.getenv("KB_PATH", str(ROOT / "data" / "knowledge-base.zip")))
MAX_HISTORY = 12
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", str(Path(os.getenv("INDEX_CACHE_DIR", str(ROOT))) / "feedback.jsonl")))
_fb_lock = threading.Lock()

app = FastAPI(title="Юридический ассистент", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def assistant() -> lc.LegalAssistant:
    return lc.build_assistant(KB_PATH)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


class Feedback(BaseModel):
    vote: str = Field(pattern="^(up|down)$")
    question: str = Field(max_length=8000)
    answer: str = Field(max_length=12000)
    status: str = Field(default="", max_length=32)
    sources: list[str] = Field(default_factory=list, max_length=50)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=50)


@app.on_event("startup")
def warmup() -> None:
    try:
        assistant()
    except Exception as exc:  # noqa: BLE001
        lc.log(f"Прогрев не удался: {exc}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    a = assistant()
    return {
        "ok": True,
        "llm": a.client is not None,
        "model": a.s.model,
        "documents": len(a.kb.docs),
        "chunks": len(a.kb.chunks),
        "embeddings": a.kb.emb is not None,
        "reranker": a.s.use_reranker,
        "verifier": a.s.use_verifier,
    }


@app.get("/api/documents")
def documents() -> list[dict]:
    a = assistant()
    return [{"id": d.doc_id, "title": d.title, "kind": d.kind, "status": d.status, "date": d.date}
            for d in a.kb.docs]


@app.get("/api/documents/{doc_id}")
def document(doc_id: str) -> dict:
    a = assistant()
    d = a.kb.by_id.get(doc_id)
    if not d:
        raise HTTPException(404, "Документ не найден")
    chunks = [{"id": c.chunk_id, "label": c.label, "text": c.text} for c in a.kb.chunks if c.doc_id == doc_id]
    return {"id": d.doc_id, "title": d.title, "kind": d.kind, "status": d.status, "date": d.date,
            "filename": d.filename, "chunks": chunks}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    a = assistant()
    msgs = [m.model_dump() for m in req.messages][-MAX_HISTORY:]
    if msgs[-1]["role"] != "user":
        raise HTTPException(400, "Последнее сообщение должно быть от пользователя")
    try:
        if a.client is None:
            raise RuntimeError("не заданы LLM_BASE_URL / LLM_API_KEY")
        res = a.answer(msgs)
    except Exception as exc:  # noqa: BLE001
        lc.log(f"Ошибка ответа, отдаём найденные положения: {exc}")
        res = a.extractive_answer(msgs)
    return {
        "answer": res.answer,
        "status": res.status,
        "sources": res.sources,
        "retrieved": res.retrieved,
        "verified": res.verified,
        "verify_note": res.verify_note,
        "timings": res.timings,
    }


@app.post("/api/feedback")
def feedback(fb: Feedback) -> dict:
    """Оценка ответа юристом — журнал для пилота (доля верных ответов, разбор ошибок)."""
    record = {"ts": dt.datetime.now().isoformat(timespec="seconds"), **fb.model_dump()}
    with _fb_lock:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/api/feedback/stats")
def feedback_stats() -> dict:
    up = down = 0
    if FEEDBACK_PATH.exists():
        for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
            try:
                v = json.loads(line).get("vote")
            except json.JSONDecodeError:
                continue
            up += v == "up"
            down += v == "down"
    total = up + down
    return {"up": up, "down": down, "total": total, "positive_share": round(up / total, 3) if total else None}
