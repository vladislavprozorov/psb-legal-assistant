"""Проверка подключения к модели — запустить первым делом, когда появится ключ.

    python tools/check_llm.py            # берёт LLM_BASE_URL / LLM_API_KEY из окружения или .env
    python tools/check_llm.py --full     # плюс прогон 3 вопросов через весь пайплайн

Проверяет /chat/completions, /embeddings, /rerank, формат JSON-ответа и задержки,
оценивает время прогона 100 вопросов. Ключ на экран не выводится.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import legal_core as lc  # noqa: E402

ok_all = True


def step(name: str, fn):
    global ok_all
    t = time.time()
    try:
        info = fn()
        print(f"OK    {name:<28} {time.time() - t:5.1f} c  {info or ''}")
        return True
    except Exception as exc:  # noqa: BLE001
        ok_all = False
        print(f"FAIL  {name:<28} {time.time() - t:5.1f} c  {type(exc).__name__}: {str(exc)[:200]}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    s = lc.Settings()
    if not s.api_base or not s.api_key:
        sys.exit("Нет LLM_BASE_URL / LLM_API_KEY: заполните .env (см. .env.example)")
    print(f"Endpoint: {urlparse(s.api_base).netloc} | модель: {s.model} | эмбеддер: {s.emb_model} | реранкер: {s.reranker_model}\n")
    c = lc.LLMClient(s)

    def chat():
        out = c.chat([{"role": "user", "content": 'Верни JSON {"answer": "пинг"} без пояснений.'}], max_tokens=200)
        parsed = lc.parse_json(out)
        if parsed.get("answer") is None:
            raise RuntimeError(f"не JSON: {out[:120]!r}")
        return f"JSON разобран: {parsed}"

    def emb():
        v = c.embed(["договор аренды", "арендная плата"])
        return f"размерность {v.shape[1]}, сходство пары {float(v[0] @ v[1]):.2f}"

    def rerank():
        r = c.rerank("срок испытания главного бухгалтера",
                     ["Арендная плата вносится до 5-го числа.", "Срок испытания не может превышать шести месяцев для главных бухгалтеров."], 2)
        if not r or r[0][0] != 1:
            raise RuntimeError(f"неожиданный порядок {r}")
        return f"порядок верный {r}"

    step("генерация /chat/completions", chat)
    emb_ok = step("эмбеддинги /embeddings", emb) if s.use_embeddings else print("SKIP  эмбеддинги (USE_EMBEDDINGS=0) — поиск на BM25") or True
    rr_ok = step("реранкер /rerank", rerank) if s.use_reranker else print("SKIP  реранкер (USE_RERANKER=0)") or True
    if not emb_ok:
        print("      → без эмбеддингов поиск работает на BM25 (USE_EMBEDDINGS=0 уберёт попытки)")
    if not rr_ok:
        print("      → без реранкера используется гибридный порядок (USE_RERANKER=0 уберёт попытки); проверьте RERANKER_MODEL")

    if args.full:
        a = lc.build_assistant(ROOT / "data/knowledge-base.zip", s)
        times = []
        for q in ["Какая арендная плата по договору с ООО «Северный Склад» и до какого числа её нужно вносить?",
                  "Договор на 12 млн рублей заключали в августе 2024 года. Нужно ли было одобрение генерального директора?",
                  "Какой размер пособия по беременности и родам?"]:
            def run(q=q):
                r = a.answer(q)
                times.append(r.timings["total"])
                return f"[{r.status}{', проверен' if r.verified else ''}] {r.answer[:150]}"
            step("вопрос: " + q[:24] + "…", run)
        if times:
            avg = sum(times) / len(times)
            workers = lc.env_int("MAX_WORKERS", 4, 1, 16)
            est = 100 * avg / workers
            print(f"\nСреднее время ответа {avg:.1f} c; 100 вопросов при MAX_WORKERS={workers} ≈ {est / 60:.1f} мин "
                  f"({'в лимите' if est < 900 else 'РИСК тайм-аута: уменьшите VERIFY_MODE или контекст'})")
    print("\nГотово к работе" if ok_all else "\nЕсть проблемы — см. FAIL выше")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
