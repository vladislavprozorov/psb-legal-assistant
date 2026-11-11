"""Мок OpenAI-совместимого прокси платформы для офлайн-проверки пайплайна.

Отвечает на /chat/completions, /embeddings и /rerank детерминированными
данными. Качество ответов не проверяет — только формат, ошибки и тайминги.

    python tools/mock_proxy.py 8765
    MOCK_DELAY=4 python tools/mock_proxy.py 8765     # медленная модель
    MOCK_CHAOS=1 python tools/mock_proxy.py 8765     # сбои и «грязные» форматы ответов
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHAOS = os.environ.get("MOCK_CHAOS") == "1"
DELAY = float(os.environ.get("MOCK_DELAY", "0"))
# Имитация reasoning-модели: рассуждения занимают столько токенов и входят в max_tokens
REASONING = int(os.environ.get("MOCK_REASONING", "0"))
_counter = itertools.count()
_lock = threading.Lock()
STATS: dict[str, int] = {}

# Сценарии хаоса для /chat/completions (по кругу, чтобы прогон был воспроизводимым)
CHAOS_CHAT = ["ok", "fenced", "think", "http429", "plain", "ok", "truncated",
              "empty_length", "http500", "json_no_answer", "ok", "parts"]


def vec(text: str, dim: int = 64) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]


def answer_json(last: str) -> str:
    return json.dumps({"status": "answer",
                       "answer": "Мок-ответ: «" + last[:60] + "». Арендная плата — 450 000 рублей в месяц "
                                 "(договор № А-08/2024, п. 3.1), вносится до 5-го числа (п. 3.2).",
                       "sources": ["D1.9", "D1.10"]}, ensure_ascii=False)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # тишина
        pass

    def _send(self, code: int, obj: dict | None = None) -> None:
        data = json.dumps(obj or {"error": {"message": "mock error"}}, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with _lock:
            n = next(_counter)
        if self.path.endswith("/chat/completions"):
            time.sleep(DELAY)
            last = body["messages"][-1]["content"]
            verify = "ЧЕРНОВИК ОТВЕТА" in last
            mode = CHAOS_CHAT[n % len(CHAOS_CHAT)] if CHAOS else "ok"
            STATS[mode] = STATS.get(mode, 0) + 1
            if mode == "http429":
                return self._send(429)
            if mode == "http500":
                return self._send(500)
            finish = "stop"
            budget = int(body.get("max_tokens", 0))
            if REASONING and budget < REASONING:
                return self._send(200, {"choices": [{"message": {"role": "assistant", "content": "",
                                        "reasoning_content": "думаю..." * 50}, "finish_reason": "length"}]})
            if REASONING and budget < REASONING + 300 and not verify:
                return self._send(200, {"choices": [{"message": {"role": "assistant",
                                        "content": '{"status": "answer", "answer": "Арендная плата по договору № А-08/2024 сост',
                                        "reasoning_content": "думаю..."}, "finish_reason": "length"}]})
            if verify:
                content = json.dumps({"ok": True, "issues": "", "answer": ""}, ensure_ascii=False)
            elif mode == "fenced":
                content = "```json\n" + answer_json(last) + "\n```"
            elif mode == "think":
                content = "<think>Рассуждаю о вопросе…</think>\n" + answer_json(last)
            elif mode == "plain":
                content = "Мок-ответ простым текстом без JSON, п. 3.1."
            elif mode == "truncated":
                content = '{"status": "answer", "answer": "Мок-ответ оборван на полу'
                finish = "length"
            elif mode == "empty_length" and body.get("max_tokens", 0) < 3000:
                content, finish = "", "length"
            elif mode == "json_no_answer":
                content = '{"status": "answer", "sources": []}'
            elif mode == "parts":
                content = [{"type": "text", "text": answer_json(last)}]
            else:
                content = answer_json(last)
            return self._send(200, {"choices": [{"message": {"role": "assistant", "content": content},
                                                 "finish_reason": finish}]})
        if self.path.endswith("/embeddings"):
            if CHAOS and n % 7 == 3:
                return self._send(503)
            inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self._send(200, {"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(inp)]})
        if self.path.endswith("/rerank"):
            if CHAOS:
                return self._send(404)  # реранкер «не развёрнут» — должен сработать откат
            k = min(int(body.get("top_n", 5)), len(body["documents"]))
            return self._send(200, {"results": [{"index": i, "relevance_score": 1 - i / 10} for i in range(k)]})
        self._send(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
