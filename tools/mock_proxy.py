"""Мок OpenAI-совместимого прокси платформы для офлайн-проверки пайплайна.

Отвечает на /chat/completions, /embeddings и /rerank детерминированными
данными. Качество ответов не проверяет — только формат, ошибки и тайминги.
Запуск: python tools/mock_proxy.py 8765
"""

from __future__ import annotations

import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def vec(text: str, dim: int = 64) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # тишина
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path.endswith("/chat/completions"):
            last = body["messages"][-1]["content"]
            if "ЧЕРНОВИК ОТВЕТА" in last:
                content = json.dumps({"ok": True, "issues": "", "answer": ""}, ensure_ascii=False)
            else:
                content = json.dumps({"status": "answer", "answer": "Мок-ответ: «" + last[:60] + "», п. 3.1.",
                                      "sources": ["D1.10"]}, ensure_ascii=False)
            out = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        elif self.path.endswith("/embeddings"):
            inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
            out = {"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(inp)]}
        elif self.path.endswith("/rerank"):
            n = min(int(body.get("top_n", 5)), len(body["documents"]))
            out = {"results": [{"index": i, "relevance_score": 1 - i / 10} for i in range(n)]}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(out, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
