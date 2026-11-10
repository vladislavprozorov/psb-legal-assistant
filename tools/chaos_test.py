"""Хаос-тест: solution.py против прокси, который отвечает 429/500, рассуждениями <think>,
обрезанным JSON, пустыми ответами, а реранкер «не развёрнут». python tools/chaos_test.py"""

from __future__ import annotations

import csv
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def main() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mock = subprocess.Popen([PY, str(ROOT / "tools/mock_proxy.py"), str(port)], env={**os.environ, "MOCK_CHAOS": "1"})
    time.sleep(1)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "data").mkdir()
            for f in ("solution.py", "legal_core.py"):
                shutil.copy(ROOT / f, work / f)
            shutil.copy(ROOT / "data/knowledge-base.zip", work / "data")
            rows = list(csv.DictReader(open(ROOT / "tools/extra_questions.csv", encoding="utf-8")))
            with open(work / "data/questions.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "question"])
                w.writerows([(r["id"], r["question"]) for r in rows])
            env = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_", "OPENAI_"))}
            env.update(LLM_BASE_URL=f"http://127.0.0.1:{port}", LLM_API_KEY="x", VERIFY_MODE="always",
                       MAX_WORKERS="4", INDEX_CACHE_DIR=tmp)
            t = time.time()
            p = subprocess.run([PY, "solution.py"], cwd=work, env=env, capture_output=True, text=True, timeout=600)
            out = list(csv.DictReader(open(work / "submission.csv", encoding="utf-8")))
    finally:
        mock.terminate()
    raw_json = sum(1 for r in out if r["answer"].lstrip().startswith("{") or '"status"' in r["answer"])
    think = sum(1 for r in out if "<think>" in r["answer"])
    model = sum(1 for r in out if r["answer"].startswith("Мок"))
    problems = []
    if p.returncode != 0:
        problems.append(f"код выхода {p.returncode}")
    if len(out) != len(rows):
        problems.append(f"строк {len(out)} из {len(rows)}")
    if raw_json or think:
        problems.append(f"сырой JSON: {raw_json}, <think>: {think}")
    print(f"{len(out)} ответов за {time.time() - t:.0f} c: от модели {model}, остальные — из найденных положений")
    print("OK   устойчивость к сбоям прокси" if not problems else "FAIL " + "; ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
