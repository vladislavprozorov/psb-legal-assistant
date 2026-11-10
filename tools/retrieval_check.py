"""Проверка поиска по законам без LLM: попадает ли ожидаемая статья в контекст.

    python tools/retrieval_check.py            # только BM25 (офлайн)
С LLM_BASE_URL/LLM_API_KEY использует также эмбеддинги и реранкер.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import legal_core as lc  # noqa: E402


def main() -> None:
    a = lc.build_assistant(ROOT / "data/knowledge-base.zip")
    rows = list(csv.DictReader(open(ROOT / "tools/extra_questions.csv", encoding="utf-8")))
    rows = [r for r in rows if r["expect_law"]]
    hit = 0
    for r in rows:
        _, used = a.build_context(r["question"])
        docs = []
        for c in used:
            d = a.kb.by_id[c.doc_id]
            if d.kind == lc.LAW and d.filename not in docs:
                docs.append(d.filename)
        ok = any(r["expect_law"] in f for f in docs)
        hit += ok
        short = [f.split("__")[1].replace("_действующая.txt", "") for f in docs]
        print(f"{'OK ' if ok else 'MISS'} {r['id']:>4} ждём {r['expect_law']:<12} получили {short}")
    print(f"\nПопадание статьи в контекст: {hit}/{len(rows)}")
    ctx, _ = a.build_context(rows[0]["question"])
    print(f"Размер контекста для первого вопроса: {len(ctx)} символов")


if __name__ == "__main__":
    main()
