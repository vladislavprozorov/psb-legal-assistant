"""Локальная оценка на демо-вопросах: ответ ассистента против эталона из demo_output.csv.

Запуск (нужны LLM_BASE_URL / LLM_API_KEY в окружении или .env):
    python tools/eval_demo.py [--judge] [--ids 1,5,17]

--judge  — дополнительно попросить ту же LLM оценить ответ относительно эталона
           по шкале как у судьи платформы (0 / 0.25 / 0.5 / 0.75 / 1, выдумка = 0).
Отчёт сохраняется в dist/eval_report.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import legal_core as lc  # noqa: E402

JUDGE = """Ты — строгий судья ответов юридического ассистента. Сравни ОТВЕТ с ЭТАЛОНОМ.
Критерии: фактическая точность (числа, сроки, пункты), полнота относительно эталона, корректный отказ/уточнение, отсутствие выдуманных фактов.
Если в ответе есть выдуманный факт или норма, которых нет в эталоне и которые противоречат ему, — оценка 0.
Верни JSON: {"score": 0|0.25|0.5|0.75|1, "reason": "<одно предложение>"}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--ids", default="")
    ap.add_argument("--questions", default=str(ROOT / "data/raw/demo_questions.csv"))
    ap.add_argument("--golden", default=str(ROOT / "data/raw/demo_output.csv"))
    args = ap.parse_args()

    qs = list(csv.DictReader(open(args.questions, encoding="utf-8-sig")))
    gold = {r["id"]: r["answer"] for r in csv.DictReader(open(args.golden, encoding="utf-8-sig"))}
    if args.ids:
        keep = set(args.ids.split(","))
        qs = [q for q in qs if q["id"] in keep]

    a = lc.build_assistant(ROOT / "data/knowledge-base.zip")
    if a.client is None:
        sys.exit("Задайте LLM_BASE_URL и LLM_API_KEY")

    def run(q):
        t = time.time()
        try:
            res = a.answer(q["question"])
            ans, status, note = res.answer, res.status, res.verify_note
        except Exception as exc:  # noqa: BLE001
            ans, status, note = f"ОШИБКА: {exc}", "error", ""
        score, reason = None, ""
        if args.judge and q["id"] in gold:
            j = lc.parse_json(a.client.chat([
                {"role": "system", "content": JUDGE},
                {"role": "user", "content": f"ВОПРОС:\n{q['question']}\n\nЭТАЛОН:\n{gold[q['id']]}\n\nОТВЕТ:\n{ans}"},
            ], max_tokens=300, temperature=0))
            score, reason = j.get("score"), j.get("reason", "")
        return q, ans, status, note, score, reason, time.time() - t

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, qs))

    lines = ["# Оценка на демо-вопросах\n"]
    scores = []
    for q, ans, status, note, score, reason, sec in results:
        if isinstance(score, (int, float)):
            scores.append(float(score))
        lines += [f"## {q['id']}. {q['question']}", "",
                  f"**Ответ** ({status}, {sec:.1f} c): {ans}", "",
                  f"**Эталон:** {gold.get(q['id'], '—')}", ""]
        if note:
            lines += [f"_Самопроверка:_ {note}", ""]
        if score is not None:
            lines += [f"**Оценка судьи:** {score} — {reason}", ""]
        print(f"[{q['id']}] {status} {score if score is not None else ''} | {ans[:140]}")
    if scores:
        summary = f"Средняя оценка: {sum(scores) / len(scores):.3f} на {len(scores)} вопросах"
        lines.insert(1, summary + "\n")
        print(summary)
    out = ROOT / "dist" / "eval_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Отчёт:", out)


if __name__ == "__main__":
    main()
