"""Точка входа для автоматической проверки на платформе Hack&Change 2026 (кейс ПСБ).

Читает DATA_DIR/questions.csv и базу знаний (knowledge-base.zip / kb.zip),
отвечает на каждый вопрос через legal_core.LegalAssistant и пишет submission.csv
(id,answer) в рабочий каталог. Укладывается в лимит времени: при приближении
к тайм-ауту отключает самопроверку, а неуспевшие вопросы получают корректный
ответ-заглушку, чтобы файл всегда был полным.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import legal_core as lc

START = time.time()
TIME_LIMIT = lc.env_int("TIME_BUDGET_SEC", 20 * 60, 60, 3 * 3600)
SAFETY = 90           # секунд запаса на запись файла и завершение
VERIFY_CUTOFF = 0.55  # после этой доли бюджета самопроверка отключается
FALLBACK = ("Не удалось сформировать проверенный ответ по базе знаний. "
            "Уточните вопрос: укажите документ, контрагента или период.")


def find_file(names: list[str]) -> Path:
    data_dir = Path(lc.env("DATA_DIR", "data"))
    roots = [data_dir, Path(__file__).resolve().parent / data_dir, Path.cwd(), Path(__file__).resolve().parent]
    for root in roots:
        for name in names:
            p = root / name
            if p.exists():
                return p
    raise FileNotFoundError("Не найден ни один из файлов: " + ", ".join(names))


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [(str(r["id"]).strip(), str(r.get("question") or "").strip()) for r in reader]
    return [(qid, q) for qid, q in rows if qid]


def write_submission(rows: list[tuple[str, str]], answers: dict[str, str]) -> None:
    seen = set()
    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writerow(["id", "answer"])
        for qid, _ in rows:
            if qid in seen:
                continue
            seen.add(qid)
            ans = (answers.get(qid) or FALLBACK).strip()[:9800]
            w.writerow([qid, ans])


def main() -> None:
    settings = lc.Settings()
    workers = lc.env_int("MAX_WORKERS", 4, 1, 16)
    questions = load_questions(find_file(["questions.csv"]))
    kb_path = find_file(["knowledge-base.zip", "kb.zip", "knowledge_base.zip", "knowledge-base", "kb"])
    lc.log(f"Вопросов: {len(questions)}; база: {kb_path}; модель: {settings.model}; потоков: {workers}")

    assistant = lc.build_assistant(kb_path, settings)
    if assistant.client is None:
        raise SystemExit("Нет LLM_BASE_URL / LLM_API_KEY")

    answers: dict[str, str] = {}
    deadline = START + TIME_LIMIT - SAFETY

    def work(qid: str, question: str) -> tuple[str, str]:
        if time.time() > START + TIME_LIMIT * VERIFY_CUTOFF and settings.use_verifier:
            settings.use_verifier = False
            lc.log("Бюджет времени: самопроверка отключена для оставшихся вопросов")
        for attempt in range(2):
            if time.time() > deadline:
                break
            try:
                res = assistant.answer(question)
                if res.answer.strip():
                    return qid, res.answer
            except Exception as exc:  # noqa: BLE001
                lc.log(f"Вопрос {qid}, попытка {attempt + 1}: {type(exc).__name__}: {str(exc)[:200]}")
                time.sleep(2)
        return qid, FALLBACK

    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {pool.submit(work, qid, q) for qid, q in questions}
    done_count = 0
    while pending:
        timeout = deadline - time.time()
        if timeout <= 0:
            lc.log(f"Тайм-аут: без ответа осталось {len(pending)} вопросов")
            break
        finished, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
        for fut in finished:
            qid, ans = fut.result()
            answers[qid] = ans
            done_count += 1
        if done_count % 10 == 0 or not pending:
            lc.log(f"Готово {done_count}/{len(questions)} за {time.time() - START:.0f} c")
        write_submission(questions, answers)  # промежуточное сохранение
    write_submission(questions, answers)
    pool.shutdown(wait=False, cancel_futures=True)
    lc.log(f"submission.csv: {len(questions)} строк, {time.time() - START:.0f} c")
    os._exit(0)  # не ждём зависших HTTP-запросов после записи файла


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        lc.log(f"Критическая ошибка: {type(exc).__name__}: {exc}")
        # даже при сбое создаём корректный файл, чтобы сабмит не получил 0
        try:
            qs = load_questions(find_file(["questions.csv"]))
            write_submission(qs, {})
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
