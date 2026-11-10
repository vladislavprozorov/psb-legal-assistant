"""Офлайн-самотест ядра (без LLM): python tools/selftest.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for k in ("LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(k, None)

import legal_core as lc  # noqa: E402

failed = 0


def check(name: str, cond: bool) -> None:
    global failed
    print(("OK   " if cond else "FAIL ") + name)
    failed += not cond


a = lc.build_assistant(ROOT / "data/knowledge-base.zip")
kb = a.kb
titles = {d.filename: d for d in kb.docs}

# загрузка и санитизация
check("21 документ", len(kb.docs) == 21)
memo = next(d for d in kb.docs if "архиве" in d.filename)
check("инструкция для ИИ вырезана из служебной записки", memo.removed_lines == 1 and "90 календарных" not in memo.text)
check("полезный текст записки сохранён", "не более 3 (трёх) рабочих дней" in memo.text)

# метаданные
old = next(d for d in kb.docs if "ред 2023" in d.filename)
new = next(d for d in kb.docs if "ред 2025" in d.filename)
check("регламент 2023 — утратил силу", old.status == "утратил силу")
check("регламент 2025 — действует, редакция от 01.03.2025", new.status == "действует" and "01.03.2025" in new.date)
ds = next(d for d in kb.docs if "Доп соглашение" in d.filename)
check("дата доп. соглашения — 3 сентября 2024", "3 сентября 2024" in ds.date)
tk59 = next(d for d in kb.docs if "ТК_ст59" in d.filename)
check("ТК ст.59: найдено изменение с 01.03.2027", any(e[0] == "2027-03-01" for e in tk59.editions))
mrot = next(d for d in kb.docs if "МРОТ" in d.filename)
check("название статьи МРОТ без текста нормы", "27 093" not in mrot.title)

# разбиение
rent = [c for c in kb.chunks if c.doc_id == next(d for d in kb.docs if "аренды" in d.filename).doc_id]
check("договор аренды: есть п. 3.1 с суммой", any(c.label == "п. 3.1" and "450 000" in c.text for c in rent))
check("метки законов без «часть N»", not any(c.label.startswith("часть") for c in kb.chunks))

# диалог
msgs = lc.parse_dialog("Пользователь: Привет\nАссистент: Уточните договор\nПользователь: А-08/2024. Акт нужен?")
check("диалог: 3 реплики, последняя — пользователь",
      [m["role"] for m in msgs] == ["user", "assistant", "user"] and msgs[-1]["content"].startswith("А-08"))
check("одиночный вопрос", lc.parse_dialog("Какая ставка НДС?") == [{"role": "user", "content": "Какая ставка НДС?"}])

# JSON и проверки
check("JSON в markdown-обёртке", lc.parse_json('```json\n{"answer": "да", "status": "answer"}\n```')["answer"] == "да")
check("JSON с текстом вокруг", lc.parse_json('Вот: {"answer": "нет"} конец')["answer"] == "нет")
check("подозрительное число найдено", lc.unsupported_numbers("плата 500 000 рублей", "плата 450 000 рублей") == ["500 000"])
check("подтверждённое число не помечено", lc.unsupported_numbers("плата 450 000 рублей, п. 3.1", "п. 3.1 плата 450 000") == [])
check("clean_answer убирает id фрагментов", "D1.2" not in lc.clean_answer("Ответ: да [D1.2]"))

# контекст
ctx, used = a.build_context("Какой максимальный срок испытания для главного бухгалтера?")
check("контекст: реестр + все документы компании + ТК ст.70", "РЕЕСТР" in ctx and "А-08/2024" in ctx and "шести месяцев" in ctx)
check("контекст в разумных пределах", len(ctx) < 70000)
check("в контексте нет внедрённой инструкции", "90 календарных дней" not in ctx)

# аудит ссылок
import csv  # noqa: E402

qs = {r["id"]: r["question"] for r in csv.DictReader(open(ROOT / "data/raw/demo_questions.csv", encoding="utf-8-sig"))}
gold = list(csv.DictReader(open(ROOT / "data/raw/demo_output.csv", encoding="utf-8-sig")))
check("аудит: 0 ложных срабатываний на 18 эталонах", not any(lc.audit_answer(r["answer"], kb, qs[r["id"]]) for r in gold))
check("аудит: выдуманная статья", bool(lc.audit_answer("Срок до 6 месяцев (ТК РФ, ст. 72).", kb, "")))
check("аудит: выдуманный пункт договора", bool(lc.audit_answer("Оплата 90 дней (договор № П-17/2024, п. 4.7).", kb, "")))
check("аудит: пункт из доп. соглашения допустим", not lc.audit_answer("Адрес изменён (договор № П-17/2024, п. 3.2).", kb, ""))
check("аудит: выдуманный номер договора", bool(lc.audit_answer("По договору № А-09/2024 плата 450 000 рублей.", kb, "")))
check("аудит: оговорка «отсутствует» допустима", not lc.audit_answer("Статья 6 152-ФЗ в базе отсутствует.", kb, ""))
check("smart: простой ответ без самопроверки", not lc.needs_verification("smart", "answer", "Какая плата?", "450 000 (п. 3.1)", [], []))
check("smart: вопрос с датой — самопроверка", lc.needs_verification("smart", "answer", "Договор в августе 2024", "да", [], []))
fb = a.extractive_answer("Какая арендная плата по договору с «Северным Складом»?")
check("запасной ответ без LLM содержит положение договора", fb.status == "fallback" and "Арендн" in fb.answer)

print(f"\n{'Всё прошло' if not failed else f'Провалено: {failed}'}")
sys.exit(1 if failed else 0)
