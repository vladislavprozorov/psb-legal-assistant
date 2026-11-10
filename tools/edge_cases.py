"""Крайние случаи входных данных платформы: python tools/edge_cases.py

Запускает solution.py как отдельный процесс против мок-прокси на разных вариантах
questions.csv и базы знаний и проверяет, что submission.csv всегда корректен.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "data/knowledge-base.zip"
PY = sys.executable
failed = 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_docx(text: str) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in text.splitlines())
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
           f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def kb_variant(kind: str, dst: Path) -> Path:
    if kind == "standard":
        shutil.copy(KB, dst / "knowledge-base.zip")
        return dst / "knowledge-base.zip"
    src = zipfile.ZipFile(KB)
    if kind == "kb.zip+nested":       # другое имя архива и вложенная папка
        out = dst / "kb.zip"
        with zipfile.ZipFile(out, "w") as z:
            for n in src.namelist():
                z.writestr("knowledge-base/" + n, src.read(n))
            z.writestr("__MACOSX/._junk.txt", b"\x00\x01")
        return out
    if kind == "docx+cp1251+noprefix":  # DOCX, cp1251, имена без префикса «законы__»
        out = dst / "knowledge-base.zip"
        with zipfile.ZipFile(out, "w") as z:
            for n in src.namelist():
                text = src.read(n).decode("utf-8-sig")
                name = n.split("__", 1)[-1]
                if "аренды" in n:
                    z.writestr(name.replace(".txt", ".docx"), make_docx(text))
                elif "ТК_ст70" in n:
                    z.writestr(name, text.encode("cp1251", errors="ignore"))
                else:
                    z.writestr(name, text.encode("utf-8"))
            z.writestr("readme.pdf", b"%PDF-1.4 not supported")
        return out
    if kind == "folder":              # база распакована в папку
        out = dst / "knowledge-base"
        src.extractall(out)
        return out
    raise ValueError(kind)


def questions_variant(kind: str, path: Path) -> list[str]:
    rows = [("1", "Какая арендная плата по договору с ООО «Северный Склад»?"),
            ("2", "Пользователь: Гарантия по договору с «Альфой»?\nАссистент: Уточните поставщика.\nПользователь: «Альфа-Трейдинг»."),
            ("3", "Вопрос, с запятыми, \"кавычками\" и ; точкой с запятой"),
            ("4", ""),
            ("5", "Срок испытания главбуха? " * 300)]
    if kind == "bom+crlf+extra":
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, lineterminator="\r\n")
            w.writerow(["id", "question", "category"])
            for r in rows:
                w.writerow([*r, "x"])
        return [r[0] for r in rows]
    if kind == "dup+numeric":
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "question"])
            for r in rows + [("1", "Повтор id")]:
                w.writerow(r)
        return [r[0] for r in rows]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["id", "question"])
        w.writerows(rows)
    return [r[0] for r in rows]


def run_case(name: str, kb_kind: str, q_kind: str, port: int, data_dir_env: bool = True) -> None:
    global failed
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        data = work / "data"
        data.mkdir()
        for f in ("solution.py", "legal_core.py"):
            shutil.copy(ROOT / f, work / f)
        kb_variant(kb_kind, data)
        ids = questions_variant(q_kind, data / "questions.csv")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_", "OPENAI_"))}
        env.update(LLM_BASE_URL=f"http://127.0.0.1:{port}", LLM_API_KEY="x", INDEX_CACHE_DIR=tmp, MAX_WORKERS="3")
        if data_dir_env:
            env["DATA_DIR"] = "data"
        else:
            env.pop("DATA_DIR", None)
        p = subprocess.run([PY, "solution.py"], cwd=work, env=env, capture_output=True, text=True, timeout=300)
        sub = work / "submission.csv"
        problems = []
        if p.returncode != 0:
            problems.append(f"код выхода {p.returncode}: {p.stderr[-300:]}")
        if not sub.exists():
            problems.append("нет submission.csv")
        else:
            with open(sub, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                out = list(reader)
            got = [r[0] for r in out]
            if header != ["id", "answer"]:
                problems.append(f"заголовок {header}")
            if got != ids:
                problems.append(f"id {got} вместо {ids}")
            if any(not r[1].strip() or len(r[1]) > 10000 for r in out):
                problems.append("пустой или слишком длинный ответ")
            if any(len(r) != 2 for r in out):
                problems.append("не 2 колонки")
        docs_line = next((l for l in p.stderr.splitlines() if "База знаний:" in l), "?")
        status = "OK  " if not problems else "FAIL"
        failed += bool(problems)
        print(f"{status} {name:<34} {docs_line.split(': ', 1)[-1]}" + ("" if not problems else " | " + "; ".join(problems)))


def main() -> None:
    port = free_port()
    mock = subprocess.Popen([PY, str(ROOT / "tools/mock_proxy.py"), str(port)])
    time.sleep(1)
    try:
        run_case("стандарт", "standard", "standard", port)
        run_case("BOM, CRLF, лишняя колонка", "standard", "bom+crlf+extra", port)
        run_case("повтор id, числовые id", "standard", "dup+numeric", port)
        run_case("kb.zip, вложенная папка, __MACOSX", "kb.zip+nested", "standard", port)
        run_case("DOCX, cp1251, без префиксов", "docx+cp1251+noprefix", "standard", port)
        run_case("база в папке, без DATA_DIR", "folder", "standard", port, data_dir_env=False)
    finally:
        mock.terminate()
    print("\nВсё прошло" if not failed else f"\nПровалено: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
