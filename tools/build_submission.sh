#!/usr/bin/env bash
# Собирает ZIP-сабмит для платформы: корневые .py + requirements.txt, без вложенных папок.
#   ./tools/build_submission.sh            → dist/submission.zip
#   ./tools/build_submission.sh noverify   → dist/submission_noverify.zip (самопроверка выключена по умолчанию)
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:-}"
out="dist/submission.zip"
mkdir -p dist
tmp=$(mktemp -d)
cp solution.py legal_core.py "$tmp/"
cp submission/requirements.txt "$tmp/requirements.txt"
if [ "$variant" = "noverify" ]; then
  out="dist/submission_noverify.zip"
  # на платформе нельзя задать переменные окружения, поэтому меняем значение по умолчанию
  python3 - "$tmp/legal_core.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
old = 'env("VERIFY_MODE", "smart")'
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, 'env("VERIFY_MODE", "off")'))
PY
fi
rm -f "$out"
(cd "$tmp" && zip -q -X out.zip solution.py legal_core.py requirements.txt)
mv "$tmp/out.zip" "$out"
rm -rf "$tmp"
unzip -l "$out"
