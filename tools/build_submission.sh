#!/usr/bin/env bash
# Собирает ZIP-сабмит для платформы: корневые .py + requirements.txt, без вложенных папок.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/submission.zip
tmp=$(mktemp -d)
cp solution.py legal_core.py "$tmp/"
cp submission/requirements.txt "$tmp/requirements.txt"
(cd "$tmp" && zip -q -X ../submission.zip solution.py legal_core.py requirements.txt && mv ../submission.zip "$OLDPWD/dist/submission.zip")
rm -rf "$tmp"
unzip -l dist/submission.zip
