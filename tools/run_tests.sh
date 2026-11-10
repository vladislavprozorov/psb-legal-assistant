#!/usr/bin/env bash
# Все офлайн-проверки одной командой (без ключа). Использование: ./tools/run_tests.sh [python]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${1:-.venv/bin/python}
fail=0
run() { echo; echo "=== $1"; shift; "$@" || fail=1; }

run "Самотест ядра"            env -u LLM_BASE_URL -u LLM_API_KEY "$PY" tools/selftest.py
run "Поиск статей законов"     env -u LLM_BASE_URL -u LLM_API_KEY "$PY" tools/retrieval_check.py
run "Крайние случаи входа"     "$PY" tools/edge_cases.py
run "Хаос-тест прокси"         "$PY" tools/chaos_test.py
run "Сборка ZIP-сабмита"       ./tools/build_submission.sh

echo
if [ $fail -eq 0 ]; then echo "ВСЕ ПРОВЕРКИ ПРОШЛИ"; else echo "ЕСТЬ ПРОВАЛЫ"; fi
exit $fail
