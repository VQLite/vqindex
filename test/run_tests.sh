#!/bin/bash
# Runs the VQIndex Python tests against test/vqindex_py.so.
#
#   test/run_tests.sh            # test.py + test_add.py + test_concurrency.py
#                                # + test_format.py
#   test/run_tests.sh --bench    # additionally the SIMD level benchmark
#
# Environment:
#   PYTHON_BIN=python3   Must be the Python version vqindex_py.so was built
#                        for, with numpy installed.
#   VQINDEX_TEST_DIR     Scratch directory for index files (default: a fresh
#                        directory under $TMPDIR, removed afterwards). The
#                        tests write about 500 MB.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

[ -f "${here}/vqindex_py.so" ] || { echo >&2 "test/vqindex_py.so not found; run ./build.sh vqindex_py first."; exit 1; }
"${PYTHON_BIN}" -c "import numpy" 2>/dev/null || { echo >&2 "numpy is required: ${PYTHON_BIN} -m pip install numpy"; exit 1; }

if [ -n "${VQINDEX_TEST_DIR:-}" ]; then
    work="${VQINDEX_TEST_DIR}"
    mkdir -p "${work}"
else
    work="$(mktemp -d "${TMPDIR:-/tmp}/vqindex_test.XXXXXX")"
    trap 'rm -rf "${work}"' EXIT
fi

echo "=== test.py"
mkdir -p "${work}/basic"
PYTHONPATH="${here}" "${PYTHON_BIN}" "${here}/test.py" "${work}/basic" 2>&1 | cut -c1-300

echo "=== test_add.py"
"${PYTHON_BIN}" "${here}/test_add.py" "${here}" "${work}"

echo "=== test_concurrency.py"
"${PYTHON_BIN}" "${here}/test_concurrency.py" "${here}" "${work}"

echo "=== test_format.py"
"${PYTHON_BIN}" "${here}/test_format.py" "${here}" "${work}" "${here}/../tools/vqindex_convert.py"

if [ "${1:-}" = "--bench" ]; then
    echo "=== bench_simd.py"
    "${PYTHON_BIN}" "${here}/bench_simd.py" "${here}" "${work}"
fi
echo "ALL TESTS PASSED"
