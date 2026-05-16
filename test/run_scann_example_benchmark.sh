#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ "${VQINDEX_SKIP_BUILD:-0}" != "1" ]]; then
  ./build.sh vqindex_api
fi

python_bin="${PYTHON_BIN:-python3}"
if "${python_bin}" - <<'PY' >/dev/null 2>&1
import h5py
import numpy
import requests
PY
then
  exec "${python_bin}" test/run_scann_example_benchmark.py "$@"
fi

if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  elif [[ -x "${HOME}/.cargo/bin/uv" ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Python deps are missing and uv was not found; installing uv..." >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

venv_dir="${VQINDEX_SCANN_EXAMPLE_VENV:-${repo_root}/.tools/scann-example-venv}"
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  uv venv "${venv_dir}" >/dev/null
fi
uv pip install --python "${venv_dir}/bin/python" numpy h5py requests >/dev/null

exec "${venv_dir}/bin/python" test/run_scann_example_benchmark.py "$@"
