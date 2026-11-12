#!/bin/bash
# Builds the VQIndex libraries on top of ScaNN (no TensorFlow needed).
#
#   ./build.sh vqindex_py    -> test/vqindex_py.so   (CPython extension)
#   ./build.sh vqindex_api   -> libs/libvqindex_api.so + libs/vqindex_api.h
#
# Environment:
#   BAZEL=/path/to/bazel     Bazel 7.x binary (default: tools/bazel/bazel as
#                            installed by tools/install_bazel.sh, else `bazel`
#                            from PATH).
#   CC=clang                 C/C++ compiler; ScaNN needs clang (default: clang).
#   VQINDEX_COPTS="-mavx"    Baseline instruction set (default: "-mavx", runs on
#                            Sandy Bridge and later). Wider SIMD (AVX2, AVX-512)
#                            is selected at runtime either way; upstream ScaNN
#                            builds with "-mavx -mfma".
#   VQINDEX_ALIGN_BRANCHES=0 Do not pad conditional jumps away from 32-byte
#                            boundaries (see the JCC erratum note below).
#   PYTHON_BIN=python3       Python the build runs with and vqindex_py is built
#                            for (default: the `python3` found in PATH). For
#                            vqindex_py its development headers must be
#                            installed, and the module only imports in that
#                            same Python minor version.
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "build.sh [vqindex_api | vqindex_py]"
    exit 1
fi

case "$(uname -s)" in
    Linux|Darwin) ;;
    *) echo "Only support Linux and MacOS"; exit 1 ;;
esac

cd "$(dirname "$0")"

if [ -z "${BAZEL:-}" ]; then
    if [ -x tools/bazel/bazel ]; then
        BAZEL="${PWD}/tools/bazel/bazel"
    else
        BAZEL=bazel
    fi
fi
export CC="${CC:-clang}"
VQINDEX_COPTS="${VQINDEX_COPTS:--mavx}"

command -v "${BAZEL}" >/dev/null 2>&1 || { echo >&2 "Bazel 7.x is required (run tools/install_bazel.sh or set BAZEL=/path/to/bazel)."; exit 1; }
command -v "${CC}" >/dev/null 2>&1 || { echo >&2 "clang is required (set CC=clang-NN)."; exit 1; }

bazel_major="$("${BAZEL}" --version | awk '{print $NF}' | cut -d. -f1)"
if ! [[ "${bazel_major}" =~ ^[0-9]+$ ]] || [ "${bazel_major}" -lt 7 ]; then
    echo >&2 "Bazel 7.x is required, found: $("${BAZEL}" --version)"
    exit 1
fi

copts=()
for opt in ${VQINDEX_COPTS}; do
    copts+=("--copt=${opt}")
done

# Upstream ScaNN targets clang 19, where relaxed template template argument
# matching (P0522) is on by default; older clang needs it switched on.
cc_major="$("${CC}" -dumpversion 2>/dev/null | cut -d. -f1)"
if "${CC}" --version 2>/dev/null | grep -qi clang && [[ "${cc_major}" =~ ^[0-9]+$ ]] \
    && [ "${cc_major}" -lt 19 ]; then
    copts+=("--cxxopt=-frelaxed-template-template-args")
fi

# Intel's microcode fix for the Skylake-family "JCC erratum" keeps a conditional
# jump that crosses or ends on a 32-byte boundary out of the uop cache.  Whether
# the back edge of the LUT16 scan loop lands there is an accident of code
# layout (it does with clang 18), and it costs several percent of search
# throughput, so have the assembler pad such jumps.  With ThinLTO the code is
# generated at link time, hence the linker option as well.
linkopts=()
if "${CC}" --version 2>/dev/null | grep -qi clang && [ "${VQINDEX_ALIGN_BRANCHES:-1}" = "1" ] \
    && [ "$(uname -m)" = "x86_64" ]; then
    copts+=("--copt=-mbranches-within-32B-boundaries")
    linkopts+=("--linkopt=-Wl,-mllvm,-x86-branches-within-32B-boundaries")
fi

bazel_build() {
    "${BAZEL}" build -c opt --features=thin_lto "${copts[@]}" "${linkopts[@]}" --cxxopt="-std=c++17" \
        --copt=-fsized-deallocation --copt=-w "$@"
}

# MODULE.bazel locates the Python interpreter (and its headers) as `python` in
# PATH, which many systems do not provide; point it at PYTHON_BIN.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo >&2 "${PYTHON_BIN} not found (set PYTHON_BIN=/path/to/python3)."
    exit 1
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
PYTHON_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)/$(basename "${PYTHON_BIN}")"
mkdir -p .build_python
ln -sfn "${PYTHON_BIN}" .build_python/python
export PATH="${PWD}/.build_python:${PATH}"

case "$1" in
vqindex_py)
    py_include="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
    if [ ! -f "${py_include}/Python.h" ]; then
        echo >&2 "${py_include}/Python.h not found; install the Python development headers (e.g. python3-dev)."
        exit 1
    fi
    bazel_build //scann/scann_ops/cc:vqindex_py
    mkdir -p test
    cp -f bazel-bin/scann/scann_ops/cc/libvqindex_py.so test/vqindex_py.so
    echo "Build suss, you can use test/run_tests.sh to test."
    ;;
vqindex_api)
    bazel_build //scann/scann_ops/cc:vqindex_api
    mkdir -p libs
    cp -f scann/scann_ops/cc/vqindex_api.h libs/vqindex_api.h
    cp -f bazel-bin/scann/scann_ops/cc/libvqindex_api.so libs/libvqindex_api.so
    echo "Build libs/libvqindex_api.so suss."
    ;;
*)
    echo "build.sh [vqindex_api | vqindex_py]"
    exit 1
    ;;
esac
