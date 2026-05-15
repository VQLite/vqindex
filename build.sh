#!/bin/bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "build.sh [vqindex_api | vqindex_py]"
    exit 1
fi

if [[ "$(uname -s)" != "Darwin" && "$(uname -s)" != "Linux" ]]; then
    echo "Only support Linux and MacOS"
    exit 1
fi

BAZEL_BIN="${BAZEL_BIN:-}"
if [ -z "${BAZEL_BIN}" ]; then
    if [ -x ".tools/bazel" ]; then
        BAZEL_BIN=".tools/bazel"
    elif command -v bazel >/dev/null 2>&1; then
        BAZEL_BIN="bazel"
    elif command -v bazelisk >/dev/null 2>&1; then
        BAZEL_BIN="bazelisk"
    else
        echo "Bazel 7.x is required by the latest ScaNN. Please install bazel or bazelisk, or set BAZEL_BIN."
        exit 1
    fi
fi

BAZEL_VERSION="$(${BAZEL_BIN} --version | awk '{print $2}')"
BAZEL_MAJOR="${BAZEL_VERSION%%.*}"
if [ "${BAZEL_MAJOR}" -lt 7 ]; then
    echo "Bazel 7.x is required by the latest ScaNN, found ${BAZEL_VERSION}."
    exit 1
fi

CC_BIN="${CC:-clang}"
command -v "${CC_BIN}" >/dev/null 2>&1 || { echo >&2 "I require ${CC_BIN} but it's not installed. Aborting."; exit 1; }

PYTHON_BIN="${PYTHON_BIN_PATH:-$(command -v python3)}"

ARCH="$(uname -m)"
ARCH_COPTS=()
if [[ "${ARCH}" == "x86_64" || "${ARCH}" == "amd64" ]]; then
    ARCH_COPTS=(--copt=-mavx2 --copt=-mfma)
else
    ARCH_COPTS=(--copt=-march=armv8-a+simd)
fi

COMMON_ARGS=(
    build
    -c opt
    --features=thin_lto
    "${ARCH_COPTS[@]}"
    --cxxopt=-std=c++17
    --copt=-fsized-deallocation
    --copt=-w
)

if [ "$1" = "vqindex_py" ]; then
    CC="${CC_BIN}" "${BAZEL_BIN}" "${COMMON_ARGS[@]}" //scann/scann_ops/cc:vqindex_py --action_env "PYTHON_BIN_PATH=${PYTHON_BIN}"
    if [[ "$(uname -s)" == "Darwin" ]]; then
        py_lib="bazel-bin/scann/scann_ops/cc/libvqindex_py.dylib"
    else
        py_lib="bazel-bin/scann/scann_ops/cc/libvqindex_py.so"
    fi
    cp -rf "${py_lib}" test/vqindex_py.so
    echo "Build success, you can use test/test.py to test."
    exit 0
fi

if [ "$1" = "vqindex_api" ]; then
    mkdir -p libs
    CC="${CC_BIN}" "${BAZEL_BIN}" "${COMMON_ARGS[@]}" //scann/scann_ops/cc:vqindex_api
    if [[ "$(uname -s)" == "Darwin" ]]; then
        api_lib="bazel-bin/scann/scann_ops/cc/libvqindex_api.dylib"
    else
        api_lib="bazel-bin/scann/scann_ops/cc/libvqindex_api.so"
    fi
    cp -rf scann/scann_ops/cc/vqindex_api.h libs/vqindex_api.h
    cp -rf "${api_lib}" libs/
    echo "Build ${api_lib} success."
    exit 0
fi

echo "Unknown target: $1"
echo "build.sh [vqindex_api | vqindex_py]"
exit 1
