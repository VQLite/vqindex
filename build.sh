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

BAZEL_DOWNLOAD_VERSION="${BAZEL_DOWNLOAD_VERSION:-}"
if [ -z "${BAZEL_DOWNLOAD_VERSION}" ] && [ -f ".bazeliskrc" ]; then
    BAZEL_DOWNLOAD_VERSION="$(awk -F= '/^USE_BAZEL_VERSION=/ {print $2; exit}' .bazeliskrc)"
fi
BAZEL_DOWNLOAD_VERSION="${BAZEL_DOWNLOAD_VERSION:-7.6.1}"

bazel_version() {
    "$1" --version 2>/dev/null | awk '{print $NF; exit}'
}

bazel_major_version() {
    bazel_version "$1" | cut -d. -f1
}

is_bazel_7_or_newer() {
    local candidate="$1"
    local major
    major="$(bazel_major_version "${candidate}")"
    [[ "${major}" =~ ^[0-9]+$ ]] && [ "${major}" -ge 7 ]
}

download_bazel() {
    local os arch platform url tmp_path
    os="$(uname -s)"
    arch="$(uname -m)"

    case "${os}" in
        Darwin) platform="darwin" ;;
        Linux) platform="linux" ;;
        *) echo "Only support Linux and MacOS"; exit 1 ;;
    esac

    case "${arch}" in
        x86_64|amd64) arch="x86_64" ;;
        arm64|aarch64) arch="arm64" ;;
        *) echo "Unsupported architecture for Bazel download: ${arch}"; exit 1 ;;
    esac

    command -v curl >/dev/null 2>&1 || {
        echo "curl is required to download Bazel ${BAZEL_DOWNLOAD_VERSION}."
        exit 1
    }

    mkdir -p .tools
    url="https://github.com/bazelbuild/bazel/releases/download/${BAZEL_DOWNLOAD_VERSION}/bazel-${BAZEL_DOWNLOAD_VERSION}-${platform}-${arch}"
    tmp_path=".tools/bazel.${BAZEL_DOWNLOAD_VERSION}.${platform}-${arch}.tmp"

    echo "Downloading Bazel ${BAZEL_DOWNLOAD_VERSION} for ${platform}-${arch}..."
    curl -fL --retry 3 -o "${tmp_path}" "${url}"
    chmod +x "${tmp_path}"
    mv "${tmp_path}" .tools/bazel
}

BAZEL_BIN="${BAZEL_BIN:-}"
if [ -n "${BAZEL_BIN}" ]; then
    if ! is_bazel_7_or_newer "${BAZEL_BIN}"; then
        echo "Bazel 7.x is required by the latest ScaNN, found $(${BAZEL_BIN} --version 2>/dev/null || echo unknown)."
        exit 1
    fi
else
    for candidate in .tools/bazel bazel bazelisk; do
        if command -v "${candidate}" >/dev/null 2>&1 && is_bazel_7_or_newer "${candidate}"; then
            BAZEL_BIN="${candidate}"
            break
        fi
    done

    if [ -z "${BAZEL_BIN}" ]; then
        download_bazel
        BAZEL_BIN=".tools/bazel"
    fi
fi

echo "Using $(${BAZEL_BIN} --version)"

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
