#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
build.sh [vqindex_api | vqindex_py]

Environment:
  BAZEL_BIN=/path/to/bazel             Use an explicit Bazel 7.x binary.
  BAZEL_DOWNLOAD_VERSION=7.6.1         Bazel version to download when missing.
  VQINDEX_CPU_OPT=native|portable      native squeezes the current machine; portable
                                       avoids machine-specific baseline code.
  VQINDEX_HWY_TARGETS=all|static       all enables Highway multi-target dispatch.
  VQINDEX_EXTRA_COPTS="..."            Extra compiler flags appended to Bazel.
EOF
}

if [ $# -eq 0 ]; then
    usage
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
OS_NAME="$(uname -s)"
CPU_OPT_MODE="${VQINDEX_CPU_OPT:-native}"
HWY_TARGET_MODE="${VQINDEX_HWY_TARGETS:-all}"

compile_flag_supported() {
    local flag="$1"
    local output
    output="$(mktemp "${TMPDIR:-/tmp}/vqindex-flag-test.XXXXXX")"
    printf 'int main() { return 0; }\n' | "${CC_BIN}" -x c++ -c -o "${output}" "${flag}" - >/dev/null 2>&1
    local result=$?
    rm -f "${output}"
    return "${result}"
}

append_copt_if_supported() {
    local flag="$1"
    if compile_flag_supported "${flag}"; then
        ARCH_COPTS+=("--copt=${flag}")
        ARCH_CXXOPTS+=("--cxxopt=${flag}")
    fi
}

cpu_flags() {
    if [[ "${OS_NAME}" == "Linux" && -r /proc/cpuinfo ]]; then
        awk -F: '/flags|Features/ {print tolower($2); exit}' /proc/cpuinfo
    elif [[ "${OS_NAME}" == "Darwin" ]]; then
        {
            sysctl -n machdep.cpu.features 2>/dev/null || true
            sysctl -n machdep.cpu.leaf7_features 2>/dev/null || true
        } | tr '\n' ' ' | tr '[:upper:]' '[:lower:]'
    fi
}

CPU_FLAGS="$(cpu_flags)"

has_cpu_flag() {
    local flag="$1"
    [[ " ${CPU_FLAGS} " == *" ${flag} "* ]]
}

append_cpu_flag_if_supported() {
    local cpu_flag="$1"
    local compiler_flag="$2"
    if has_cpu_flag "${cpu_flag}"; then
        append_copt_if_supported "${compiler_flag}"
    fi
}

ARCH_COPTS=()
ARCH_CXXOPTS=()
case "${CPU_OPT_MODE}" in
    native|portable) ;;
    *)
        echo "Unknown VQINDEX_CPU_OPT=${CPU_OPT_MODE}; use native or portable."
        exit 1
        ;;
esac

if [[ "${CPU_OPT_MODE}" == "native" ]]; then
    if [[ "${ARCH}" == "x86_64" || "${ARCH}" == "amd64" ]]; then
        append_copt_if_supported "-march=native"
        append_copt_if_supported "-mtune=native"

        # -march=native is the main switch.  The feature flags below make the
        # selected SIMD set explicit for compilers/toolchains that do not expand
        # every extension we care about from -march=native.
        append_cpu_flag_if_supported "sse4_1" "-msse4.1"
        append_cpu_flag_if_supported "sse4_2" "-msse4.2"
        append_cpu_flag_if_supported "popcnt" "-mpopcnt"
        append_cpu_flag_if_supported "avx" "-mavx"
        append_cpu_flag_if_supported "avx2" "-mavx2"
        append_cpu_flag_if_supported "fma" "-mfma"
        append_cpu_flag_if_supported "f16c" "-mf16c"
        append_cpu_flag_if_supported "bmi1" "-mbmi"
        append_cpu_flag_if_supported "bmi2" "-mbmi2"
        if has_cpu_flag "abm" || has_cpu_flag "lzcnt"; then
            append_copt_if_supported "-mlzcnt"
        fi
        append_cpu_flag_if_supported "aes" "-maes"
        append_cpu_flag_if_supported "pclmulqdq" "-mpclmul"
        append_cpu_flag_if_supported "sha_ni" "-msha"
        append_cpu_flag_if_supported "avx512f" "-mavx512f"
        append_cpu_flag_if_supported "avx512cd" "-mavx512cd"
        append_cpu_flag_if_supported "avx512vl" "-mavx512vl"
        append_cpu_flag_if_supported "avx512bw" "-mavx512bw"
        append_cpu_flag_if_supported "avx512dq" "-mavx512dq"
        append_cpu_flag_if_supported "avx512vbmi" "-mavx512vbmi"
        append_cpu_flag_if_supported "avx512ifma" "-mavx512ifma"
        append_cpu_flag_if_supported "avx512_vnni" "-mavx512vnni"
        append_cpu_flag_if_supported "avx512_bf16" "-mavx512bf16"
        append_cpu_flag_if_supported "avx512_fp16" "-mavx512fp16"
    elif [[ "${ARCH}" == "arm64" || "${ARCH}" == "aarch64" ]]; then
        append_copt_if_supported "-mcpu=native"
        if [ "${#ARCH_COPTS[@]}" -eq 0 ]; then
            append_copt_if_supported "-march=native"
        fi
        if [ "${#ARCH_COPTS[@]}" -eq 0 ]; then
            append_copt_if_supported "-march=armv8-a+simd"
        fi
    else
        append_copt_if_supported "-march=native"
    fi
else
    if [[ "${ARCH}" == "x86_64" || "${ARCH}" == "amd64" ]]; then
        append_copt_if_supported "-msse4.2"
        append_copt_if_supported "-mpopcnt"
    elif [[ "${ARCH}" == "arm64" || "${ARCH}" == "aarch64" ]]; then
        append_copt_if_supported "-march=armv8-a+simd"
    fi
fi

if [[ "${HWY_TARGET_MODE}" == "all" ]]; then
    ARCH_COPTS+=("--copt=-DHWY_COMPILE_ALL_ATTAINABLE")
    ARCH_CXXOPTS+=("--cxxopt=-DHWY_COMPILE_ALL_ATTAINABLE")
elif [[ "${HWY_TARGET_MODE}" != "static" ]]; then
    echo "Unknown VQINDEX_HWY_TARGETS=${HWY_TARGET_MODE}; use all or static."
    exit 1
fi

if [ -n "${VQINDEX_EXTRA_COPTS:-}" ]; then
    read -r -a EXTRA_COPTS <<< "${VQINDEX_EXTRA_COPTS}"
    for flag in "${EXTRA_COPTS[@]}"; do
        ARCH_COPTS+=("--copt=${flag}")
        ARCH_CXXOPTS+=("--cxxopt=${flag}")
    done
fi

echo "CPU optimization mode: ${CPU_OPT_MODE}; Highway targets: ${HWY_TARGET_MODE}"
echo "Compiler flags: ${ARCH_COPTS[*]}"

COMMON_ARGS=(
    build
    -c opt
    --features=thin_lto
    "${ARCH_COPTS[@]}"
    "${ARCH_CXXOPTS[@]}"
    --cxxopt=-std=c++17
    --copt=-O3
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
usage
exit 1
