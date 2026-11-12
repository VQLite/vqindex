#!/bin/bash
# Downloads the Bazel release VQIndex is built with into tools/bazel/bazel and
# verifies its checksum. build.sh picks that binary up automatically.
#
#   tools/install_bazel.sh
#
# Only Linux x86-64 is pinned here; on other platforms install Bazel 7.x
# yourself (https://bazel.build/install) and pass BAZEL=/path/to/bazel to
# build.sh.
set -euo pipefail

BAZEL_VERSION=7.6.1
BAZEL_SHA256=ac6249d1192aea9feaf49dfee2ab50c38cee2454b00cf29bbec985a11795c025

if [ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]; then
    echo >&2 "Only Linux x86-64 is pinned; install Bazel ${BAZEL_VERSION} manually."
    exit 1
fi

dir="$(cd "$(dirname "$0")" && pwd)/bazel"
dst="${dir}/bazel"
url="https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64"

mkdir -p "${dir}"
if [ -x "${dst}" ] && echo "${BAZEL_SHA256}  ${dst}" | sha256sum -c --status; then
    echo "${dst} is already Bazel ${BAZEL_VERSION}."
    exit 0
fi

tmp="${dst}.download"
if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -C - -o "${tmp}" "${url}"
else
    wget -c -O "${tmp}" "${url}"
fi
if ! echo "${BAZEL_SHA256}  ${tmp}" | sha256sum -c --status; then
    echo >&2 "Checksum mismatch for ${tmp}; delete it and retry."
    exit 1
fi
chmod +x "${tmp}"
mv -f "${tmp}" "${dst}"
echo "Installed $("${dst}" --version) at ${dst}"
