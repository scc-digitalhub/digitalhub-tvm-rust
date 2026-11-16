#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2025 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

# Build the native rust tvm-serve image (tvm-runtime-rust): loads
# $TVM_MODEL_DIR/model.so + metadata.json and serves OpenInference v2
# (REST 8080 / gRPC 9000). Model-centric — nothing model-specific is baked.
#
# Usage: ./build-image.sh [--load] [--push]
#   TAG=<repo:tag>         image name         (default tvm-runtime-rust:0.26)
#   REGISTRY=<host[/org]>  push target prefix (the pushed ref is $REGISTRY/$TAG)
#   TVM_BACKEND=cpu|cuda  CUDA additionally needs libtvm_runtime_cuda.so (default cpu)
#   TVM_EXPORTED_ARTIFACTS=1 verifies CUDA toolkit exports instead of a local CMake build
#   --load  -> minikube image load (local dev)     --push -> push to the registry
# Remote cluster: build, --push to your registry, then point CORE at the exact
# pushed ref via RUNTIME_TVM_SERVE (e.g. RUNTIME_TVM_SERVE=ghcr.io/acme/tvm-runtime-rust:0.26).
# Requires a locally-built TVM (override with TVM_HOME / TVM_BUILD).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TVM_HOME="${TVM_HOME:-$HOME/tvm/src/tvm-current}"
TVM_BUILD="${TVM_BUILD:-$TVM_HOME/build}"
TVM_BACKEND="${TVM_BACKEND:-cpu}"
CUDA_BASE_IMAGE="nvidia/cuda:13.2.1-base-ubuntu24.04@sha256:7be56e69d8ae7c3648b8fca009fa35980ecd9c6eefaafbff3ee8c224e0043eb5"
libraries=(libtvm_ffi.so libtvm_runtime.so)
cargo_features=()
verify_args=()
if [ "${TVM_EXPORTED_ARTIFACTS:-0}" = 1 ]; then
    verify_args+=(--exported-artifacts)
fi
require_cuda_tag() {
    local cuda_image_tag="${1##*/}"
    if [[ "$cuda_image_tag" != *:* || ! "${cuda_image_tag##*:}" =~ (^|[-_.])cuda13[.]2[.]1($|[-_]) ]]; then
        echo "CUDA image tags must contain a cuda13.2.1 component to preserve CPU tags" >&2
        return 2
    fi
}
case "$TVM_BACKEND" in
    cpu) BASE_IMAGE="${BASE_IMAGE:-ubuntu:24.04}"; backend_tag="" ;;
    cuda)
        BASE_IMAGE="${BASE_IMAGE:-$CUDA_BASE_IMAGE}"
        backend_tag="-cuda13.2.1"
        if [ -n "${TAG:-}" ]; then require_cuda_tag "$TAG"; fi
        libraries+=(libtvm_runtime_cuda.so)
        cargo_features=(--features cuda)
        case "$(uname -m)" in x86_64|aarch64) ;; *) echo "CUDA image supports only AMD64 and ARM64" >&2; exit 1 ;; esac
        ;;
    *) echo "TVM_BACKEND must be cpu or cuda" >&2; exit 2 ;;
esac
for tool in cargo docker git python3 readelf sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
done
IFS=$'\t' read -r TVM_RELEASE SOURCE_TAG TVM_GIT_COMMIT TVM_FFI_GIT_COMMIT TVM_FFI_VERSION < <(
    python3 "$HERE/scripts/verify-tvm-build.py" --source "$TVM_HOME" --build "$TVM_BUILD" --backend "$TVM_BACKEND" "${verify_args[@]}"
)
TVM_TAG="${TVM_TAG:-$SOURCE_TAG}"
TAG="${TAG:-tvm-runtime-rust:${TVM_TAG}${backend_tag}}"
if [ "$TVM_BACKEND" = cuda ]; then require_cuda_tag "$TAG"; fi
REGISTRY="${REGISTRY:-}" # empty = push bare $TAG; set host[/org] to push for a remote cluster

LOAD=0; PUSH=0
for a in "$@"; do case "$a" in --load) LOAD=1 ;; --push) PUSH=1 ;; esac; done

FFI_RUST_SOURCE="$(readlink -f "$TVM_HOME/3rdparty/tvm-ffi/rust")"
FFI_RUST_LINK="$HERE/.tvm-ffi-rust"
if [ -e "$FFI_RUST_LINK" ] && [ ! -L "$FFI_RUST_LINK" ]; then
    echo "$FFI_RUST_LINK exists and is not a symlink" >&2
    exit 1
fi
ln -sfn "$FFI_RUST_SOURCE" "$FFI_RUST_LINK"
[ "$(readlink -f "$FFI_RUST_LINK")" = "$FFI_RUST_SOURCE" ] || {
    echo "failed to align Cargo tvm-ffi source with $FFI_RUST_SOURCE" >&2
    exit 1
}

LINK_BUILD="$TVM_BUILD"
if [ "$TVM_BACKEND" = cuda ] && [ "${TVM_EXPORTED_ARTIFACTS:-0}" = 1 ]; then
    # Keep toolkit-export link dependencies in an ignored build directory. The
    # driver stub supports builds on GPU-less hosts and is never staged below.
    LINK_BUILD="$HERE/.cuda-build/exported-link-$(uname -m)"
    mkdir -p "$LINK_BUILD/lib"
    for library in "${libraries[@]}"; do
        ln -sfn "$(realpath "$TVM_BUILD/lib/$library")" "$LINK_BUILD/lib/$library"
    done
    cudart="${CUDA_CUDART_LIBRARY:-$TVM_BUILD/cuda/libcudart.so.13}"
    driver_stub="${CUDA_DRIVER_STUB_LIBRARY:-$TVM_BUILD/cuda-driver-stubs/libcuda.so.1}"
    [ -f "$cudart" ] && [ -f "$driver_stub" ] || {
        echo "CUDA export needs cuda/libcudart.so.13 and cuda-driver-stubs/libcuda.so.1 (or CUDA_CUDART_LIBRARY / CUDA_DRIVER_STUB_LIBRARY)" >&2
        exit 1
    }
    ln -sfn "$(realpath "$cudart")" "$LINK_BUILD/lib/libcudart.so.13"
    ln -sfn "$(realpath "$driver_stub")" "$LINK_BUILD/lib/libcuda.so.1"
    cc -shared -x c /dev/null -o "$LINK_BUILD/lib/libtvm_ffi_testing.so"
fi

echo "== cargo build --release tvm-serve (TVM_BUILD_DIR=$TVM_BUILD) =="
binary="$(PATH="$HERE/scripts:$PATH" \
    TVM_BUILD_DIR="$LINK_BUILD" \
    TVM_FFI_LIBDIR="$LINK_BUILD/lib" \
    LD_LIBRARY_PATH="$LINK_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    TVM_VERSION="$TVM_RELEASE" \
    TVM_GIT_COMMIT="$TVM_GIT_COMMIT" \
    cargo build --locked --release --manifest-path "$HERE/Cargo.toml" --bin tvm-serve \
        --message-format=json-render-diagnostics "${cargo_features[@]}" | python3 -c '
import json, sys
binary = None
for line in sys.stdin:
    message = json.loads(line)
    if message.get("reason") == "compiler-artifact" and message.get("target", {}).get("name") == "tvm-serve":
        binary = message.get("executable") or binary
if binary is None:
    raise SystemExit("Cargo did not report the tvm-serve executable")
print(binary)
')"

for library in "${libraries[@]}"; do
    readelf -d "$binary" | grep -F "Shared library: [$library]" >/dev/null || {
        echo "tvm-serve is not linked to $library" >&2
        exit 1
    }
done
if [ "$TVM_BACKEND" = cpu ] && readelf -d "$binary" | grep -E 'NEEDED.*(libtvm_runtime_cuda|libcuda|libnvidia)' >/dev/null; then
    echo "CPU binary unexpectedly depends on NVIDIA libraries" >&2
    exit 1
fi

# Retain an isolated context so existing artifacts and build evidence survive.
if [ -n "${TVM_BUILD_CONTEXT:-}" ]; then
    ctx="$TVM_BUILD_CONTEXT"
    mkdir "$ctx"
else
    ctx="$(mktemp -d -t tvm-rust-image.XXXXXXXX)"
fi
echo "== build context: $ctx =="
cp "$HERE/Dockerfile" "$ctx/Dockerfile"
mkdir -p "$ctx/lib" "$ctx/evidence"
for library in "${libraries[@]}"; do
    cp "$TVM_BUILD/lib/$library" "$ctx/lib/"
done
cp "$binary" "$ctx/tvm-serve"
provenance="$TVM_BUILD/digitalhub-build-provenance.json"
if [ "${TVM_EXPORTED_ARTIFACTS:-0}" = 1 ] && [ ! -f "$provenance" ]; then
    provenance="$TVM_BUILD/provenance.json"
fi
cp "$provenance" "$ctx/provenance.json"
if [ "$TVM_BACKEND" = cuda ]; then
    cuda_cache="$TVM_BUILD/cuda-CMakeCache.txt"
    [ -f "$cuda_cache" ] || cuda_cache="$TVM_BUILD/CMakeCache.txt"
    cp "$cuda_cache" "$ctx/evidence/cuda-CMakeCache.txt"
fi

python3 - "$HERE" "$ctx" <<'PY_SOURCE'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

source, context = map(Path, sys.argv[1:])
paths = subprocess.check_output(
    ["git", "-C", str(source), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
).decode().split("\0")
files = {
    name: hashlib.sha256((source / name).read_bytes()).hexdigest()
    for name in sorted(set(paths))
    if name and (name.startswith("crates/") or name in {"Cargo.toml", "Cargo.lock", "Dockerfile"})
    and (source / name).is_file()
}
manifest = json.loads((context / "provenance.json").read_text())
manifest["runtime_source"] = {
    "revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
    "files_sha256": files,
    "binary_sha256": hashlib.sha256((context / "tvm-serve").read_bytes()).hexdigest(),
}
(context / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY_SOURCE

echo "== docker build $TAG =="
docker build \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "TVM_BACKEND=$TVM_BACKEND" \
    --build-arg "TVM_VERSION=$TVM_RELEASE" \
    --build-arg "TVM_GIT_COMMIT=$TVM_GIT_COMMIT" \
    --build-arg "TVM_FFI_VERSION=$TVM_FFI_VERSION" \
    --build-arg "TVM_FFI_GIT_COMMIT=$TVM_FFI_GIT_COMMIT" \
    -t "$TAG" "$ctx"

for library in "${libraries[@]}"; do
    source_line="$(sha256sum "$TVM_BUILD/lib/$library")"
    image_line="$(docker run --rm --entrypoint /usr/bin/sha256sum "$TAG" "/opt/tvm/lib/$library")"
    source_hash="${source_line%% *}"
    image_hash="${image_line%% *}"
    [ "$image_hash" = "$source_hash" ] || {
        echo "$library in $TAG does not match the verified TVM build" >&2
        exit 1
    }
done

[ "$LOAD" = 1 ] && { echo ">> minikube image load $TAG"; minikube image load "$TAG"; }
if [ "$PUSH" = 1 ]; then
    ref="${REGISTRY:+$REGISTRY/}$TAG"
    image_tag="${ref##*:}"
    if [ "$TVM_BACKEND" = cpu ] && [[ "${image_tag,,}" == *cuda* ]]; then
        echo "CPU publication cannot use a CUDA tag" >&2
        exit 1
    fi
    python3 "$HERE/scripts/check-image-tag.py" "$ref"
    docker tag "$TAG" "$ref"; docker push "$ref"
    echo ">> pushed $ref  (point CORE at it: RUNTIME_TVM_SERVE=$ref)"
fi
echo "DONE $TAG"
