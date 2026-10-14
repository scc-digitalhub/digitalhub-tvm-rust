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
#   --load  -> minikube image load (local dev)     --push -> push to the registry
# Remote cluster: build, --push to your registry, then point CORE at the exact
# pushed ref via RUNTIME_TVM_SERVE (e.g. RUNTIME_TVM_SERVE=ghcr.io/acme/tvm-runtime-rust:0.26).
# Requires a locally-built TVM (override with TVM_HOME / TVM_BUILD).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TVM_HOME="${TVM_HOME:-$HOME/tvm/src/tvm-current}"
TVM_BUILD="${TVM_BUILD:-$TVM_HOME/build}"
for tool in cargo docker git python3 readelf sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
done
IFS=$'\t' read -r TVM_RELEASE SOURCE_TAG TVM_GIT_COMMIT TVM_FFI_GIT_COMMIT TVM_FFI_VERSION < <(
    python3 "$HERE/scripts/verify-tvm-build.py" --source "$TVM_HOME" --build "$TVM_BUILD"
)
TVM_TAG="${TVM_TAG:-$SOURCE_TAG}"
TAG="${TAG:-tvm-runtime-rust:${TVM_TAG}}"
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

echo "== cargo build --release tvm-serve (TVM_BUILD_DIR=$TVM_BUILD) =="
PATH="$HERE/scripts:$PATH" \
    TVM_BUILD_DIR="$TVM_BUILD" \
    TVM_FFI_LIBDIR="$TVM_BUILD/lib" \
    TVM_VERSION="$TVM_RELEASE" \
    TVM_GIT_COMMIT="$TVM_GIT_COMMIT" \
    cargo build --locked --release --bin tvm-serve

for library in libtvm_ffi.so libtvm_runtime.so; do
    readelf -d "$HERE/target/release/tvm-serve" | grep -F "Shared library: [$library]" >/dev/null || {
        echo "tvm-serve is not linked to $library" >&2
        exit 1
    }
done

cleanup() { rm -rf "$HERE/lib" "$HERE/tvm-serve" "$HERE/provenance.json"; }
trap cleanup EXIT
cleanup
mkdir -p "$HERE/lib"
cp "$TVM_BUILD/lib/libtvm_runtime.so" "$TVM_BUILD/lib/libtvm_ffi.so" "$HERE/lib/"
cp "$HERE/target/release/tvm-serve" "$HERE/tvm-serve"
cp "$TVM_BUILD/digitalhub-build-provenance.json" "$HERE/provenance.json"

echo "== docker build $TAG =="
docker build \
    --build-arg "TVM_VERSION=$TVM_RELEASE" \
    --build-arg "TVM_GIT_COMMIT=$TVM_GIT_COMMIT" \
    --build-arg "TVM_FFI_VERSION=$TVM_FFI_VERSION" \
    --build-arg "TVM_FFI_GIT_COMMIT=$TVM_FFI_GIT_COMMIT" \
    -t "$TAG" "$HERE"

for library in libtvm_runtime.so libtvm_ffi.so; do
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
    docker tag "$TAG" "$ref"; docker push "$ref"
    echo ">> pushed $ref  (point CORE at it: RUNTIME_TVM_SERVE=$ref)"
fi
echo "DONE $TAG"
