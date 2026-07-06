#!/usr/bin/env bash
# Build the native rust tvm-serve image (tvm-runtime-rust): loads
# $TVM_MODEL_DIR/model.so + metadata.json and serves OpenInference v2
# (REST 8080 / gRPC 9000). Model-centric — nothing model-specific is baked.
#
# Usage: ./build-image.sh [--load] [--push]
#   TAG=<repo:tag>         image name         (default tvm-runtime-rust:0.24)
#   REGISTRY=<host[/org]>  push target prefix (the pushed ref is $REGISTRY/$TAG)
#   --load  -> minikube image load (local dev)     --push -> push to the registry
# Remote cluster: build, --push to your registry, then point CORE at the exact
# pushed ref via RUNTIME_TVM_SERVE (e.g. RUNTIME_TVM_SERVE=ghcr.io/acme/tvm-runtime-rust:0.24).
# Requires a locally-built TVM (override with TVM_HOME / TVM_BUILD).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TVM_HOME="${TVM_HOME:-$HOME/tvm/src/tvm-current}"
TVM_BUILD="${TVM_BUILD:-$TVM_HOME/build}"
# Default tag = major.minor of the TVM actually packaged (resolved from the
# TVM_HOME dir name, e.g. tvm-0.24.0 -> 0.24), so the image tag always matches
# its contents. Override with TVM_TAG or a full TAG.
if [ -z "${TVM_TAG:-}" ]; then
    _v="$(basename "$(readlink -f "$TVM_HOME")")"; _v="${_v#tvm-}"
    if [[ "$_v" =~ ^[0-9]+\.[0-9]+ ]]; then TVM_TAG="${_v%.*}"; else TVM_TAG="0.24"; fi
fi
TAG="${TAG:-tvm-runtime-rust:${TVM_TAG}}"
REGISTRY="${REGISTRY:-}" # empty = push bare $TAG; set host[/org] to push for a remote cluster

LOAD=0; PUSH=0
for a in "$@"; do case "$a" in --load) LOAD=1 ;; --push) PUSH=1 ;; esac; done

[ -f "$TVM_BUILD/lib/libtvm_runtime.so" ] || { echo "no TVM build at $TVM_BUILD/lib"; exit 1; }

echo "== cargo build --release tvm-serve (TVM_BUILD_DIR=$TVM_BUILD) =="
PATH="$HERE/scripts:$PATH" TVM_BUILD_DIR="$TVM_BUILD" cargo build --release --bin tvm-serve

rm -rf "$HERE/lib"; mkdir -p "$HERE/lib"
cp "$TVM_BUILD/lib/libtvm_runtime.so" "$TVM_BUILD/lib/libtvm_ffi.so" "$HERE/lib/"
cp "$HERE/target/release/tvm-serve" "$HERE/tvm-serve"

echo "== docker build $TAG =="
docker build -t "$TAG" "$HERE"
rm -rf "$HERE/lib" "$HERE/tvm-serve"

[ "$LOAD" = 1 ] && { echo ">> minikube image load $TAG"; minikube image load "$TAG"; }
if [ "$PUSH" = 1 ]; then
    ref="${REGISTRY:+$REGISTRY/}$TAG"
    docker tag "$TAG" "$ref"; docker push "$ref"
    echo ">> pushed $ref  (point CORE at it: RUNTIME_TVM_SERVE=$ref)"
fi
echo "DONE $TAG"
