# SPDX-FileCopyrightText: © 2025 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

# Model-agnostic TVM serving image (tvm-runtime-rust): runtime .so + tvm-serve
# binary only. A model (model.so + metadata.json) is injected into TVM_MODEL_DIR
# at deploy time; serves OpenInference v2 (REST 8080 / gRPC 9000).
#
# Artifacts are x86_64/glibc; arm64/armv7l need a rebuild on that arch.
# Base needs GLIBC_2.38 + GLIBCXX_3.4.32 -> ubuntu:24.04 (glibc 2.39, libstdc++13).
FROM ubuntu:24.04
ARG TVM_VERSION=unknown
ARG TVM_GIT_COMMIT=unknown
ARG TVM_FFI_VERSION=unknown
ARG TVM_FFI_GIT_COMMIT=unknown
LABEL org.opencontainers.image.version="${TVM_VERSION}" \
    org.opencontainers.image.source="https://github.com/apache/tvm" \
    org.opencontainers.image.revision="${TVM_GIT_COMMIT}" \
    org.digitalhub.tvm-ffi.version="${TVM_FFI_VERSION}" \
    org.digitalhub.tvm-ffi.revision="${TVM_FFI_GIT_COMMIT}"

COPY lib/libtvm_runtime.so lib/libtvm_ffi.so /opt/tvm/lib/
COPY provenance.json /opt/tvm/provenance.json
COPY tvm-serve /usr/local/bin/tvm-serve

# LD_LIBRARY_PATH beats the binary's DT_RUNPATH (which points at the build-host path).
ENV LD_LIBRARY_PATH=/opt/tvm/lib \
    TVM_VERSION=${TVM_VERSION} \
    TVM_GIT_COMMIT=${TVM_GIT_COMMIT} \
    TVM_FFI_VERSION=${TVM_FFI_VERSION} \
    TVM_FFI_GIT_COMMIT=${TVM_FFI_GIT_COMMIT} \
    TVM_MODEL_DIR=/shared/model \
    TVM_SERVE_PORT=8080 \
    TVM_SERVE_GRPC_PORT=9000

RUN ldd /usr/local/bin/tvm-serve | grep -F '/opt/tvm/lib/libtvm_ffi.so' \
    && ldd /usr/local/bin/tvm-serve | grep -F '/opt/tvm/lib/libtvm_runtime.so' \
    && ! ldd /usr/local/bin/tvm-serve | grep -F 'not found'

EXPOSE 8080 9000
ENTRYPOINT ["/usr/local/bin/tvm-serve"]
