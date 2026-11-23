# SPDX-FileCopyrightText: © 2025 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

# Model-agnostic TVM serving image (tvm-runtime-rust): runtime .so + tvm-serve
# binary only. A model (model.so + metadata.json) is injected into TVM_MODEL_DIR
# at deploy time; serves OpenInference v2 (REST 8080 / gRPC 9000).
#
# Artifacts are x86_64/glibc; arm64/armv7l need a rebuild on that arch.
# Base needs GLIBC_2.38 + GLIBCXX_3.4.32 -> ubuntu:24.04 (glibc 2.39, libstdc++13).
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}
ARG TVM_BACKEND=cpu
ARG TVM_VERSION=unknown
ARG TVM_GIT_COMMIT=unknown
ARG TVM_FFI_VERSION=unknown
ARG TVM_FFI_GIT_COMMIT=unknown
LABEL org.opencontainers.image.version="${TVM_VERSION}" \
    org.opencontainers.image.source="https://github.com/apache/tvm" \
    org.opencontainers.image.revision="${TVM_GIT_COMMIT}" \
    org.digitalhub.tvm-ffi.version="${TVM_FFI_VERSION}" \
    org.digitalhub.tvm-ffi.revision="${TVM_FFI_GIT_COMMIT}" \
    org.digitalhub.tvm.backend="${TVM_BACKEND}"

COPY lib/ /opt/tvm/lib/
COPY provenance.json /opt/tvm/provenance.json
COPY evidence/ /opt/tvm/
COPY tvm-serve /usr/local/bin/tvm-serve

# Preserve the CUDA base's NVIDIA driver search paths. Ubuntu's empty default
# still yields exactly /opt/tvm/lib for CPU images.
ENV LD_LIBRARY_PATH=/opt/tvm/lib${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH} \
    TVM_VERSION=${TVM_VERSION} \
    TVM_GIT_COMMIT=${TVM_GIT_COMMIT} \
    TVM_FFI_VERSION=${TVM_FFI_VERSION} \
    TVM_FFI_GIT_COMMIT=${TVM_FFI_GIT_COMMIT} \
    TVM_MODEL_DIR=/shared/model \
    TVM_SERVE_PORT=8080 \
    TVM_SERVE_GRPC_PORT=9000

RUN set -eu; \
    dependencies="$(ldd /usr/local/bin/tvm-serve)"; \
    printf '%s\n' "$dependencies" | grep -F '/opt/tvm/lib/libtvm_ffi.so'; \
    printf '%s\n' "$dependencies" | grep -F '/opt/tvm/lib/libtvm_runtime.so'; \
    case "$TVM_BACKEND" in \
        cpu) \
            ! printf '%s\n' "$dependencies" | grep -E 'not found|libcuda|libnvidia|libnvrtc'; \
            test ! -e /opt/tvm/lib/libtvm_runtime_cuda.so ;; \
        cuda) \
            printf '%s\n' "$dependencies" | grep -F '/opt/tvm/lib/libtvm_runtime_cuda.so'; \
            printf '%s\n' "$dependencies" | grep -F 'libcudart.so.13'; \
            # The host NVIDIA container runtime provides libcuda.so.1 on deployment.
            ! printf '%s\n' "$dependencies" | grep 'not found' | grep -v 'libcuda.so.1 => not found'; \
            test ! -e /opt/tvm/lib/libcuda.so.1 ;; \
        *) echo "unsupported TVM_BACKEND=$TVM_BACKEND" >&2; exit 1 ;; \
    esac

EXPOSE 8080 9000
ENTRYPOINT ["/usr/local/bin/tvm-serve"]
