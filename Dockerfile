# Model-agnostic TVM serving image (tvm-runtime-rust): runtime .so + tvm-serve
# binary only. A model (model.so + metadata.json) is injected into TVM_MODEL_DIR
# at deploy time; serves OpenInference v2 (REST 8080 / gRPC 9000).
#
# Artifacts are x86_64/glibc; arm64/armv7l need a rebuild on that arch.
# Base needs GLIBC_2.38 + GLIBCXX_3.4.32 -> ubuntu:24.04 (glibc 2.39, libstdc++13).
FROM ubuntu:24.04

COPY lib/libtvm_runtime.so lib/libtvm_ffi.so /opt/tvm/lib/
COPY tvm-serve /usr/local/bin/tvm-serve

# LD_LIBRARY_PATH beats the binary's DT_RUNPATH (which points at the build-host path).
ENV LD_LIBRARY_PATH=/opt/tvm/lib \
    TVM_MODEL_DIR=/shared/model \
    TVM_SERVE_PORT=8080 \
    TVM_SERVE_GRPC_PORT=9000

EXPOSE 8080 9000
ENTRYPOINT ["/usr/local/bin/tvm-serve"]
