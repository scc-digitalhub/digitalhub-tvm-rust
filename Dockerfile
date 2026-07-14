# Model-agnostic serving image for TVM in Rust (tvm-runtime-rust).
#
# Contains ONLY the runtime (libtvm_runtime.so + libtvm_ffi.so, NOT the 1.4GB
# libtvm_compiler.so) and the `tvm-serve` binary. Nothing model-specific is
# baked in: at deploy time an init container downloads the compiled Model
# (model.so + metadata.json) into TVM_MODEL_DIR and tvm-serve loads it from
# there, exposing OpenInference v2 (REST 8080 / gRPC 9000).
#
# The .so libs and the binary come from the build context staged by
# build-image.sh. NB: x86_64/glibc artifacts; other architectures need a
# rebuild on that arch.
#
# The base MUST have glibc/libstdc++ >= the build host: these TVM .so require
# GLIBC_2.38 + GLIBCXX_3.4.32 -> ubuntu:24.04 (glibc 2.39, libstdc++13).
FROM ubuntu:24.04

COPY lib/libtvm_runtime.so lib/libtvm_ffi.so /opt/tvm/lib/
COPY tvm-serve /usr/local/bin/tvm-serve

# LD_LIBRARY_PATH takes precedence over DT_RUNPATH -> the binary finds the .so
# here (the RUNPATH points at the build-host path, which does not exist in the
# container). TVM_MODEL_DIR matches the binary default and the serve runner.
ENV LD_LIBRARY_PATH=/opt/tvm/lib \
    TVM_MODEL_DIR=/shared/model \
    TVM_SERVE_PORT=8080 \
    TVM_SERVE_GRPC_PORT=9000

EXPOSE 8080 9000
ENTRYPOINT ["/usr/local/bin/tvm-serve"]
