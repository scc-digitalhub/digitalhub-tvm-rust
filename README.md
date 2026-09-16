# digitalhub-tvm-rust

The **`ghcr.io/scc-digitalhub/tvm-runtime-rust`** image: **`tvm-serve`**, a Rust server
that serves a model compiled by TVM over the **Open Inference Protocol v2**, REST on
`8080` and gRPC on `9000`.

It is one of the two serve images of DigitalHub CORE's **`tvm+serve`** task. The default is
the Go image of `digitalhub-serverless`; both behave the same and can be swapped.

```
 TVM_MODEL_DIR                     tvm-serve
 ├── model.so        ──load──►   worker 1 ─ model copy ─┐
 └── metadata.json               worker 2 ─ model copy ─┤◄── REST :8080
                                 ...                    │◄── gRPC :9000
                                 worker N ─ model copy ─┘
```

Nothing model-specific is baked into the image. At startup `tvm-serve` reads `model.so`
and `metadata.json` from `TVM_MODEL_DIR`, checks that it can run them, loads one copy per
worker and starts both servers. Inference runs in-process on the TVM Relax VM, with no
Python.

## Configuration

| Variable              | Default         | Description                                                                      |
| --------------------- | --------------- | -------------------------------------------------------------------------------- |
| `TVM_MODEL_DIR`       | `/shared/model` | Folder with `model.so` and `metadata.json`.                                      |
| `TVM_MODEL_NAME`      | `model`         | Model name in the URLs, `/v2/models/<name>`.                                     |
| `TVM_SERVE_WORKERS`   | `1`             | Inferences run in parallel; each worker loads its own model copy.                |
| `TVM_NUM_THREADS`     | every core      | TVM threads of each worker. Keep `workers × threads` within the CPUs of the pod. |
| `TVM_SERVE_PORT`      | `8080`          | REST port.                                                                       |
| `TVM_SERVE_GRPC_PORT` | `9000`          | gRPC port.                                                                       |

## Endpoints

| What            | REST                                                        | gRPC             |
| --------------- | ----------------------------------------------------------- | ---------------- |
| Server live     | `GET /v2/health/live`                                       | `ServerLive`     |
| Server ready    | `GET /v2/health/ready`                                      | `ServerReady`    |
| Server metadata | `GET /v2`                                                   | `ServerMetadata` |
| Model ready     | `GET /v2/models/<name>/ready`                               | `ModelReady`     |
| Model metadata  | `GET /v2/models/<name>`                                     | `ModelMetadata`  |
| Inference       | `POST /v2/models/<name>/infer` (also `/versions/<v>/infer`) | `ModelInfer`     |

```bash
curl -X POST http://localhost:8080/v2/models/model/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"images","datatype":"FP32","shape":[1,3,640,640],"data":[...]}]}'
```

- **Inputs** are matched by name when every input has one and the names match the model,
  otherwise by position.
- **Data types**: `FP32`, `FP64`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`,
  `UINT32`, `UINT64`. `FP16` is not supported yet.
- **Size limits**: 1 GiB for a REST request, 512 MB for a gRPC message.
- **Quantized models** (`int8` / `uint8` tensors): the REST model metadata adds `scale`,
  `zero_point` and, per axis, `quantized_dimension` under `parameters`, so the client can
  convert values (`real = (q - zero_point) * scale`). The gRPC metadata has no field for them.
- **gRPC clients** need the proto, `crates/tvm-serve/proto/grpc_predict_v2.proto` (no server
  reflection).
- The REST response reports `inference_time_ms` in `parameters`.

## Which models it serves

At startup the server refuses the model, with a clear error, unless:

- `metadata.json` has the same `tvm_version` and `tvm_git_commit` as the TVM built into the
  image: **compile with the `tvm-toolkit` of the same release**;
- the model was compiled for a CPU (LLVM target) of the image architecture;
- every input and output uses a supported data type.

The image exists for `linux/amd64`, `linux/arm64` and `linux/arm/v7`.

## Run it

**From CORE**: set `RUNTIME_TVM_SERVE=ghcr.io/scc-digitalhub/tvm-runtime-rust:<version>` to
use it for every serve, or `image` on a single `tvm+serve` task. CORE downloads the model
into `TVM_MODEL_DIR` with an init container and sets `TVM_MODEL_NAME`, `TVM_SERVE_WORKERS`
and `TVM_NUM_THREADS` (the task CPUs divided by the workers).

**With Docker**, given a folder with `model.so` and `metadata.json`:

```bash
docker run --rm -p 8080:8080 -p 9000:9000 \
  -v "$PWD/my-model:/shared/model" \
  ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0
```

## Versions and release

**The image tag is the git tag, and it is the Apache TVM version**: tag `0.26.0` builds
`tvm-runtime-rust:0.26.0` on Apache TVM `0.26.0`.

Push a tag `X.Y.Z` (or `X.Y`) and GitHub Actions (`.github/workflows/tvm-runtime-rust-image.yml`)
builds the image. For each architecture it:

1. compiles only the TVM runtime (`libtvm_runtime.so`, `libtvm_ffi.so`) of Apache TVM
   `vX.Y.Z`: natively on amd64 and arm64, cross-compiled for armv7;
2. builds `tvm-serve` with `cargo build --locked --release`, embedding the TVM version and
   commit (armv7 also applies `patches/tvm-ffi-rust-32bit.patch`);
3. builds the image and checks the libraries inside it (`ldd` and SHA-256);
4. pushes `<tag>-<arch>`.

A last job joins the images into the multi-architecture tag.

## Development

| Path                     | Content                                                                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/tvm-relax`       | Library that loads `model.so`, runs the Relax VM and checks `metadata.json`.                                                                    |
| `crates/tvm-serve`       | The server: `main.rs` (configuration and REST), `worker.rs` (worker pool), `protocol.rs` (v2 types), `grpc.rs`, `build.rs` (proto and linking). |
| `scripts/tvm-ffi-config` | Tells the `tvm-ffi` bindings where `libtvm_ffi.so` is.                                                                                          |
| `patches/`               | Fix applied to the `tvm-ffi` Rust bindings for 32-bit ARM.                                                                                      |

Building needs a local build of the same Apache TVM release:

```bash
TVM=~/tvm/src/tvm-0.26.0
ln -sfn "$TVM/3rdparty/tvm-ffi/rust" .tvm-ffi-rust   # Rust bindings of that release

export TVM_BUILD_DIR="$TVM/build" TVM_FFI_LIBDIR="$TVM/build/lib"
export PATH="$PWD/scripts:$PATH" LD_LIBRARY_PATH="$TVM/build/lib"
export TVM_VERSION=0.26.0 TVM_GIT_COMMIT="$(git -C "$TVM" rev-parse HEAD)"

cargo test -p tvm-relax -p tvm-serve
cargo build --release --bin tvm-serve
```

Without `TVM_VERSION` and `TVM_GIT_COMMIT` the binary refuses every model.

## Limitations

- CPU only, no GPU.
- One model per pod; no batching and no metrics.
- `FP16` tensors are not supported yet.
