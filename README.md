# digitalhub-tvm-rust · native `tvm-serve` image

Native **Rust** serving of compiled **TVM Relax** models. This standalone project
builds the `tvm-serve` binary and packages it into the **`tvm-runtime-rust`**
container image, one of the two serve images DigitalHub CORE can launch for its
**`tvm+serve`** task (the other, and the default, is the Go runtime of
`digitalhub-serverless`).

The image is **model-centric**: nothing model-specific is baked in. At startup
`tvm-serve` reads `model.so` + `metadata.json` from `$TVM_MODEL_DIR` and exposes
the model over the **Open Inference Protocol v2** (KServe) on **REST `:8080`** and
**gRPC `:9000`**. Inference runs directly on the TVM VirtualMachine driven from
Rust.

Startup fails closed unless the model metadata TVM version and source revision
match the immutable identity compiled into the server from the verified build,
and the model declares an LLVM target compatible with the runtime architecture.

## Purpose

CORE's `tvm+serve` needs a base image that can take a freshly compiled Relax
`model.so` and serve it. `tvm-runtime-rust` is such an image:

- a single self-contained `tvm-serve` binary + the TVM runtime `.so`s;
- model-agnostic — the model is injected at deploy time (init container), not
  baked;
- serves OpenInference v2 over REST and gRPC from the same process;
- **CPU only**, native dtypes (FP16 deferred);
- published for `linux/amd64`, `linux/arm64` and `linux/arm/v7`.

## Architecture

```
                    ┌──────────────────────────── tvm-serve process ───────────────────────────┐
                    │                                                                          │
  HTTP :8080  ─────▶│  axum REST server ─┐                                                     │
  (OpenInf v2)      │  (protocol.rs)     │                                                     │
                    │                    ├──▶ Arc<Handle> ──▶ shared queue ──▶ worker pool     │
  gRPC :9000  ─────▶│  tonic gRPC server ┘   (Send jobs:      (mpsc)          (N threads)      │
  (GRPCInference)   │  (grpc.rs)             typed data+shape)                each own a       │
                    │                                                         model copy       │
                    │                                                                          │
                    │                                          RelaxModel (tvm-relax) × N      │
                    │                                          model.so + Relax VM             │
                    └────────────────────────────────────────────────────────────────┼─────────┘
                                                                                      │ tvm-ffi C ABI
                                                                             libtvm_runtime.so
                                                                             libtvm_ffi.so
```

Two async servers (REST + gRPC) share a **pool of worker threads**. The TVM VM and
all `tvm-ffi` handles are **not `Send`/`Sync`**, so each worker loads its **own copy**
of the model on a dedicated OS thread and never crosses threads. REST and gRPC
handlers submit inference jobs to a **shared mpsc queue** that all workers drain (an
actor pattern); the channel only carries `Send` data (typed tensor bytes + shape).
With **N** workers up to N inferences run concurrently, at the cost of N model
copies in memory. The pool size is set by `TVM_SERVE_WORKERS` (default `1`); with a
single worker inferences are serialized — one at a time.

Each worker thread also gets its **own TVM thread pool** for the operators of one
inference, sized by `TVM_NUM_THREADS`. Keep `workers × TVM_NUM_THREADS` within the
CPUs of the pod: CORE does this for you by splitting the requested CPUs among the
workers.

### Crates

| Crate       | Path               | Role                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tvm-relax` | `crates/tvm-relax` | The `RelaxModel` inference library. Loads `model.so` and drives the Relax VirtualMachine through raw `PackedFunc` calls over the `tvm-ffi` C ABI. `src/lib.rs` is the core.                                                                                                                                                                                                                                                                        |
| `tvm-serve` | `crates/tvm-serve` | The server binary. `main.rs` reads env config, spins up the worker pool (each thread loads its own model copy), starts REST + gRPC. `worker.rs` is the pool of model-owning threads + the shared inference queue. `protocol.rs` has the REST OpenInference v2 handlers and the v2 JSON structs. `grpc.rs` implements the gRPC `GRPCInferenceService`. `build.rs` compiles the proto and does the native link setup (force-links `libtvm_runtime`). |

### The crux: driving the Relax VM from Rust

There is no high-level Rust binding for the Relax VM — `tvm-ffi` only exposes
`Module`/`Function`/`Tensor` at the PackedFunc level. `tvm-relax` reproduces the
sequence the TVM C++ runtime performs internally, entirely by name over the C ABI
(`RelaxModel::load` in `src/lib.rs`):

```text
lib = Module::load_from_file("model.so")          // the compiled DSO
vm  = lib["vm_load_executable"]()                  // instantiate the Relax VM module  -> Module
      vm["vm_initialization"](kDLCPU,0,kPooled, kDLCPU,0,kPooled)   // bind CPU device + allocator
out = vm[entry](inputs…)                            // run the model's entry function
```

A single Relax output Tensor or an output tuple (`Array<Tensor>`) is normalized to
`Vec<Tensor>`.

**Why `build.rs` force-links `libtvm_runtime` (`--no-as-needed`):** the runtime
`.so` registers the `relax.VMExecutable` loader through a static initializer. The
binary never references its symbols directly, so the default `--as-needed` linker
behavior would drop it and the VM loader would not be registered at runtime.
`tvm-serve/build.rs` wraps `-ltvm_runtime` in `--no-as-needed` / `--as-needed` and
adds a relative `$ORIGIN` rpath. (`cargo:rustc-link-arg` does not propagate from a dependency to the
binary crate, so the setup lives in the binary crate's `build.rs`.)

## Building the image

The image is built by GitHub Actions (`.github/workflows/tvm-runtime-rust-image.yml`)
when a tag is pushed. **The image tag is the git tag**, and it names the Apache TVM
release: pushing `0.26.0` publishes `ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0`
built on TVM `0.26.0`. For each architecture the workflow:

1. compiles only the TVM runtime (`libtvm_runtime.so`, `libtvm_ffi.so`, no LLVM) from
   that Apache TVM release — natively on amd64 and arm64, cross-compiled for armv7;
2. builds `tvm-serve` against it with `cargo build --locked --release`. The 32-bit
   armv7 build applies `patches/tvm-ffi-rust-32bit.patch`, which fixes a duplicated
   padding field in the upstream `tvm-ffi` Rust bindings;
3. packages the binary and the two libraries, and checks them inside the image
   (`ldd`, SHA-256 against the build provenance).

A last job joins the three images into one multi-architecture tag.

The Rust build finds the TVM libraries two ways: `build.rs` requires `TVM_BUILD_DIR`,
and `tvm-ffi-sys` calls `tvm-ffi-config`, a shim in `scripts/tvm-ffi-config` (put it
on `PATH` and set `TVM_FFI_LIBDIR`). The ignored relative Cargo path `.tvm-ffi-rust`
must point at the `tvm-ffi` submodule of the same TVM checkout, so the Rust bindings
and the packaged library cannot drift.

The image is based on **`ubuntu:24.04`** (glibc ≥ 2.38 / `GLIBCXX_3.4.32`).

## How CORE uses it

CORE's `TvmServeRunner` deploys the serve image as a Kubernetes Deployment. It
is a build-free, model-injection pattern:

```
             (init container)                         (tvm-serve container)
  S3 store:// .so folder  ──download──▶  <home>/model  ──TVM_MODEL_DIR──▶  tvm-serve
  model.so + metadata.json                                                 REST :8080
                                                                           gRPC :9000
```

- An **init container** downloads the compiled `.so` Model folder (`model.so` +
  `metadata.json`) from S3 into `<home-dir>/model`.
- `tvm-serve` is pointed there via **`TVM_MODEL_DIR`**, with `TVM_MODEL_NAME` set
  to the served model name (used in `/v2/models/<name>`).
- `TVM_SERVE_WORKERS` comes from the task `workers`, and `TVM_NUM_THREADS` from the
  task CPU request divided by the workers.
- The Deployment declares service ports **8080** (REST) and **9000** (gRPC).

The serve image is configured by **`runtime.tvm.serve`** (env `RUNTIME_TVM_SERVE`),
which defaults to the Go image `tvm-runtime-go`. Point it at
`ghcr.io/scc-digitalhub/tvm-runtime-rust:<version>` to use this image everywhere, or
set `task.image` on a single `tvm+serve` run.

### Runtime env config (read by `tvm-serve`)

| Env var               | Default                  | Meaning                                                                                                                                          |
| --------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TVM_MODEL_DIR`       | `/shared/model`          | Dir holding `model.so` + `metadata.json`                                                                                                         |
| `TVM_MODEL_NAME`      | `model`                  | Name in `/v2/models/<name>`                                                                                                                      |
| `TVM_SERVE_PORT`      | `8080`                   | REST port                                                                                                                                        |
| `TVM_SERVE_GRPC_PORT` | `9000`                   | gRPC port                                                                                                                                        |
| `TVM_SERVE_WORKERS`   | `1`                      | Worker threads in the pool (each loads its own model copy); up to N concurrent inferences. Wired from the `tvm+serve` spec field `task.workers`. |
| `TVM_NUM_THREADS`     | every core (TVM default) | Threads of each worker's TVM thread pool, read by the TVM runtime itself. CORE sets it from the task CPU request divided by the workers.         |

## OpenInference v2 endpoints

REST (axum) and gRPC (`inference.GRPCInferenceService`) expose the same v2 surface.

| Concern         | REST                                                           | gRPC             |
| --------------- | -------------------------------------------------------------- | ---------------- |
| Server live     | `GET /v2/health/live`                                          | `ServerLive`     |
| Server ready    | `GET /v2/health/ready`                                         | `ServerReady`    |
| Server metadata | `GET /v2`                                                      | `ServerMetadata` |
| Model ready     | `GET /v2/models/:name/ready`                                   | `ModelReady`     |
| Model metadata  | `GET /v2/models/:name`                                         | `ModelMetadata`  |
| Infer           | `POST /v2/models/:name/infer` (and `/versions/:version/infer`) | `ModelInfer`     |

### Quantized models

For a model whose boundary tensors are `int8`/`uint8` (a TFLite full-integer export, a
QDQ ONNX), `GET /v2/models/:name` adds the affine params under the v2 `parameters` map,
read from `metadata.json`:

```json
{
  "name": "images",
  "datatype": "INT8",
  "shape": [1, 224, 224, 3],
  "parameters": { "scale": [0.003921568859368563], "zero_point": [-128] }
}
```

They are what lets a client quantize its input and dequantize the output —
`real = (q - zero_point) * scale`. Per-axis quantization yields more than one entry plus
a `quantized_dimension`. The server never uses them for inference (TVM baked the
quantization into `model.so`); it only forwards them. **Float models keep `parameters`
absent entirely**, so their response is unchanged. Note the gRPC `ModelMetadata` does
not carry them: the generated v2 proto has no `parameters` field on `TensorMetadata`.

Inputs are matched by name when every input is named and the names match the model,
otherwise **positionally** (in `metadata.inputs` order). Input `datatype` may be any of
the supported native dtypes — `FP32`, `FP64`, `INT8`/`INT16`/`INT32`/`INT64`,
`UINT8`/`UINT16`/`UINT32`/`UINT64`; `FP16` is not yet supported and is rejected with a
clear error. Message limits are raised well above the protocol defaults: REST body limit
**1 GiB**, gRPC max message **512 MB** (v2 tensors easily exceed the 2 MB / 4 MB
defaults). The gRPC server does **not** expose server reflection, so clients need the
`.proto` (`crates/tvm-serve/proto/grpc_predict_v2.proto`).

## Limitations

- **CPU only.** The VM is initialized on `kDLCPU`. No GPU.
- **Native dtypes, FP16 deferred.** `FP32`/`FP64`, `INT8`/`INT16`/`INT32`/`INT64`
  and `UINT8`/`UINT16`/`UINT32`/`UINT64` are supported. `FP16` needs an unsafe
  half path in Rust and is currently rejected with a clear error.
- **One model per pod, no batching, no metrics** — these remain future work.
  Concurrency is available within a pod via the worker pool (`TVM_SERVE_WORKERS`)
  and across pods via `replicas`.

## Relation to the Go runtime

`digitalhub-serverless` ships the equivalent Go runtime (`tvm-runtime-go`), a Nuclio
processor that drives the same TVM runtime through cgo. It is CORE's default serve
image. Both serve the same `model.so` with the same `TVM_MODEL_DIR` contract, ports,
env variables and OpenInference v2 surface, so switching between them needs no
change to the compile or serve flow.
