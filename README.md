# digitalhub-tvm-rust · native `tvm-serve` image

Native **Rust** serving of compiled **TVM Relax** models. This standalone project
builds the `tvm-serve` binary and packages it into the **`tvm-runtime-rust`**
container image that DigitalHub CORE launches for its **`tvm+serve`** task.

The image is **model-centric**: nothing model-specific is baked in. At startup
`tvm-serve` reads `model.so` + `metadata.json` from `$TVM_MODEL_DIR` and exposes
the model over the **Open Inference Protocol v2** (KServe) on **REST `:8080`** and
**gRPC `:9000`**. Inference runs directly on the TVM VirtualMachine driven from
Rust.

> This project was moved out of the CORE monorepo. It no longer lives under
> `runtime-tvm`, and it has no dependency on the old Kaniko multistage build, the
> in-tree `docker/` assets, or any `examples/` / `rebuild-images.sh` tooling —
> those are gone. The only artifact it produces is the `tvm-runtime-rust` image.

## Purpose

CORE's `tvm+serve` needs a base image that can take a freshly compiled Relax
`model.so` and serve it. `tvm-runtime-rust` is that image:

- a single self-contained `tvm-serve` binary + the TVM runtime `.so`s;
- model-agnostic — the model is injected at deploy time (init container), not
  baked;
- serves OpenInference v2 over REST and gRPC from the same process;
- **CPU only**, native dtypes (FP16 deferred).

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

### Crates

| Crate | Path | Role |
|-------|------|------|
| `tvm-relax` | `crates/tvm-relax` | The `RelaxModel` inference library. Loads `model.so` and drives the Relax VirtualMachine through raw `PackedFunc` calls over the `tvm-ffi` C ABI. `src/lib.rs` is the core. |
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
adds an rpath. (`cargo:rustc-link-arg` does not propagate from a dependency to the
binary crate, so the setup lives in the binary crate's `build.rs`.)

## Building the image

`./build-image.sh` compiles `tvm-serve` in release mode and packages it with the
TVM runtime `.so`s into `tvm-runtime-rust:<tag>`.

```bash
./build-image.sh            # build the image
./build-image.sh --load     # ...and `minikube image load` it
./build-image.sh --push     # ...and push to $REGISTRY
```

It requires a **locally-built TVM** — the script copies `libtvm_runtime.so` and
`libtvm_ffi.so` out of the TVM build tree and links `tvm-serve` against them.

| Env var | Default | Meaning |
|---------|---------|---------|
| `TVM_HOME` | `$HOME/tvm/src/tvm-current` | Root of the local TVM checkout/build |
| `TVM_BUILD` | `$TVM_HOME/build` | TVM build dir (must contain `lib/libtvm_runtime.so`) |
| `TVM_TAG` | derived from `TVM_HOME` (e.g. `tvm-0.25.0` -> `0.25`) | Image tag `major.minor`, matches the packaged TVM |
| `TAG` | `tvm-runtime-rust:$TVM_TAG` | Full image name:tag |
| `REGISTRY` | *(empty)* | Push prefix for `--push`: the pushed ref is `$REGISTRY/$TAG`; when empty the bare `$TAG` is pushed |

The Rust build itself resolves the TVM libs two ways: `build.rs` reads
`TVM_BUILD_DIR` (`build-image.sh` sets it to `$TVM_BUILD`), and `tvm-ffi-sys`'s
build invokes `tvm-ffi-config` — a shim in `scripts/tvm-ffi-config` (put it on
`PATH`, override with `TVM_FFI_LIBDIR`). The `tvm-ffi` Rust
bindings are the ones bundled with the active TVM version and must be ABI-coherent
with the `.so`s being linked.

The image is based on **`ubuntu:24.04`** (needs the build host's glibc ≥ 2.38 /
`GLIBCXX_3.4.32`). Artifacts are x86_64/glibc; other architectures require
rebuilding the binary + `.so`s on that arch.

## How CORE uses it

CORE's `TvmServeRunner` deploys `tvm-runtime-rust` as a Kubernetes Deployment. It
is a build-free, model-injection pattern:

```
             (init container)                         (tvm-serve container)
  S3 store:// .so folder  ──download──▶  <home>/model  ──TVM_MODEL_DIR──▶  tvm-serve
  model.so + metadata.json                                                 REST :8080
                                                                           gRPC :9000
```

- An **init container** downloads the compiled `.so` Model folder (`model.so` +
  `metadata.json`, optional `params.bin`) from S3 into `<home-dir>/model`.
- `tvm-serve` is pointed there via **`TVM_MODEL_DIR`**, with `TVM_MODEL_NAME` set
  to the served model name (used in `/v2/models/<name>`).
- The Deployment declares service ports **8080** (REST) and **9000** (gRPC).

The base serve image is configured by **`runtime.tvm.serve`** (env
`RUNTIME_TVM_SERVE`), defaulting to
`ghcr.io/scc-digitalhub/tvm-runtime-rust:0.25`. A `tvm+serve` task can override it
per-run via `task.image`.

### Runtime env config (read by `tvm-serve`)

| Env var | Default | Meaning |
|---------|---------|---------|
| `TVM_MODEL_DIR` | `/shared/model` (image: `/model`) | Dir holding `model.so` + `metadata.json` |
| `TVM_MODEL_NAME` | `model` | Name in `/v2/models/<name>` |
| `TVM_SERVE_PORT` | `8080` | REST port |
| `TVM_SERVE_GRPC_PORT` | `9000` | gRPC port |
| `TVM_SERVE_WORKERS` | `1` | Worker threads in the pool (each loads its own model copy); up to N concurrent inferences. Wired from the `tvm+serve` spec field `task.workers`. |

## OpenInference v2 endpoints

REST (axum) and gRPC (`inference.GRPCInferenceService`) expose the same v2 surface.

| Concern | REST | gRPC |
|---------|------|------|
| Server live | `GET /v2/health/live` | `ServerLive` |
| Server ready | `GET /v2/health/ready` | `ServerReady` |
| Server metadata | `GET /v2` | `ServerMetadata` |
| Model ready | `GET /v2/models/:name/ready` | `ModelReady` |
| Model metadata | `GET /v2/models/:name` | `ModelMetadata` |
| Infer | `POST /v2/models/:name/infer` (and `/versions/:version/infer`) | `ModelInfer` |

Inputs are matched **positionally** (client sends tensors in `metadata.inputs`
order). Input `datatype` may be any of the supported native dtypes — `FP32`,
`FP64`, `INT8`/`INT16`/`INT32`/`INT64`, `UINT8`/`UINT16`/`UINT32`/`UINT64`; `FP16`
is not yet supported and is rejected with a clear error. Message limits are raised well above the
protocol defaults: REST body limit **1 GiB**, gRPC max message **512 MB** (v2
tensors easily exceed the 2 MB / 4 MB defaults). The gRPC server does **not**
expose server reflection, so clients need the `.proto`
(`crates/tvm-serve/proto/grpc_predict_v2.proto`).

## End-to-end testing

The E2E test tooling lives in the CORE repo, next to the runtime that deploys
this image: `digitalhub-core/runtimes/runtime-tvm/test_infer/run_infer.py`. It
discovers the running `tvm+serve` from the CORE API, downloads a test image,
calls inference over **REST or gRPC** (the KServe proto is bundled there), and
saves the picture with the decoded bounding boxes.

```bash
cd ../digitalhub-core/runtimes/runtime-tvm/test_infer
python3 run_infer.py                 # REST
python3 run_infer.py --mode grpc     # gRPC
```

## Limitations

- **CPU only.** The VM is initialized on `kDLCPU`. No GPU.
- **Native dtypes, FP16 deferred.** `FP32`/`FP64`, `INT8`/`INT16`/`INT32`/`INT64`
  and `UINT8`/`UINT16`/`UINT32`/`UINT64` are supported. `FP16` needs an unsafe
  half path in Rust and is currently rejected with a clear error.
- **One model per pod, no batching, no metrics** — these remain future work.
  Concurrency is available within a pod via the worker pool (`TVM_SERVE_WORKERS`)
  and across pods via `replicas`.

## Relation to the Go runtime

The equivalent in the older stack is a native Nuclio runtime in
**`digitalhub-serverless`** (Go), which does the same job — load a compiled TVM
model and serve OpenInference v2 — but drives the TVM C runtime through cgo. This
project is the Rust-native alternative: the same serving contract without cgo,
packaged as the `tvm-runtime-rust` image that `tvm+serve` launches by default.
