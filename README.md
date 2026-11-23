# DigitalHub TVM Runtime Rust

The image **`ghcr.io/scc-digitalhub/tvm-runtime-rust`** serves a model compiled by
[Apache TVM](https://tvm.apache.org/) with the **Open Inference Protocol v2**: REST on
port `8080` and gRPC on port `9000`. Inside it runs **`tvm-serve`**, a small Rust server
with no Python.

It is one of the two serve images for a DigitalHub CORE **`tvm+serve`** run. The default
one is **DigitalHub TVM Runtime Go**: both accept Open Inference v2 requests. Their size limits and extra response
fields are listed below.

Each worker loads its own `model.so` and `metadata.json`, then executes one
inference at a time. Workers provide parallel requests; `TVM_NUM_THREADS` controls
threads within each inference. No model-specific image is needed.
CPU remains the default, including ARM32, with no NVIDIA dependencies. Optional
[CUDA FP32 builds](#optional-cuda-fp32) use separate AMD64/ARM64 images.

## Quick start

Download a model compiled by `tvm+compile` for your machine and start the image:

```bash
dhcli download model -p my-project -n my-model-amd64 -d ./my-model   # model.so + metadata.json

docker run --rm -p 8080:8080 -p 9000:9000 \
  -v "$PWD/my-model:/shared/model" \
  -e TVM_MODEL_NAME=my-model \
  ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0-r2

curl http://localhost:8080/v2/models/my-model
```

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

Health endpoints return HTTP `200` with a JSON object (`{}`) on success, including
`Content-Type: application/json`, so they also work through the CORE proxy.

| What            | REST                                                        | gRPC             |
| --------------- | ----------------------------------------------------------- | ---------------- |
| Server live     | `GET /v2/health/live`                                       | `ServerLive`     |
| Server ready    | `GET /v2/health/ready`                                      | `ServerReady`    |
| Server metadata | `GET /v2`                                                   | `ServerMetadata` |
| Model ready     | `GET /v2/models/<name>/ready`                               | `ModelReady`     |
| Model metadata  | `GET /v2/models/<name>`                                     | `ModelMetadata`  |
| Inference       | `POST /v2/models/<name>/infer` (also `/versions/<v>/infer`) | `ModelInfer`     |

Read the metadata first and match its input names, datatypes, shapes and value
counts. For a model with `x: FP32 [1,4]`, a complete request is:

```bash
curl -X POST http://localhost:8080/v2/models/my-model/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"x","datatype":"FP32","shape":[1,4],"data":[1.0,2.0,3.0,4.0]}]}'
```

This example must be adapted for other signatures, including image models. Images
must be preprocessed by the client; the server consumes tensors. The versioned
inference route serves the same loaded model, not a model-version repository.

- **Inputs** are matched by name when every input has one and the names match the model,
  otherwise by position.
- **Data types**: `FP32`, `FP64`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`,
  `UINT32`, `UINT64`.
- **Size limits**: 1 GiB for a REST request, 512 MiB for a gRPC message.
- **Quantized models** (`int8` / `uint8` tensors): the REST model metadata adds `scale`,
  `zero_point` and `quantized_dimension` under `parameters`, so the client can convert the
  values (`real = (q - zero_point) * scale`).
- **Timing**: the REST response reports `inference_time_ms` in `parameters`.
  CUDA timing includes tensor transfers and waits for GPU completion; client
  end-to-end latency additionally includes transport and request/response encoding.
- **gRPC clients** need the proto `crates/tvm-serve/proto/grpc_predict_v2.proto` (no
  server reflection).

### Input validation

The runtime validates the model's datatype, rank and static dimensions before
queuing work for TVM. Negative metadata dimensions are dynamic; request dimensions
must be nonnegative. Input names still use the positional fallback described above.
Tensor validation failures return HTTP `400` / gRPC `INVALID_ARGUMENT`, including wrong element
counts, unsupported datatypes, nonfinite values and numeric overflow. Integers must
fit their declared datatype and cannot be fractional; integral JSON decimal and
exponent notation such as `1.0` remains accepted. Integer values retain full
`INT64`/`UINT64` precision without floating-point rounding during validation.

gRPC accepts either typed `contents` or little-endian `raw_input_contents`,
never both. Raw encoding requires exactly one buffer per input; malformed buffers
are rejected. True execution failures return HTTP `500` / gRPC `INTERNAL` with a
generic message; detailed native errors remain in the server logs.

These validation fixes are included in `0.26.0-r2`; the older `0.26.0-r1` image
does not include them. No model recompilation is needed when the TVM version
and revision remain unchanged. The `0.26.0-r2` release also handles `SIGTERM`
and Ctrl-C to stop the server.

## Which models it serves

At startup the server refuses the model, with a clear error, unless:

- `metadata.json` has the same `tvm_version` and `tvm_git_commit` as the TVM built into the
  image: compile with a DigitalHub TVM Toolkit image using that TVM version and commit;
- the model's Linux GNU/glibc host matches the image architecture: an LLVM CPU
  target, or a CUDA target with an explicit nested LLVM host in a CUDA-enabled build;
- `target_triple`, when present, agrees with the runtime, and the actual ELF library
  has the matching machine, 32/64-bit class and byte order;
- every input and output uses a supported data type.

Runtime image builds cover `linux/amd64` (Intel/AMD 64-bit), `linux/arm64`
(ARM 64-bit/AArch64) and `linux/arm/v7` (ARM 32-bit). CORE's `arm32` profile produces
an ARMv7 hard-float model requiring NEON, for the `linux/arm/v7` runtime.

Compile profiles are configured in CORE's `runtime-tvm.yaml`; the server does not
need that file or its profile names. It reads the effective target from the compiled
model. Older metadata without `target_triple` falls back to `target`; TVM version
and revision checks still apply. ARM32 requires the hard-float ABI.

These checks do not prove that the CPU supports every instruction selected by
`mcpu`/`mattr`. Deploy to a matching CPU. A newly configured compile target does not
automatically add a serving platform. CUDA requires the separate build below.

## Optional CUDA FP32

The CUDA build uses the upstream CUDA TVM runtime alongside its CPU host runtime.
Artifact metadata selects the backend; no new serving environment or device field
is needed. CUDA targets require an explicit GPU architecture and nested LLVM host,
for example the Jetson Orin target:

```json
{"kind":"cuda","arch":"sm_87","host":{"kind":"llvm","mtriple":"aarch64-linux-gnu"}}
```

Host triple, actual ELF architecture and TVM version/commit checks still apply.
A CPU build rejects CUDA artifacts. A CUDA build uses visible CUDA device 0 plus
the CPU VM host and fails when its CUDA backend, driver or device is unavailable;
it does not fall back to CPU. Inputs and outputs are copied and GPU work completes
before the response. Requests and supported protocol datatypes are unchanged; the
initial GPU validation scope is FP32. FP16/BFLOAT16 protocol inputs and outputs remain unsupported;
internal mixed precision kernels and accuracy require separate hardware qualification.

Build from the repository root with a matching, attested CUDA TVM build:

```bash
TVM_BACKEND=cuda TVM_HOME=/path/to/tvm-0.26.0 TVM_BUILD=/path/to/cuda-build \
  TAG=local/tvm-runtime-rust:0.26-cuda13.2.1 ./build-image.sh
```

The script enables the Cargo `cuda` feature and uses the pinned CUDA 13.2.1
Ubuntu 24.04 runtime base. Provenance must attest CPU/CUDA backends and the matching
hashes of `libtvm_runtime.so`, `libtvm_ffi.so` and `libtvm_runtime_cuda.so`.
For artifacts exported from the CUDA toolkit, add `TVM_EXPORTED_ARTIFACTS=1` and
point `TVM_BUILD` at the exported evidence directory; `TVM_HOME` must still contain
the matching pinned source. Commit, library-hash and CUDA cache checks remain active.
The final image needs no Python, `nvcc` or driver stubs. CUDA images support Linux
AMD64/ARM64; CPU builds retain ARMv7. The `local/...` tag is a development example,
not a published GHCR CUDA release.

On a configured AMD64 NVIDIA host, start a CUDA artifact with real device access:

```bash
docker run --rm --gpus all -p 8080:8080 -p 9000:9000 \
  -v "$PWD/my-cuda-model:/shared/model" -e TVM_MODEL_NAME=my-model \
  local/tvm-runtime-rust:0.26-cuda13.2.1
```

For Jetson, use the ARM64 image and GPU container configuration compatible with its
installed JetPack stack. Start with one worker; every worker has a separate model
copy and GPU memory allocation. CORE reuses `image`, GPU `profile`, `resources.gpu`
and `envs`. Set `so_model_path` to the versioned Model key returned by the selected
Compile Run. Selecting a profile does not choose or recompile a Model.

The intended Jetson baseline is JetPack 7.2 / L4T 39.2 / CUDA 13.2.1 / Ubuntu 24.04.
Actual Jetson and AMD64 NVIDIA execution, accuracy, tuning and performance remain
**NOT_RUN**. Tests without a GPU do not validate real GPU inference. See the
[CUDA guide](https://scc-digitalhub.github.io/docs/0.16/runtimes/tvm/#cuda-fp32-optional).

## Use from DigitalHub CORE

Set `RUNTIME_TVM_SERVE=ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0-r2`
on CORE to use Rust by default, or set `image` on one serve Run. TVM-specific settings belong to
the **Run**, while the Task supplies common Kubernetes settings.

The Function keeps the reference to the IR produced by build. Each Compile Run
records its own compiled Model key in `status.outputs.compiled_so`. The serve Run
requires that versioned key in `so_model_path`; CORE does not take it from the
Function. Another compile does not change the Model selected by an existing
serve Run. For example:

```yaml
kind: tvm+serve:run
spec:
  task: tvm+serve://my-project/<task-id>
  so_model_path: store://my-project/model/tvm-so/my-model-amd64:<compiled-model-id>
  image: ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0-r2
  served_name: my-model
  workers: 1
  resources: {cpu: "4", mem: 2Gi}
```

| Run option | Default | Effect |
| --- | --- | --- |
| `so_model_path` | required | Versioned `tvm-so` Model key from the selected Compile Run's `status.outputs.compiled_so`. |
| `served_name` | Function name | Model name in `/v2/models/<name>` and `TVM_MODEL_NAME`. |
| `image` | `RUNTIME_TVM_SERVE` | Runtime image for this Run. |
| `replicas` | `1` | Number of pods. |
| `workers` | `1` | Model copies and parallel requests per pod. |
| `service_type` | `ClusterIP` | Kubernetes Service type. |
| `service_name` | unset | Additional Service name, `<function>-<service_name>`; does not rename the model. |

CORE downloads the Model with an init container, sets the model/worker environment
and derives TVM threads from the Run's CPUs divided by workers (at least one).
`envs` can explicitly set `TVM_NUM_THREADS`. The Run `profile` selects scheduling
settings, not a compile target. Its node selector must match the Model architecture
and CPU requirements. The platform reports the endpoint in `status.inference_v2`.

Runtime builds cover AMD64, ARM64 and ARMv7. CORE's default init image covers only
AMD64/ARM64, so CORE serving on ARM32 also needs an ARMv7 init image. Standalone
Docker serving with a mounted Model does not use it. See the
[CORE guide](https://github.com/scc-digitalhub/digitalhub-core/tree/main/runtimes/runtime-tvm)
for the Function/Task/Run relationship and target configuration.

## Versions and release

The current image is `ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0-r2`, built
with Apache TVM `0.26.0`. To publish the latest `main` with the same TVM release,
run the image workflow, replacing `<new-image-tag>` with a new, unused image tag:

```bash
gh workflow run tvm-runtime-rust-image.yml \
  --repo scc-digitalhub/digitalhub-tvm-rust --ref main \
  -f tvm_version=0.26.0 -f image_tag='<new-image-tag>'
```

Manual publication accepts only `main`. It builds that run's source commit and publishes
`tvm-runtime-rust:<image_tag>`, plus `-amd64`, `-arm64` and `-armv7` tags. Use a new image
tag for each release to preserve earlier images. Reusing a tag replaces its images;
it does not move a Git tag. The image tag can contain at most 122 characters, leaving
room for the architecture suffix.

The existing Git tag workflow remains available: pushing `0.26.0` publishes image
`0.26.0` from that tagged commit with Apache TVM `0.26.0`; a tag `X.Y` selects TVM `X.Y.0`.
Merging into `main` does not publish images automatically.

For each architecture, `.github/workflows/tvm-runtime-rust-image.yml`:

1. compiles only the selected TVM runtime (`libtvm_runtime.so`, `libtvm_ffi.so`):
   natively on amd64 and arm64, cross-compiled for armv7;
2. builds `tvm-serve`, embedding the TVM version and commit (armv7 also applies
   `patches/tvm-ffi-rust-32bit.patch`);
3. runs the TVM runtime and inference transport tests on native amd64 and arm64 builds;
4. builds the image and checks the libraries inside it (`ldd` and SHA-256);
5. pushes `<tag>-<arch>`.

A last job verifies the source commit and TVM/FFI versions and revisions, then publishes
the multi-architecture tag from the exact image digests produced by this run.
`/opt/tvm/provenance.json` records the source repository and commit, image tag, TVM/FFI
revisions and library hashes. The label `org.digitalhub.source.revision` identifies the
runtime source commit; the existing TVM/FFI labels retain their meaning.

CUDA has a separate manual workflow, `tvm-runtime-rust-cuda-image.yml`, covering
AMD64/ARM64. Its `publish` input defaults to `false`; publication must be explicitly
requested from `main` and uses a distinct CUDA image tag. No CUDA release is
published by these development instructions, and the existing CPU tags are retained.

## Development

| Path               | Content                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| `crates/tvm-relax` | Library that loads `model.so`, runs the Relax VM and checks `metadata.json`.                                |
| `crates/tvm-serve` | The server: `main.rs` (configuration and REST), `worker.rs` (workers), `protocol.rs` (v2 types), `grpc.rs`. |
| `scripts/`         | `tvm-ffi-config`, which tells the `tvm-ffi` bindings where `libtvm_ffi.so` is.                              |
| `patches/`         | Fix applied to the `tvm-ffi` Rust bindings for 32-bit ARM.                                                  |

Building needs a local build of the same Apache TVM release:

```bash
TVM=~/tvm/src/tvm-0.26.0
ln -sfn "$TVM/3rdparty/tvm-ffi/rust" .tvm-ffi-rust   # Rust bindings of that release

export TVM_BUILD_DIR="$TVM/build" TVM_FFI_LIBDIR="$TVM/build/lib"
export PATH="$PWD/scripts:$PATH" LD_LIBRARY_PATH="$TVM/build/lib"
export TVM_VERSION=0.26.0 TVM_GIT_COMMIT="$(git -C "$TVM" rev-parse HEAD)"

cargo build --release --bin tvm-serve
```

Without `TVM_VERSION` and `TVM_GIT_COMMIT` the binary refuses every model.

To build a native image locally (Python 3.11+, Cargo, Docker and binutils required):

```bash
TVM_HOME=~/tvm/src/tvm-0.26.0 ./build-image.sh
```

The script verifies the TVM release, submodules and library hashes using
`scripts/verify-tvm-build.py`. The TVM build must include
`build/digitalhub-build-provenance.json`; `TVM_BUILD` overrides that build directory.
The image stays local unless `--push` is explicitly supplied.
The executable is selected from Cargo's build result, including `CARGO_TARGET_DIR`
and configured targets. Cargo normally reuses up-to-date artifacts. To rebuild from
scratch without deleting an existing cache, set `CARGO_TARGET_DIR` to a new directory.

For a manual CUDA binary build, add `--features cuda` to the Cargo command and use
the matching CUDA-enabled TVM libraries. A build or an unavailable-device check
does not substitute for repeated numerical inference on actual NVIDIA hardware.

## Limitations

- CPU is the default; optional CUDA FP32 requires a separate build and real NVIDIA device access.
- One model per pod; no automatic request batching and no metrics.
- `FP16` tensors are not supported yet.

## Security Policy

The current release is the supported version. Security fixes are released together with all other fixes in each new release.

If you discover a security vulnerability in this project, please do not open a public issue.

Instead, report it privately by emailing us at digitalhub@fbk.eu. Include as much detail as possible to help us understand and address the issue quickly and responsibly.

## Contributing

To report a bug or request a feature, please first check the existing issues to avoid duplicates. If none exist, open a new issue with a clear title and a detailed description, including any steps to reproduce if it's a bug.

To contribute code, start by forking the repository. Clone your fork locally and create a new branch for your changes. Make sure your commits follow the [Conventional Commits v1.0](https://www.conventionalcommits.org/en/v1.0.0/) specification to keep history readable and consistent.

Once your changes are ready, push your branch to your fork and open a pull request against the main branch. Be sure to include a summary of what you changed and why. If your pull request addresses an issue, mention it in the description (e.g., “Closes #123”).

Please note that new contributors may be asked to sign a Contributor License Agreement (CLA) before their pull requests can be merged. This helps us ensure compliance with open source licensing standards.

We appreciate contributions and help in improving the project!

## Authors

This project is developed and maintained by **DSLab – Fondazione Bruno Kessler**, with contributions from the open source community. A complete list of contributors is available in the project’s commit history and pull requests.

For questions or inquiries, please contact: [digitalhub@fbk.eu](mailto:digitalhub@fbk.eu)

## Copyright and license

Copyright © 2025 DSLab – Fondazione Bruno Kessler and individual contributors.

This project is licensed under the Apache License, Version 2.0.
You may not use this file except in compliance with the License. Ownership of contributions remains with the original authors and is governed by the terms of the Apache 2.0 License, including the requirement to grant a license to the project.

For Orin with the current ARM64 base, the host needs NVIDIA Container Toolkit
1.19.1 or newer (`nvidia-ctk --version`); 1.19.0 fails before the entrypoint in the
CUDA compatibility hook. See the [CUDA deployment guide](https://scc-digitalhub.github.io/docs/0.16/runtimes/tvm/#cuda-fp32-optional).

Local image builds retain an isolated context; `TVM_BUILD_CONTEXT` selects a new
directory. CPU publication rejects CUDA tags, CUDA tags require a `cuda13.2.1`
component, and publication checks both architecture tags and the final manifest.
Existing registry tags and failed lookups stop publication. Rebuilt releases need
new tags. These guards do not publish any image by themselves.
