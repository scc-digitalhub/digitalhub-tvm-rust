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

## Quick start

Download a model compiled by `tvm+compile` for your machine and start the image:

```bash
dhcli download model -p my-project -n my-model-amd64 -d ./my-model   # model.so + metadata.json

docker run --rm -p 8080:8080 -p 9000:9000 \
  -v "$PWD/my-model:/shared/model" \
  -e TVM_MODEL_NAME=my-model \
  ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0

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

Health endpoints return HTTP `200` with an empty body on success. Do not require
a JSON response for these checks.

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
- **gRPC clients** need the proto `crates/tvm-serve/proto/grpc_predict_v2.proto` (no
  server reflection).

## Which models it serves

At startup the server refuses the model, with a clear error, unless:

- `metadata.json` has the same `tvm_version` and `tvm_git_commit` as the TVM built into the
  image: **compile with the DigitalHub TVM Toolkit of the same release**;
- the model was compiled for a Linux GNU/glibc CPU (LLVM target) of the image architecture;
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
automatically add a serving platform; CUDA/GPU inference is not implemented here.

## Use from DigitalHub CORE

Set `RUNTIME_TVM_SERVE=ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0`
on CORE to use Rust by default, or set `image` on one serve Run. TVM-specific settings belong to
the **Run**, while the Task supplies common Kubernetes settings. For example:

```yaml
kind: tvm+serve:run
spec:
  task: tvm+serve://my-project/<task-id>
  image: ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0
  served_name: my-model
  workers: 1
  resources: {cpu: "4", mem: 2Gi}
```

| Run option | Default | Effect |
| --- | --- | --- |
| `model_path` | Function `so_model` | Versioned `tvm-so` reference to serve; an explicit value wins. |
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

**The image tag is the git tag, and it is the Apache TVM version**: tag `0.26.0` builds
`tvm-runtime-rust:0.26.0` on Apache TVM `0.26.0`. Each architecture also gets its own tag:
`0.26.0-amd64`, `0.26.0-arm64` and `0.26.0-armv7`.

Push a tag `X.Y.Z` (or `X.Y`) and the GitHub Action
`.github/workflows/tvm-runtime-rust-image.yml` builds the image. For each architecture it:

1. compiles only the TVM runtime (`libtvm_runtime.so`, `libtvm_ffi.so`) of Apache TVM
   `vX.Y.Z`: natively on amd64 and arm64, cross-compiled for armv7;
2. builds `tvm-serve`, embedding the TVM version and commit (armv7 also applies
   `patches/tvm-ffi-rust-32bit.patch`);
3. builds the image and checks the libraries inside it (`ldd` and SHA-256);
4. pushes `<tag>-<arch>`.

A last job publishes the multi-architecture tag.

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

cargo test -p tvm-relax -p tvm-serve
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

## Limitations

- CPU only, no GPU.
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
