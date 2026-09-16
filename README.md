# DigitalHub TVM Runtime Rust

The image **`ghcr.io/scc-digitalhub/tvm-runtime-rust`** serves a model compiled by
[Apache TVM](https://tvm.apache.org/) with the **Open Inference Protocol v2**: REST on
port `8080` and gRPC on port `9000`. Inside it runs **`tvm-serve`**, a small Rust server
with no Python.

It is one of the two serve images of the DigitalHub CORE **`tvm+serve`** task. The default
one is **DigitalHub TVM Runtime Go**: both behave the same and can be swapped.

```
 TVM_MODEL_DIR                     tvm-serve
 ├── model.so        ──load──►   worker 1 ─ model copy ─┐
 └── metadata.json               worker 2 ─ model copy ─┤◄── REST :8080
                                 ...                    │◄── gRPC :9000
                                 worker N ─ model copy ─┘
```

Nothing model-specific is baked into the image. At startup `tvm-serve` checks the model,
loads one copy per worker and starts both servers.

## Quick start

Download a model compiled by `tvm+compile` for your machine and start the image:

```bash
dhcli download model -p my-project -n my-model-x86 -d ./my-model   # model.so + metadata.json

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

| What            | REST                                                        | gRPC             |
| --------------- | ----------------------------------------------------------- | ---------------- |
| Server live     | `GET /v2/health/live`                                       | `ServerLive`     |
| Server ready    | `GET /v2/health/ready`                                      | `ServerReady`    |
| Server metadata | `GET /v2`                                                   | `ServerMetadata` |
| Model ready     | `GET /v2/models/<name>/ready`                               | `ModelReady`     |
| Model metadata  | `GET /v2/models/<name>`                                     | `ModelMetadata`  |
| Inference       | `POST /v2/models/<name>/infer` (also `/versions/<v>/infer`) | `ModelInfer`     |

```bash
curl -X POST http://localhost:8080/v2/models/my-model/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"images","datatype":"FP32","shape":[1,3,640,640],"data":[...]}]}'
```

- **Inputs** are matched by name when every input has one and the names match the model,
  otherwise by position.
- **Data types**: `FP32`, `FP64`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`,
  `UINT32`, `UINT64`.
- **Size limits**: 1 GiB for a REST request, 512 MB for a gRPC message.
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
- the model was compiled for a CPU (LLVM target) of the image architecture;
- every input and output uses a supported data type.

The image exists for `linux/amd64`, `linux/arm64` and `linux/arm/v7`, so the same command
runs on a Raspberry Pi.

## Use from DigitalHub CORE

Set `RUNTIME_TVM_SERVE=ghcr.io/scc-digitalhub/tvm-runtime-rust:0.26.0` on CORE to use it for
every serve, or `image` on a single `tvm+serve` run. CORE:

- downloads the `tvm-so` Model into `TVM_MODEL_DIR` with an init container;
- sets `TVM_MODEL_NAME`, `TVM_SERVE_WORKERS` and `TVM_NUM_THREADS` (the run CPUs divided by
  the workers);
- starts the pod on a node with the architecture of the model, where Kubernetes pulls the
  matching variant of the image.

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
| `scripts/`         | `tvm-ffi-config` for the `tvm-ffi` bindings, `verify-tvm-build.py` for the local image build.               |
| `patches/`         | Fix applied to the `tvm-ffi` Rust bindings for 32-bit ARM.                                                  |

Building needs a local build of the same Apache TVM release (for example with
`build-tvm.sh` of the DigitalHub TVM Toolkit):

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

To build the image locally, from `~/tvm/src/tvm-current` by default (`TVM_HOME` to change
it):

```bash
./build-image.sh            # builds tvm-runtime-rust:0.26
./build-image.sh --load     # ... and loads it into minikube
REGISTRY=registry.example.com ./build-image.sh --push
```

## Limitations

- CPU only, no GPU.
- One model per pod; no batching and no metrics.
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
