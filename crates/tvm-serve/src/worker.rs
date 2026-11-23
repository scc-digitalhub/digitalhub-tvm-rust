/*
 * SPDX-FileCopyrightText: © 2025 DSLab - Fondazione Bruno Kessler
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inference worker pool on dedicated OS threads.
//!
//! `tvm-ffi` types aren't Send/Sync and the VM isn't thread-safe, so each worker
//! owns its own model copy on its own thread; jobs cross threads carrying only
//! Send data. N workers drain one queue: N concurrent inferences, N model copies.
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, Mutex};
use tvm_ffi::{AnyView, DLDataTypeCode, DLDeviceType, Function, Tensor};
use tvm_relax::{Metadata, RelaxModel, TensorSpec};

use crate::protocol::TensorData;

const RUNTIME_TVM_VERSION: &str = match option_env!("TVM_VERSION") {
    Some(version) => version,
    None => "unknown",
};
const RUNTIME_TVM_GIT_COMMIT: &str = match option_env!("TVM_GIT_COMMIT") {
    Some(commit) => commit,
    None => "unknown",
};

pub struct InferInput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: TensorData,
}

pub struct InferOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    pub data: TensorData,
}

type Reply = oneshot::Sender<Result<Vec<InferOutput>, String>>;
type Job = (Vec<InferInput>, Reply);

// Coarse error kind so REST and gRPC each map to their own status from `serve_infer`.
#[derive(Debug)]
pub enum ServeErr {
    NotFound(String),
    BadRequest(String),
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_tensor_round_trip_preserves_supported_types() {
        let inputs = [
            TensorData::F32(vec![1.25, -2.5]),
            TensorData::F64(vec![1.25, -2.5]),
            TensorData::I8(vec![i8::MIN, i8::MAX]),
            TensorData::I16(vec![i16::MIN, i16::MAX]),
            TensorData::I32(vec![i32::MIN, i32::MAX]),
            TensorData::I64(vec![i64::MIN, i64::MAX]),
            TensorData::U8(vec![0, u8::MAX]),
            TensorData::U16(vec![0, u16::MAX]),
            TensorData::U32(vec![0, u32::MAX]),
            TensorData::U64(vec![0, u64::MAX]),
        ];
        for data in inputs {
            let tensor = build_tensor(&data, &[1, 2]).unwrap();
            let (actual, dtype) = read_tensor(&tensor).unwrap();
            assert_eq!(actual.datatype(), dtype);
            assert_eq!(actual.to_json(), data.to_json());

            let shape = tvm_ffi::Shape::from(&[1i64][..]);
            let dtype = tensor.dtype();
            let view: Tensor = Function::get_global("runtime.TVMTensorCreateView")
                .unwrap()
                .call_packed(&[
                    AnyView::from(&tensor),
                    AnyView::from(&shape),
                    AnyView::from(&dtype),
                    AnyView::from(&u64::from(dtype.bits / 8)),
                ])
                .unwrap()
                .try_into()
                .unwrap();
            let (actual, _) = read_tensor(&view).unwrap();
            assert_eq!(actual.to_json(), serde_json::json!([data.to_json()[1]]));
        }
    }

    /// Validate the pinned runtime's FFI ABI on CPU, including copying into a
    /// TVM-owned allocation. CUDA staging uses these same registered functions.
    #[test]
    fn standard_tvm_tensor_transfer_globals_work_without_gpu() {
        use tvm_ffi::{AnyView, DLDevice, DLDeviceType, Function, Shape};
        let source = Tensor::from_slice(&[1.25f32, -2.5], &[1, 2]).unwrap();
        let shape = Shape::from(source.shape());
        let dtype = source.dtype();
        let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
        let alloc = Function::get_global("runtime.TVMTensorAllocWithScope").unwrap();
        let target: Tensor = alloc
            .call_packed(&[
                AnyView::from(&shape),
                AnyView::from(&dtype),
                AnyView::from(&device),
                AnyView::new(),
            ])
            .unwrap()
            .try_into()
            .unwrap();
        Function::get_global("runtime.TVMTensorCopyFromTo")
            .unwrap()
            .call_tuple((&source, &target))
            .unwrap();
        Function::get_global("runtime.Device_StreamSync")
            .unwrap()
            .call_packed(&[AnyView::from(&device), AnyView::from(&0i64)])
            .unwrap();
        assert_eq!(target.data_as_slice::<f32>().unwrap(), &[1.25, -2.5]);
        assert_eq!(target.shape(), &[1, 2]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_image_fails_clearly_without_a_visible_gpu() {
        if std::env::var_os("TVM_TEST_REQUIRE_NO_CUDA").is_none() {
            return;
        }
        let error = RelaxModel::load("/no/model.so", "main", tvm_relax::Backend::Cuda)
            .err()
            .expect("this test requires a GPU-less runner");
        assert!(
            error.to_string().contains("CUDA device 0 is unavailable"),
            "{error}"
        );
    }

    /// Explicit hardware gate: NOT_RUN in ordinary CPU/no-GPU test runs.
    /// The fixture contains inputs/outputs with name, shape and FP32 data vectors.
    #[tokio::test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires a visible NVIDIA GPU, precompiled CUDA model and FP32 reference fixture"]
    async fn cuda_fp32_fixture_copies_inputs_and_outputs() {
        #[derive(serde::Deserialize)]
        struct FixtureTensor {
            name: String,
            shape: Vec<i64>,
            data: Vec<f32>,
        }
        #[derive(serde::Deserialize)]
        struct Fixture {
            inputs: Vec<FixtureTensor>,
            outputs: Vec<FixtureTensor>,
            #[serde(default = "tolerance")]
            atol: f32,
            #[serde(default = "tolerance")]
            rtol: f32,
        }
        fn tolerance() -> f32 {
            1e-5
        }
        let directory = std::env::var("TVM_TEST_CUDA_MODEL_DIR")
            .expect("TVM_TEST_CUDA_MODEL_DIR must contain model.so and metadata.json");
        let data_path = std::env::var("TVM_TEST_CUDA_DATA")
            .unwrap_or_else(|_| format!("{directory}/cuda-test.json"));
        let fixture: Fixture =
            serde_json::from_str(&std::fs::read_to_string(data_path).unwrap()).unwrap();
        assert!(!fixture.inputs.is_empty() && !fixture.outputs.is_empty());
        assert!(fixture.atol.is_finite() && fixture.atol >= 0.0);
        assert!(fixture.rtol.is_finite() && fixture.rtol >= 0.0);
        let metadata = Metadata::from_file(&format!("{directory}/metadata.json")).unwrap();
        assert_eq!(
            metadata
                .validate_runtime(
                    RUNTIME_TVM_VERSION,
                    RUNTIME_TVM_GIT_COMMIT,
                    std::env::consts::ARCH
                )
                .unwrap(),
            tvm_relax::Backend::Cuda
        );
        let handle = start(&directory, "cuda-fixture".into(), 1).unwrap();
        // Repeated requests also exercise device tensor ownership and pooled VM allocations.
        for _ in 0..3 {
            let inputs = fixture
                .inputs
                .iter()
                .map(|input| InferInput {
                    name: input.name.clone(),
                    shape: input.shape.clone(),
                    data: TensorData::F32(input.data.clone()),
                })
                .collect();
            let outputs = handle
                .serve_infer("cuda-fixture", inputs)
                .await
                .unwrap_or_else(|error| panic!("CUDA fixture inference failed: {error:?}"));
            assert_eq!(outputs.len(), fixture.outputs.len());
            for (actual, expected) in outputs.iter().zip(&fixture.outputs) {
                assert_eq!(actual.name, expected.name);
                assert_eq!(actual.shape, expected.shape);
                let TensorData::F32(values) = &actual.data else {
                    panic!("expected FP32 output")
                };
                assert_eq!(values.len(), expected.data.len());
                for (i, (&value, &reference)) in values.iter().zip(&expected.data).enumerate() {
                    assert!(value.is_finite() && reference.is_finite());
                    assert!(
                        (value - reference).abs() <= fixture.atol + fixture.rtol * reference.abs(),
                        "output {}[{i}]: {value} != {reference}",
                        expected.name
                    );
                }
            }
        }
    }

    fn handle() -> Handle {
        let (tx, _) = mpsc::channel(1);
        Handle {
            tx,
            model_name: "test".into(),
            metadata: serde_json::from_value(serde_json::json!({
                "entry": "main", "inputs": [{"name": "x", "shape": [1, 2], "dtype": "float32"}],
                "outputs": []
            }))
            .unwrap(),
        }
    }

    #[tokio::test]
    async fn incompatible_signature_is_a_client_error() {
        for input in [
            InferInput {
                name: "x".into(),
                shape: vec![1, 2],
                data: TensorData::F32(vec![1.]),
            },
            InferInput {
                name: "x".into(),
                shape: vec![-1, 2],
                data: TensorData::F32(vec![1., 2.]),
            },
            InferInput {
                name: "x".into(),
                shape: vec![i64::MAX, 2],
                data: TensorData::F32(vec![1., 2.]),
            },
            InferInput {
                name: "x".into(),
                shape: vec![1, 2],
                data: TensorData::F32(vec![f32::NAN, f32::INFINITY]),
            },
            InferInput {
                name: "x".into(),
                shape: vec![2, 1],
                data: TensorData::F32(vec![1., 2.]),
            },
            InferInput {
                name: "x".into(),
                shape: vec![2],
                data: TensorData::F32(vec![1., 2.]),
            },
            InferInput {
                name: "x".into(),
                shape: vec![1, 2],
                data: TensorData::I32(vec![1, 2]),
            },
        ] {
            assert!(matches!(
                handle().serve_infer("test", vec![input]).await,
                Err(ServeErr::BadRequest(_))
            ));
        }
    }

    #[tokio::test]
    async fn wrong_arity_is_rejected_before_dispatch() {
        for count in [0, 2] {
            let inputs = (0..count)
                .map(|_| InferInput {
                    name: "x".into(),
                    shape: vec![1, 2],
                    data: TensorData::F32(vec![1., 2.]),
                })
                .collect();
            assert!(matches!(
                handle().serve_infer("test", inputs).await,
                Err(ServeErr::BadRequest(_))
            ));
        }
    }

    #[tokio::test]
    async fn nonfinite_fp64_values_are_rejected_before_dispatch() {
        let mut handle = handle();
        handle.metadata.inputs[0].dtype = "float64".into();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let input = InferInput {
                name: "x".into(),
                shape: vec![1, 2],
                data: TensorData::F64(vec![1., value]),
            };
            assert!(matches!(
                handle.serve_infer("test", vec![input]).await,
                Err(ServeErr::BadRequest(_))
            ));
        }
    }

    #[tokio::test]
    async fn multiple_inputs_reorder_only_for_an_exact_name_match() {
        for first_name in ["x", "", "unknown", "y"] {
            let mut handle = handle();
            handle.metadata.inputs = serde_json::from_value(serde_json::json!([
                {"name": "x", "shape": [1], "dtype": "float32"},
                {"name": "y", "shape": [2], "dtype": "int32"}
            ]))
            .unwrap();
            let (tx, mut rx) = mpsc::channel(1);
            handle.tx = tx;
            let worker = tokio::spawn(async move {
                let (inputs, reply) = rx.recv().await.unwrap();
                assert_eq!(inputs[0].shape, vec![1]);
                assert_eq!(inputs[1].shape, vec![2]);
                reply.send(Ok(vec![])).ok().unwrap();
            });
            let x = InferInput {
                name: first_name.into(),
                shape: vec![1],
                data: TensorData::F32(vec![1.]),
            };
            let y = InferInput {
                name: "y".into(),
                shape: vec![2],
                data: TensorData::I32(vec![2, 3]),
            };
            let inputs = if first_name == "x" {
                vec![y, x]
            } else {
                vec![x, y]
            };
            assert!(handle.serve_infer("test", inputs).await.is_ok());
            worker.await.unwrap();
        }
    }

    #[tokio::test]
    async fn valid_positional_and_dynamic_inputs_reach_worker() {
        for name in ["x", "", "unknown"] {
            let mut handle = handle();
            handle.metadata.inputs[0].shape[0] = -1;
            let (tx, mut rx) = mpsc::channel(1);
            handle.tx = tx;
            let worker = tokio::spawn(async move {
                let (inputs, reply) = rx.recv().await.unwrap();
                assert_eq!(inputs[0].shape, vec![1, 2]);
                reply.send(Ok(vec![])).ok().unwrap();
            });
            let input = InferInput {
                name: name.into(),
                shape: vec![1, 2],
                data: TensorData::F32(vec![1., 2.]),
            };
            assert!(handle.serve_infer("test", vec![input]).await.is_ok());
            worker.await.unwrap();
        }
    }
}

pub struct Handle {
    tx: mpsc::Sender<Job>,
    pub metadata: Metadata,
    pub model_name: String,
}

impl Handle {
    /// Validate a v2 request against the model signature, then run it. The one
    /// path both transports share; they differ only in wire decoding/encoding.
    pub async fn serve_infer(
        &self,
        name: &str,
        inputs: Vec<InferInput>,
    ) -> Result<Vec<InferOutput>, ServeErr> {
        let desc: Vec<String> = inputs
            .iter()
            .map(|i| format!("{}:{:?}", i.name, i.shape))
            .collect();
        eprintln!(
            "[tvm-serve] infer req model={name} inputs=[{}]",
            desc.join(", ")
        );
        let result = self.serve_infer_inner(name, inputs).await;
        // Validation rejections are only visible here; runtime errors are logged by the worker thread.
        if let Err(ServeErr::NotFound(m) | ServeErr::BadRequest(m)) = &result {
            eprintln!("[tvm-serve] infer rejected model={name}: {m}");
        }
        result
    }

    async fn serve_infer_inner(
        &self,
        name: &str,
        inputs: Vec<InferInput>,
    ) -> Result<Vec<InferOutput>, ServeErr> {
        if name != self.model_name {
            return Err(ServeErr::NotFound(format!("model '{name}' not found")));
        }
        if inputs.is_empty() {
            return Err(ServeErr::BadRequest("no input tensors".to_string()));
        }
        let expected = self.metadata.inputs.len();
        if inputs.len() != expected {
            return Err(ServeErr::BadRequest(format!(
                "expected {expected} input tensor(s), got {}",
                inputs.len()
            )));
        }
        // When every input is named and the name set matches exactly, reorder to
        // metadata order; otherwise fall back to positional matching.
        let meta_names: Vec<&str> = self
            .metadata
            .inputs
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        let mut inputs = inputs;
        let names_match = !meta_names.is_empty() && inputs.iter().all(|i| !i.name.is_empty()) && {
            let mut req: Vec<&str> = inputs.iter().map(|i| i.name.as_str()).collect();
            let mut meta = meta_names.clone();
            req.sort_unstable();
            meta.sort_unstable();
            req == meta
        };
        if names_match {
            inputs.sort_by_key(|i| {
                meta_names
                    .iter()
                    .position(|n| *n == i.name)
                    .unwrap_or(usize::MAX)
            });
        }
        for (idx, i) in inputs.iter().enumerate() {
            crate::protocol::validate_shape(idx, &i.shape, i.data.len())
                .map_err(ServeErr::BadRequest)?;
            let finite = match &i.data {
                TensorData::F32(values) => values.iter().all(|v| v.is_finite()),
                TensorData::F64(values) => values.iter().all(|v| v.is_finite()),
                _ => true,
            };
            if !finite {
                return Err(ServeErr::BadRequest(format!(
                    "input[{idx}]: values must be finite"
                )));
            }
            let spec = &self.metadata.inputs[idx];
            let datatype = crate::protocol::tvm_to_v2_dtype(&spec.dtype);
            if i.data.datatype() != datatype {
                return Err(ServeErr::BadRequest(format!(
                    "input '{}': expected datatype {datatype}, got {}",
                    spec.name,
                    i.data.datatype()
                )));
            }
            if i.shape.len() != spec.shape.len()
                || spec
                    .shape
                    .iter()
                    .zip(&i.shape)
                    .any(|(&expected, &actual)| expected >= 0 && expected != actual)
            {
                return Err(ServeErr::BadRequest(format!(
                    "input '{}': expected shape {:?}, got {:?}",
                    spec.name, spec.shape, i.shape
                )));
            }
        }
        self.infer(inputs).await.map_err(ServeErr::Internal)
    }

    /// Hand the already-validated inputs to the worker thread and await its reply.
    async fn infer(&self, inputs: Vec<InferInput>) -> Result<Vec<InferOutput>, String> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send((inputs, tx))
            .await
            .map_err(|_| "worker unavailable".to_string())?;
        rx.await
            .map_err(|_| "worker terminated without reply".to_string())?
    }
}

/// Starts the worker pool; blocks until every worker has loaded its own copy of
/// the model so `ready` is real. `workers` is clamped to at least 1.
pub fn start(model_dir: &str, model_name: String, workers: usize) -> anyhow::Result<Handle> {
    let meta = Metadata::from_file(&format!("{model_dir}/metadata.json"))?;
    let backend = meta.validate_runtime(
        RUNTIME_TVM_VERSION,
        RUNTIME_TVM_GIT_COMMIT,
        std::env::consts::ARCH,
    )?;
    // Fail fast at startup on an unservable dtype rather than per-request; FP16/bool deferred.
    const SUPPORTED: &[&str] = &[
        "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
        "uint64",
    ];
    for t in meta.inputs.iter().chain(meta.outputs.iter()) {
        if !t.dtype.is_empty() && !SUPPORTED.contains(&t.dtype.as_str()) {
            anyhow::bail!(
                "model tensor '{}' declares dtype '{}': not supported by this serve image \
                 (FP16 and others are deferred)",
                t.name,
                t.dtype
            );
        }
    }
    let so_path = format!("{model_dir}/model.so");
    tvm_relax::validate_library(&so_path, std::env::consts::ARCH)?;
    let workers = workers.max(1);

    // One job queue, N consumers via a Mutex around the receiver (held only to
    // dequeue). A std-sync channel reports each worker's load outcome so `start`
    // can block until the pool is warm.
    let (tx, rx) = mpsc::channel::<Job>(64);
    let rx = Arc::new(Mutex::new(rx));
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    for w in 0..workers {
        let rx = rx.clone();
        let so_path = so_path.clone();
        let entry = meta.entry.clone();
        let outputs_meta = meta.outputs.clone();
        let ready_tx = ready_tx.clone();
        std::thread::Builder::new()
            .name(format!("tvm-infer-{w}"))
            .spawn(move || {
                let model = match RelaxModel::load(&so_path, &entry, backend) {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(()));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.to_string()));
                        return;
                    }
                };
                while let Some((inputs, reply)) = {
                    let mut receiver = rx.blocking_lock();
                    receiver.blocking_recv()
                } {
                    let started = std::time::Instant::now();
                    let result = run_one(&model, &inputs, &outputs_meta);
                    match &result {
                        Ok(outs) => {
                            let shapes: Vec<String> = outs
                                .iter()
                                .map(|o| format!("{}:{:?} {}", o.name, o.shape, o.datatype))
                                .collect();
                            eprintln!(
                                "[tvm-serve] infer done worker={w} run={}ms outputs=[{}]",
                                started.elapsed().as_millis(),
                                shapes.join(", ")
                            );
                        }
                        Err(e) => eprintln!("[tvm-serve] infer error worker={w}: {e}"),
                    }
                    let _ = reply.send(result);
                }
            })?;
    }
    drop(ready_tx); // only the worker threads keep senders now

    // Block until every worker has loaded its model (or one failed).
    for _ in 0..workers {
        ready_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("a worker thread did not start"))?
            .map_err(|e| anyhow::anyhow!("model load failed: {e}"))?;
    }

    Ok(Handle {
        tx,
        metadata: meta,
        model_name,
    })
}

fn run_one(
    model: &RelaxModel,
    inputs: &[InferInput],
    outputs_meta: &[TensorSpec],
) -> Result<Vec<InferOutput>, String> {
    // Re-validate the shape (defense in depth) so a bad shape can't crash tensor construction.
    let mut tensors = Vec::with_capacity(inputs.len());
    for (i, inp) in inputs.iter().enumerate() {
        crate::protocol::validate_shape(i, &inp.shape, inp.data.len())?;
        let cpu = build_tensor(&inp.data, &inp.shape)?;
        tensors.push(model.input_from_cpu(cpu).map_err(|e| e.to_string())?);
    }

    let outs = model.run(&tensors).map_err(|e| e.to_string())?;

    let mut result = Vec::with_capacity(outs.len());
    for (i, t) in outs.iter().enumerate() {
        // Output dtype comes from the tensor, not metadata (so int64 indices report INT64).
        let cpu = model.output_to_cpu(t).map_err(|e| e.to_string())?;
        let (data, datatype) = read_tensor(&cpu)?;
        let name = outputs_meta
            .get(i)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| format!("output_{i}"));
        result.push(InferOutput {
            name,
            shape: t.shape().to_vec(),
            datatype: datatype.to_string(),
            data,
        });
    }
    Ok(result)
}

// Build a CPU TVM tensor from typed data, dispatching on the variant.
fn build_tensor(data: &TensorData, shape: &[i64]) -> Result<Tensor, String> {
    let r = match data {
        TensorData::F32(v) => Tensor::from_slice(v, shape),
        TensorData::F64(v) => Tensor::from_slice(v, shape),
        TensorData::I8(v) => Tensor::from_slice(v, shape),
        TensorData::I16(v) => Tensor::from_slice(v, shape),
        TensorData::I32(v) => Tensor::from_slice(v, shape),
        TensorData::I64(v) => Tensor::from_slice(v, shape),
        TensorData::U8(v) => Tensor::from_slice(v, shape),
        TensorData::U16(v) => Tensor::from_slice(v, shape),
        TensorData::U32(v) => Tensor::from_slice(v, shape),
        TensorData::U64(v) => Tensor::from_slice(v, shape),
    };
    r.map_err(|e| format!("Tensor::from_slice: {e:?}"))
}

// TVM's copy API respects byte_offset and rejects unsupported layouts. The pinned
// Rust data_as_slice implementation ignores byte_offset, even for contiguous views.
fn read_tensor(t: &Tensor) -> Result<(TensorData, &'static str), String> {
    let dt = t.dtype();
    if t.device().device_type != DLDeviceType::kDLCPU || dt.lanes != 1 {
        return Err("output must be a CPU tensor with one dtype lane".into());
    }
    let copy = Function::get_global("runtime.TVMTensorCopyToBytes")
        .map_err(|e| format!("output copy function: {e:?}"))?;
    let (code, bits) = (dt.code, dt.bits);
    let fl = DLDataTypeCode::kDLFloat as u8;
    let si = DLDataTypeCode::kDLInt as u8;
    let ui = DLDataTypeCode::kDLUInt as u8;

    macro_rules! read {
        ($ty:ty, $variant:path, $v2:expr) => {{
            let mut values = vec![0 as $ty; t.numel()];
            let bytes = values
                .len()
                .checked_mul(std::mem::size_of::<$ty>())
                .ok_or_else(|| "output byte size overflow".to_string())?;
            if bytes != 0 {
                copy.call_packed(&[
                    AnyView::from(t),
                    AnyView::from(&values.as_mut_ptr().cast::<std::ffi::c_void>()),
                    AnyView::from(&bytes),
                ])
                .map_err(|e| format!("output TVMTensorCopyToBytes: {e:?}"))?;
            }
            Ok(($variant(values), $v2))
        }};
    }

    match (code, bits) {
        (c, 32) if c == fl => read!(f32, TensorData::F32, "FP32"),
        (c, 64) if c == fl => read!(f64, TensorData::F64, "FP64"),
        (c, 8) if c == si => read!(i8, TensorData::I8, "INT8"),
        (c, 16) if c == si => read!(i16, TensorData::I16, "INT16"),
        (c, 32) if c == si => read!(i32, TensorData::I32, "INT32"),
        (c, 64) if c == si => read!(i64, TensorData::I64, "INT64"),
        (c, 8) if c == ui => read!(u8, TensorData::U8, "UINT8"),
        (c, 16) if c == ui => read!(u16, TensorData::U16, "UINT16"),
        (c, 32) if c == ui => read!(u32, TensorData::U32, "UINT32"),
        (c, 64) if c == ui => read!(u64, TensorData::U64, "UINT64"),
        _ => Err(format!(
            "unsupported output dtype (code={code} bits={bits}); FP16/others deferred"
        )),
    }
}
