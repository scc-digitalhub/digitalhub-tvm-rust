//! Inference worker pool on dedicated OS threads.
//!
//! `tvm-ffi` types aren't Send/Sync and the VM isn't thread-safe, so each worker
//! owns its own model copy on its own thread; jobs cross threads carrying only
//! Send data. N workers drain one queue: N concurrent inferences, N model copies.
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, Mutex};
use tvm_ffi::{DLDataTypeCode, Tensor};
use tvm_relax::{Metadata, RelaxModel, TensorSpec};

use crate::protocol::TensorData;

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
pub enum ServeErr {
    NotFound(String),
    BadRequest(String),
    Internal(String),
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
        let meta_names: Vec<&str> = self.metadata.inputs.iter().map(|t| t.name.as_str()).collect();
        let mut inputs = inputs;
        let names_match = !meta_names.is_empty() && inputs.iter().all(|i| !i.name.is_empty()) && {
            let mut req: Vec<&str> = inputs.iter().map(|i| i.name.as_str()).collect();
            let mut meta = meta_names.clone();
            req.sort_unstable();
            meta.sort_unstable();
            req == meta
        };
        if names_match {
            inputs.sort_by_key(|i| meta_names.iter().position(|n| *n == i.name).unwrap_or(usize::MAX));
        }
        for (idx, i) in inputs.iter().enumerate() {
            crate::protocol::validate_shape(idx, &i.shape, i.data.len())
                .map_err(ServeErr::BadRequest)?;
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
                let model = match RelaxModel::load(&so_path, &entry) {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(()));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.to_string()));
                        return;
                    }
                };
                loop {
                    // Hold the lock only to dequeue, then drop it so a sibling can run.
                    let job = {
                        let mut guard = rx.blocking_lock();
                        guard.blocking_recv()
                    };
                    match job {
                        Some((inputs, reply)) => {
                            let _ = reply.send(run_one(&model, &inputs, &outputs_meta));
                        }
                        None => break,
                    }
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
        tensors.push(build_tensor(&inp.data, &inp.shape)?);
    }

    let outs = model.run(&tensors).map_err(|e| e.to_string())?;

    let mut result = Vec::with_capacity(outs.len());
    for (i, t) in outs.iter().enumerate() {
        // Output dtype comes from the tensor, not metadata (so int64 indices report INT64).
        let (data, datatype) = read_tensor(t)?;
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

// Read an output tensor into typed data + its v2 datatype, dispatching on the
// actual DLDataType. data_as_slice::<T> is safe because T matches that dtype.
fn read_tensor(t: &Tensor) -> Result<(TensorData, &'static str), String> {
    let dt = t.dtype();
    let (code, bits) = (dt.code, dt.bits);
    let fl = DLDataTypeCode::kDLFloat as u8;
    let si = DLDataTypeCode::kDLInt as u8;
    let ui = DLDataTypeCode::kDLUInt as u8;

    macro_rules! read {
        ($ty:ty, $variant:path, $v2:expr) => {{
            let s = t
                .data_as_slice::<$ty>()
                .map_err(|e| format!("output data_as_slice: {e:?}"))?;
            Ok(($variant(s.to_vec()), $v2))
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
