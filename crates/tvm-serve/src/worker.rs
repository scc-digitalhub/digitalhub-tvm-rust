//! Inference worker pool on dedicated OS threads.
//!
//! `tvm-ffi` types (Module/Function/Tensor) are not Send/Sync and the VM is
//! not thread-safe. Each worker loads its OWN copy of the model on its own
//! thread and never crosses threads; axum sends jobs over a shared channel
//! carrying only Send data (Vec<f32> + shape). N workers (`TVM_SERVE_WORKERS`)
//! drain the same queue, so up to N inferences run concurrently at the cost of
//! N model copies in memory.
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, Mutex};
use tvm_ffi::{DLDataTypeCode, Tensor};
use tvm_relax::{Metadata, RelaxModel, TensorSpec};

use crate::protocol::TensorData;

pub struct InferInput {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: TensorData,
}

pub struct InferOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    pub data: TensorData,
}

// A queued job: the input tensors plus the one-shot channel to answer on. A plain
// tuple, not an enum, since there is only ever one kind of job.
type Reply = oneshot::Sender<Result<Vec<InferOutput>, String>>;
type Job = (Vec<InferInput>, Reply);

// Coarse error kind so the REST and gRPC handlers can each map a validation or
// inference failure to their own status type from one shared path (`serve_infer`).
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
    /// Validate a v2 request against the model signature, then run it. The single
    /// path both transports share — they only differ in wire decoding/encoding.
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
        // Match by name when we can: the v2 spec identifies tensors by name, so if
        // every input is named and the set matches the model's input names exactly,
        // reorder to metadata order. Otherwise fall back to positional matching.
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
            crate::protocol::validate_input(idx, &i.datatype, &i.shape, i.data.len())
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
    // Fail fast at startup for a model whose declared dtypes this image can't
    // serve, rather than erroring on every request. Native dtypes are supported;
    // FP16 (and bool) are deferred.
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

    // One shared job queue drained by N worker threads (multi-consumer via a Mutex
    // around the single-consumer receiver). A worker holds the lock only to dequeue,
    // then releases it before running inference, so up to N jobs run concurrently.
    // The std-sync channel reports each worker's model-load outcome so `start` can
    // block until the whole pool is warm.
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
                    // Hold the receiver lock only to dequeue, then drop it so a
                    // sibling worker can take the next job while this one runs.
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

    // Block until every worker has loaded its model (or one failed to).
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
    // Re-validate the shape (defense in depth; the handlers validate too) so a
    // malformed shape can't reach tensor construction and crash the worker thread.
    let mut tensors = Vec::with_capacity(inputs.len());
    for (i, inp) in inputs.iter().enumerate() {
        crate::protocol::validate_shape(i, &inp.shape, inp.data.len())?;
        tensors.push(build_tensor(&inp.data, &inp.shape)?);
    }

    let outs = model.run(&tensors).map_err(|e| e.to_string())?;

    let mut result = Vec::with_capacity(outs.len());
    for (i, t) in outs.iter().enumerate() {
        // The output dtype comes from the tensor itself (authoritative), not the
        // metadata, so a model that returns e.g. int64 indices reports INT64.
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

// Build a CPU TVM tensor from typed data, dispatching on the variant so each
// element type maps to its DLDataType via `from_slice`.
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

// Read a TVM output tensor into typed data plus its v2 datatype string,
// dispatching on the tensor's actual DLDataType (code, bits). data_as_slice::<T>
// is safe here because T is chosen to match that dtype.
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
