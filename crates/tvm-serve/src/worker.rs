//! Inference worker on a dedicated OS thread.
//!
//! `tvm-ffi` types (Module/Function/Tensor) are not Send/Sync and the VM is
//! not thread-safe. The model is loaded on its own thread and never crosses
//! threads; axum sends jobs over a channel carrying only Send data
//! (Vec<f32> + shape).
use tokio::sync::{mpsc, oneshot};
use tvm_ffi::Tensor;
use tvm_relax::{Metadata, RelaxModel, TensorSpec};

pub struct InferInput {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

pub struct InferOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    pub data: Vec<f32>,
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
        // Name-based matching when possible: the v2 spec identifies tensors by
        // name, so if every request input is named and the names are exactly the
        // model's input names, reorder to metadata order. Otherwise fall back to
        // positional matching (inputs assumed in metadata.inputs order).
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

/// Starts the worker; blocks until the model load completes so `ready` is real.
pub fn start(model_dir: &str, model_name: String) -> anyhow::Result<Handle> {
    let meta = Metadata::from_file(&format!("{model_dir}/metadata.json"))?;
    // FP32-only serving: fail at startup with a clear message instead of erroring
    // (or worse, reinterpreting bytes) on every request for non-float32 models.
    for t in meta.inputs.iter().chain(meta.outputs.iter()) {
        if !t.dtype.is_empty() && t.dtype != "float32" {
            anyhow::bail!(
                "model tensor '{}' declares dtype '{}': this serve image is FP32-only",
                t.name,
                t.dtype
            );
        }
    }
    let so_path = format!("{model_dir}/model.so");
    let entry = meta.entry.clone();
    let outputs_meta = meta.outputs.clone();

    // Job queue into the worker (async mpsc, drained with blocking_recv on the
    // thread). The second, std-sync channel is a one-shot used only to report
    // the outcome of the initial model load back to this (synchronous) function
    // so `start` can block until the model is actually loaded — see below.
    let (tx, mut rx) = mpsc::channel::<Job>(64);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    std::thread::Builder::new()
        .name("tvm-infer".into())
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
            while let Some((inputs, reply)) = rx.blocking_recv() {
                let _ = reply.send(run_one(&model, &inputs, &outputs_meta));
            }
        })?;

    ready_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("worker thread did not start"))?
        .map_err(|e| anyhow::anyhow!("model load failed: {e}"))?;

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
    // input tensors are built as f32 (FP32) only for now. Re-validate the shape
    // here (defense in depth: the handlers validate too) so a malformed shape
    // can never reach Tensor::from_slice and corrupt/abort the worker thread.
    let mut tensors = Vec::with_capacity(inputs.len());
    for (i, inp) in inputs.iter().enumerate() {
        crate::protocol::validate_shape(i, &inp.shape, inp.data.len())?;
        tensors.push(
            Tensor::from_slice(&inp.data, &inp.shape)
                .map_err(|e| format!("Tensor::from_slice: {e:?}"))?,
        );
    }

    let outs = model.run(&tensors).map_err(|e| e.to_string())?;

    let mut result = Vec::with_capacity(outs.len());
    for (i, t) in outs.iter().enumerate() {
        let data = t
            .data_as_slice::<f32>()
            .map_err(|e| format!("output[{i}] data_as_slice: {e:?}"))?
            .to_vec();
        let (name, dtype) = match outputs_meta.get(i) {
            Some(s) => (s.name.clone(), s.dtype.clone()),
            None => (format!("output_{i}"), "float32".to_string()),
        };
        result.push(InferOutput {
            name,
            shape: t.shape().to_vec(),
            datatype: crate::protocol::tvm_to_v2_dtype(&dtype).to_string(),
            data,
        });
    }
    Ok(result)
}
