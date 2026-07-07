//! Open Inference v2 (KServe) server for TVM Relax models, REST + gRPC.
//!
//! Config via env:
//! ```text
//! TVM_MODEL_DIR        model.so + metadata.json (default /shared/model)
//! TVM_MODEL_NAME       model name in /v2/models/<name> (default tvm-model)
//! TVM_SERVE_PORT       REST port (default 8080)
//! TVM_SERVE_GRPC_PORT  gRPC port (default 9000)
//! TVM_SERVE_WORKERS    inference workers / model copies (default 1)
//! ```
mod grpc;
mod protocol;
mod worker;

use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

use protocol::{
    tvm_to_v2_dtype, InferRequest, InferResponse, ModelMetadata, ResponseOutput, ServerMetadata,
    TensorMetadata,
};
use worker::{Handle, InferInput, ServeErr};

// Per-request axum state, cloned on every request. The `Arc<Handle>` keeps the
// worker pool shared, not duplicated — the handle just enqueues jobs.
#[derive(Clone)]
struct AppState {
    handle: Arc<Handle>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_dir = std::env::var("TVM_MODEL_DIR").unwrap_or_else(|_| "/shared/model".to_string());
    let model_name = std::env::var("TVM_MODEL_NAME").unwrap_or_else(|_| "tvm-model".to_string());
    let port = env_u16("TVM_SERVE_PORT", 8080);
    let grpc_port = env_u16("TVM_SERVE_GRPC_PORT", 9000);
    let workers = env_usize("TVM_SERVE_WORKERS", 1);

    eprintln!("[tvm-serve] loading model from {model_dir} (name='{model_name}', workers={workers})...");
    let handle = worker::start(&model_dir, model_name, workers)?;
    eprintln!(
        "[tvm-serve] model ready: entry='{}' inputs={:?} outputs={:?}",
        handle.metadata.entry,
        handle
            .metadata
            .inputs
            .iter()
            .map(|i| &i.name)
            .collect::<Vec<_>>(),
        handle
            .metadata
            .outputs
            .iter()
            .map(|o| &o.name)
            .collect::<Vec<_>>(),
    );

    let state = AppState {
        handle: Arc::new(handle),
    };
    let grpc_handle = state.handle.clone();

    let app = Router::new()
        .route("/", get(|| async { "tvm-serve · Open Inference v2" }))
        .route("/v2/health/live", get(|| async { StatusCode::OK }))
        .route("/v2/health/ready", get(|| async { StatusCode::OK }))
        .route("/v2", get(server_metadata))
        .route("/v2/models/:name", get(model_metadata))
        .route("/v2/models/:name/ready", get(model_ready))
        .route("/v2/models/:name/infer", post(infer))
        .route(
            "/v2/models/:name/versions/:version/infer",
            post(infer_versioned),
        )
        // axum default body limit is 2MB; yolov8n FP32 input is ~12MB. Cap at 1 GiB.
        .layer(DefaultBodyLimit::max(1 << 30))
        .with_state(state);

    let rest_addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&rest_addr).await?;

    let grpc_addr: std::net::SocketAddr = format!("0.0.0.0:{grpc_port}").parse()?;
    let grpc_service = grpc::GrpcInferenceServiceServer::new(grpc::InferenceService {
        handle: grpc_handle,
    })
    // tonic default is 4MB; raise to 512MB for v2 tensors.
    .max_decoding_message_size(512 * 1024 * 1024)
    .max_encoding_message_size(512 * 1024 * 1024);
    let grpc_server = tonic::transport::Server::builder()
        .add_service(grpc_service)
        .serve(grpc_addr);

    eprintln!("[tvm-serve] REST on http://{rest_addr}  ·  gRPC on {grpc_addr}");

    // Run both servers concurrently; the first branch to resolve tears the whole
    // process down — a fatal error from either server, or Ctrl-C.
    tokio::select! {
        res = axum::serve(listener, app) => res?,
        res = grpc_server => res?,
        _ = shutdown_signal() => eprintln!("[tvm-serve] shutdown."),
    }
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

// Read a u16 from an env var, falling back to `default` if it is unset or unparsable.
fn env_u16(key: &str, default: u16) -> u16 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// Read a positive usize from an env var, falling back to `default` if it is
// unset, unparsable, or zero (a pool must have at least one worker).
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(default)
}

async fn server_metadata() -> Json<ServerMetadata> {
    Json(ServerMetadata {
        name: "tvm-serve".to_string(),
        version: "2".to_string(),
        extensions: vec![],
    })
}

// Turn a model's declared tensor spec into v2 wire metadata, translating the
// TVM dtype to its Open Inference name.
fn to_tensor_metadata(s: &tvm_relax::TensorSpec) -> TensorMetadata {
    TensorMetadata {
        name: s.name.clone(),
        datatype: tvm_to_v2_dtype(&s.dtype).to_string(),
        shape: s.shape.clone(),
    }
}

async fn model_ready(State(state): State<AppState>, Path(name): Path<String>) -> StatusCode {
    if name == state.handle.model_name {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn model_metadata(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ModelMetadata>, StatusCode> {
    if name != state.handle.model_name {
        return Err(StatusCode::NOT_FOUND);
    }
    let m = &state.handle.metadata;
    Ok(Json(ModelMetadata {
        name: state.handle.model_name.clone(),
        versions: vec!["1".to_string()],
        platform: "tvm_relax".to_string(),
        inputs: m.inputs.iter().map(to_tensor_metadata).collect(),
        outputs: m.outputs.iter().map(to_tensor_metadata).collect(),
    }))
}

async fn infer(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, (StatusCode, String)> {
    do_infer(&state, &name, req).await
}

async fn infer_versioned(
    State(state): State<AppState>,
    Path((name, _version)): Path<(String, String)>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, (StatusCode, String)> {
    do_infer(&state, &name, req).await
}

// Shared by the versioned and unversioned /infer routes: decode the tensors, run
// them through `Handle::serve_infer`, and shape the reply. Awaiting keeps this
// handler off the blocking, non-Send inference thread.
async fn do_infer(
    state: &AppState,
    name: &str,
    req: InferRequest,
) -> Result<Json<InferResponse>, (StatusCode, String)> {
    let mut inputs: Vec<InferInput> = Vec::with_capacity(req.inputs.len());
    for i in req.inputs {
        // Parse the v2 data into a typed buffer matching its declared datatype
        // (FP32/FP64, INT8..64, UINT8..64); rejects FP16/unknown with a 400.
        let data = protocol::TensorData::from_json(&i.data, &i.datatype)
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        inputs.push(InferInput {
            name: i.name,
            datatype: i.datatype,
            shape: i.shape,
            data,
        });
    }

    let t0 = std::time::Instant::now();
    let outs = state
        .handle
        .serve_infer(name, inputs)
        .await
        .map_err(serve_err_http)?;
    let ms = (t0.elapsed().as_secs_f64() * 1e3 * 10.0).round() / 10.0;

    let outputs = outs
        .into_iter()
        .map(|o| ResponseOutput {
            name: o.name,
            shape: o.shape,
            datatype: o.datatype,
            data: o.data.to_json(),
        })
        .collect();

    Ok(Json(InferResponse {
        model_name: state.handle.model_name.clone(),
        model_version: "1".to_string(),
        id: req.id,
        outputs,
        parameters: serde_json::json!({ "inference_time_ms": ms }),
    }))
}

// Map the shared worker error onto a REST status code + message.
fn serve_err_http(e: ServeErr) -> (StatusCode, String) {
    match e {
        ServeErr::NotFound(m) => (StatusCode::NOT_FOUND, m),
        ServeErr::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
        ServeErr::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
    }
}
