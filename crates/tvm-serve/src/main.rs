//! Open Inference v2 (KServe) server for TVM Relax models, REST + gRPC.
//!
//! Config via env:
//! ```text
//! TVM_MODEL_DIR        model.so + metadata.json (default /shared/model)
//! TVM_MODEL_NAME       model name in /v2/models/<name> (default model)
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

// Cloned per request; the `Arc<Handle>` shares one worker pool.
#[derive(Clone)]
struct AppState {
    handle: Arc<Handle>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_dir = std::env::var("TVM_MODEL_DIR").unwrap_or_else(|_| "/shared/model".to_string());
    let model_name = std::env::var("TVM_MODEL_NAME").unwrap_or_else(|_| "model".to_string());
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

    // First branch to resolve tears the process down (fatal error or Ctrl-C).
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

fn env_u16(key: &str, default: u16) -> u16 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// Clamps to >= 1: the pool needs at least one worker.
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

fn to_tensor_metadata(s: &tvm_relax::TensorSpec) -> TensorMetadata {
    // Quantized models only: surface scale/zero_point so the client can do the affine
    // mapping. Float models keep `parameters` absent (skip_serializing_if).
    let parameters = if s.scale.is_empty() {
        None
    } else {
        let mut p = serde_json::json!({ "scale": s.scale, "zero_point": s.zero_point });
        if let Some(d) = s.quantized_dimension {
            p["quantized_dimension"] = serde_json::json!(d);
        }
        Some(p)
    };
    TensorMetadata {
        name: s.name.clone(),
        datatype: tvm_to_v2_dtype(&s.dtype).to_string(),
        shape: s.shape.clone(),
        parameters,
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

// Shared by both /infer routes; awaiting keeps this off the non-Send worker thread.
async fn do_infer(
    state: &AppState,
    name: &str,
    req: InferRequest,
) -> Result<Json<InferResponse>, (StatusCode, String)> {
    let mut inputs: Vec<InferInput> = Vec::with_capacity(req.inputs.len());
    for i in req.inputs {
        let data = protocol::TensorData::from_json(&i.data, &i.datatype)
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        inputs.push(InferInput {
            name: i.name,
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

fn serve_err_http(e: ServeErr) -> (StatusCode, String) {
    match e {
        ServeErr::NotFound(m) => (StatusCode::NOT_FOUND, m),
        ServeErr::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
        ServeErr::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(dtype: &str, scale: Vec<f64>, zero_point: Vec<i64>) -> tvm_relax::TensorSpec {
        tvm_relax::TensorSpec {
            name: "images".to_string(),
            shape: vec![1, 224, 224, 3],
            dtype: dtype.to_string(),
            scale,
            zero_point,
            quantized_dimension: None,
        }
    }

    /// A quantized tensor publishes scale/zero_point under the v2 `parameters` map:
    /// it is the only way a client can turn the int8 it receives back into reals.
    #[test]
    fn quantized_tensor_publishes_parameters() {
        let m = to_tensor_metadata(&spec("int8", vec![0.003921568859368563], vec![-128]));
        assert_eq!(m.datatype, "INT8");
        let p = m.parameters.expect("un tensore quantizzato deve esporre i parametri");
        assert_eq!(p["scale"][0], 0.003921568859368563);
        assert_eq!(p["zero_point"][0], -128);
        assert!(p.get("quantized_dimension").is_none(), "per-tensore: nessun asse");
    }

    /// Per-axis quantization also carries the axis the entries are indexed by.
    #[test]
    fn per_axis_tensor_publishes_quantized_dimension() {
        let mut s = spec("int8", vec![0.1, 0.2], vec![0, 0]);
        s.quantized_dimension = Some(3);
        let p = to_tensor_metadata(&s).parameters.unwrap();
        assert_eq!(p["quantized_dimension"], 3);
    }

    /// Non-regression: a float model must serialize exactly as before. `parameters`
    /// has to disappear from the JSON, not show up as null — quantization is an
    /// independent axis and must not leak into models that have none.
    #[test]
    fn float_tensor_omits_parameters() {
        let m = to_tensor_metadata(&spec("float32", vec![], vec![]));
        assert_eq!(m.datatype, "FP32");
        assert!(m.parameters.is_none());

        let raw = serde_json::to_string(&m).unwrap();
        assert!(!raw.contains("parameters"), "JSON inatteso: {raw}");
    }
}
