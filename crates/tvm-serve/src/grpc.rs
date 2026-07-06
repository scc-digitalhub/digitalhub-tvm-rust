//! Open Inference v2 (KServe) gRPC service. Shares model and worker with the
//! REST side via `Arc<Handle>`; default port 9000.
use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::worker::{Handle, InferInput};

/// Generated from `proto/grpc_predict_v2.proto` (package `inference`).
#[allow(clippy::all, dead_code)]
pub mod pb {
    tonic::include_proto!("inference");
}

use pb::grpc_inference_service_server::GrpcInferenceService;
use pb::{
    model_infer_response::InferOutputTensor, model_metadata_response::TensorMetadata,
    InferTensorContents, ModelInferRequest, ModelInferResponse, ModelMetadataRequest,
    ModelMetadataResponse, ModelReadyRequest, ModelReadyResponse, ServerLiveRequest,
    ServerLiveResponse, ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest,
    ServerReadyResponse,
};

pub use pb::grpc_inference_service_server::GrpcInferenceServiceServer;

pub struct InferenceService {
    pub handle: Arc<Handle>,
}

#[tonic::async_trait]
impl GrpcInferenceService for InferenceService {
    async fn server_live(
        &self,
        _r: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _r: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse { ready: true }))
    }

    async fn model_ready(
        &self,
        r: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let ready = r.into_inner().name == self.handle.model_name;
        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn server_metadata(
        &self,
        _r: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        Ok(Response::new(ServerMetadataResponse {
            name: "tvm-serve".to_string(),
            version: "2".to_string(),
            extensions: vec![],
        }))
    }

    async fn model_metadata(
        &self,
        r: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let name = r.into_inner().name;
        if name != self.handle.model_name {
            return Err(Status::not_found(format!("model '{name}' not found")));
        }
        let m = &self.handle.metadata;
        let mk = |s: &tvm_relax::TensorSpec| TensorMetadata {
            name: s.name.clone(),
            datatype: crate::protocol::tvm_to_v2_dtype(&s.dtype).to_string(),
            shape: s.shape.clone(),
        };
        Ok(Response::new(ModelMetadataResponse {
            name: self.handle.model_name.clone(),
            versions: vec!["1".to_string()],
            platform: "tvm_relax".to_string(),
            inputs: m.inputs.iter().map(mk).collect(),
            outputs: m.outputs.iter().map(mk).collect(),
        }))
    }

    // gRPC counterpart of the REST `do_infer`: decode the tensors (protobuf raw
    // bytes or typed fp32), then run them through the shared validate+infer path.
    // Only the wire encoding differs; outputs are always returned as typed
    // fp32_contents (raw_output_contents left empty).
    async fn model_infer(
        &self,
        r: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let req = r.into_inner();

        // gRPC clients send tensor payloads one of two ways, and we accept both:
        //  - raw_input_contents[i]: opaque little-endian f32 bytes (what most KServe
        //    clients use; avoids protobuf repeated-field overhead), or
        //  - inputs[i].contents.fp32_contents: the typed repeated float field.
        // If raw_input_contents is present at all it takes precedence for every
        // input, matching the KServe spec. Model-name/count/shape checks live in
        // Handle::serve_infer, so only wire decoding happens here.
        let use_raw = !req.raw_input_contents.is_empty();
        let mut inputs = Vec::with_capacity(req.inputs.len());
        for (i, t) in req.inputs.iter().enumerate() {
            let data: Vec<f32> = if use_raw {
                let raw = req
                    .raw_input_contents
                    .get(i)
                    .ok_or_else(|| Status::invalid_argument("raw_input_contents incomplete"))?;
                if raw.len() % 4 != 0 {
                    return Err(Status::invalid_argument(format!(
                        "input[{i}]: raw bytes length {} is not a multiple of 4 (FP32)",
                        raw.len()
                    )));
                }
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                t.contents
                    .as_ref()
                    .map(|c| c.fp32_contents.clone())
                    .ok_or_else(|| Status::invalid_argument("input has no FP32 contents"))?
            };
            inputs.push(InferInput {
                name: t.name.clone(),
                datatype: t.datatype.clone(),
                shape: t.shape.clone(),
                data,
            });
        }

        let outs = self
            .handle
            .serve_infer(&req.model_name, inputs)
            .await
            .map_err(serve_err_status)?;
        let outputs = outs
            .into_iter()
            .map(|o| InferOutputTensor {
                name: o.name,
                datatype: o.datatype,
                shape: o.shape,
                parameters: Default::default(),
                contents: Some(InferTensorContents {
                    fp32_contents: o.data,
                    ..Default::default()
                }),
            })
            .collect();

        Ok(Response::new(ModelInferResponse {
            model_name: self.handle.model_name.clone(),
            model_version: "1".to_string(),
            id: req.id,
            parameters: Default::default(),
            outputs,
            raw_output_contents: vec![],
        }))
    }
}

// Map the shared worker error onto a gRPC Status.
fn serve_err_status(e: crate::worker::ServeErr) -> Status {
    use crate::worker::ServeErr;
    match e {
        ServeErr::NotFound(m) => Status::not_found(m),
        ServeErr::BadRequest(m) => Status::invalid_argument(m),
        ServeErr::Internal(m) => Status::internal(m),
    }
}
