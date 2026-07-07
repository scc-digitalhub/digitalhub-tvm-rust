//! Open Inference v2 (KServe) gRPC service. Shares model and worker with the
//! REST side via `Arc<Handle>`; default port 9000.
use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::protocol::TensorData;
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

    // gRPC counterpart of the REST `do_infer`: decode the tensors (raw bytes or
    // typed contents, per datatype), run them through the shared validate+infer
    // path, and encode each output into the matching typed contents field
    // (raw_output_contents left empty).
    async fn model_infer(
        &self,
        r: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let req = r.into_inner();

        // gRPC clients send tensor payloads one of two ways, and we accept both:
        //  - raw_input_contents[i]: opaque little-endian bytes of the tensor's
        //    dtype (what most KServe clients use), or
        //  - inputs[i].contents: the typed repeated field for that dtype.
        // If raw_input_contents is present at all it takes precedence for every
        // input, matching the KServe spec. Model-name/count/shape checks live in
        // Handle::serve_infer, so only wire decoding happens here.
        let use_raw = !req.raw_input_contents.is_empty();
        let mut inputs = Vec::with_capacity(req.inputs.len());
        for (i, t) in req.inputs.iter().enumerate() {
            let raw = if use_raw {
                Some(
                    req.raw_input_contents
                        .get(i)
                        .ok_or_else(|| Status::invalid_argument("raw_input_contents incomplete"))?
                        .as_slice(),
                )
            } else {
                None
            };
            let data = tensor_data_from_grpc(&t.datatype, t.contents.as_ref(), raw)
                .map_err(|e| Status::invalid_argument(format!("input[{i}]: {e}")))?;
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
                contents: Some(tensor_data_to_contents(o.data)),
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

// Decode one gRPC input tensor into typed data. Raw little-endian bytes (if
// present) take precedence; otherwise the dtype's typed `contents` field is used.
// gRPC packs INT8/16/32 in int_contents (i32) and UINT8/16/32 in uint_contents
// (u32), so those are narrowed to the declared width. FP16 is rejected (deferred).
fn tensor_data_from_grpc(
    datatype: &str,
    contents: Option<&InferTensorContents>,
    raw: Option<&[u8]>,
) -> Result<TensorData, String> {
    if let Some(raw) = raw {
        return tensor_data_from_raw(datatype, raw);
    }
    let c = contents.ok_or_else(|| "no contents and no raw_input_contents".to_string())?;
    let td = match datatype {
        "FP32" => TensorData::F32(c.fp32_contents.clone()),
        "FP64" => TensorData::F64(c.fp64_contents.clone()),
        "INT8" => TensorData::I8(c.int_contents.iter().map(|&x| x as i8).collect()),
        "INT16" => TensorData::I16(c.int_contents.iter().map(|&x| x as i16).collect()),
        "INT32" => TensorData::I32(c.int_contents.clone()),
        "INT64" => TensorData::I64(c.int64_contents.clone()),
        "UINT8" => TensorData::U8(c.uint_contents.iter().map(|&x| x as u8).collect()),
        "UINT16" => TensorData::U16(c.uint_contents.iter().map(|&x| x as u16).collect()),
        "UINT32" => TensorData::U32(c.uint_contents.clone()),
        "UINT64" => TensorData::U64(c.uint64_contents.clone()),
        "FP16" => return Err("datatype 'FP16' is not supported (deferred)".to_string()),
        other => return Err(format!("unsupported datatype '{other}'")),
    };
    Ok(td)
}

// Decode raw little-endian bytes into typed data for the given v2 datatype.
fn tensor_data_from_raw(datatype: &str, raw: &[u8]) -> Result<TensorData, String> {
    macro_rules! from_le {
        ($size:expr, $variant:path, $conv:expr) => {{
            if raw.len() % $size != 0 {
                return Err(format!(
                    "raw bytes length {} is not a multiple of {} for {datatype}",
                    raw.len(),
                    $size
                ));
            }
            Ok($variant(raw.chunks_exact($size).map($conv).collect()))
        }};
    }
    match datatype {
        "FP32" => from_le!(4, TensorData::F32, |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        "FP64" => {
            from_le!(8, TensorData::F64, |c| f64::from_le_bytes([
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            ]))
        }
        "INT8" => Ok(TensorData::I8(raw.iter().map(|&b| b as i8).collect())),
        "INT16" => from_le!(2, TensorData::I16, |c| i16::from_le_bytes([c[0], c[1]])),
        "INT32" => from_le!(4, TensorData::I32, |c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        "INT64" => {
            from_le!(8, TensorData::I64, |c| i64::from_le_bytes([
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            ]))
        }
        "UINT8" => Ok(TensorData::U8(raw.to_vec())),
        "UINT16" => from_le!(2, TensorData::U16, |c| u16::from_le_bytes([c[0], c[1]])),
        "UINT32" => from_le!(4, TensorData::U32, |c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        "UINT64" => {
            from_le!(8, TensorData::U64, |c| u64::from_le_bytes([
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            ]))
        }
        "FP16" => Err("datatype 'FP16' is not supported (deferred)".to_string()),
        other => Err(format!("unsupported datatype '{other}'")),
    }
}

// Encode typed output data into the gRPC contents field matching its dtype
// (INT8/16/32 packed into int_contents, UINT8/16/32 into uint_contents).
fn tensor_data_to_contents(data: TensorData) -> InferTensorContents {
    let mut c = InferTensorContents::default();
    match data {
        TensorData::F32(v) => c.fp32_contents = v,
        TensorData::F64(v) => c.fp64_contents = v,
        TensorData::I8(v) => c.int_contents = v.into_iter().map(|x| x as i32).collect(),
        TensorData::I16(v) => c.int_contents = v.into_iter().map(|x| x as i32).collect(),
        TensorData::I32(v) => c.int_contents = v,
        TensorData::I64(v) => c.int64_contents = v,
        TensorData::U8(v) => c.uint_contents = v.into_iter().map(|x| x as u32).collect(),
        TensorData::U16(v) => c.uint_contents = v.into_iter().map(|x| x as u32).collect(),
        TensorData::U32(v) => c.uint_contents = v,
        TensorData::U64(v) => c.uint64_contents = v,
    }
    c
}
