//! Open Inference v2 (KServe) protocol types, REST/JSON subset. `data` is
//! handled as flat f32 (FP32) only for now.
use serde::{Deserialize, Serialize};

// Some v2 fields are accepted for protocol compatibility but not consumed:
// positional matching (name ignored), FP32 assumed (datatype), all outputs
// returned (outputs), no parameters honored.
#[derive(Debug, Deserialize)]
pub struct InferRequest {
    #[serde(default)]
    pub id: String,
    pub inputs: Vec<RequestInput>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct RequestInput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    /// FP32 data, flat row-major or nested arrays (both allowed by the v2 spec);
    /// flattened with `flatten_f32` before reaching the worker.
    pub data: serde_json::Value,
}

/// Flattens a v2 `data` value (a flat or arbitrarily nested array of numbers)
/// into row-major f32s, matching the go backend's behavior.
pub fn flatten_f32(v: &serde_json::Value, out: &mut Vec<f32>) -> Result<(), String> {
    match v {
        serde_json::Value::Array(a) => {
            for e in a {
                flatten_f32(e, out)?;
            }
            Ok(())
        }
        serde_json::Value::Number(n) => {
            let f = n
                .as_f64()
                .ok_or_else(|| format!("non-finite number in data: {n}"))?;
            out.push(f as f32);
            Ok(())
        }
        other => Err(format!("unsupported element in data: {other}")),
    }
}

#[derive(Debug, Serialize)]
pub struct InferResponse {
    pub model_name: String,
    pub model_version: String,
    pub id: String,
    pub outputs: Vec<ResponseOutput>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct ResponseOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    pub data: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct TensorMetadata {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Serialize)]
pub struct ModelMetadata {
    pub name: String,
    pub versions: Vec<String>,
    pub platform: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Serialize)]
pub struct ServerMetadata {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

/// Validates one input tensor's shape against its data length: non-negative
/// dims and `product(shape) == data_len`, with overflow-checked product. Guards
/// `Tensor::from_slice` against huge/overflowing allocations driven by a
/// malformed client `shape` (which would otherwise segfault the worker thread).
pub fn validate_shape(idx: usize, shape: &[i64], data_len: usize) -> Result<(), String> {
    let mut numel: i64 = 1;
    for &d in shape {
        if d < 0 {
            return Err(format!("input[{idx}]: negative dim {d} in shape {shape:?}"));
        }
        numel = numel
            .checked_mul(d)
            .ok_or_else(|| format!("input[{idx}]: shape {shape:?} overflows"))?;
    }
    if numel as usize != data_len {
        return Err(format!(
            "input[{idx}]: shape {shape:?} implies {numel} elements but data has {data_len}"
        ));
    }
    Ok(())
}

/// Validates one input tensor: FP32-only (the worker reinterprets all bytes as
/// f32) plus `validate_shape`. Returns a human-readable error for a 400 /
/// invalid_argument response.
pub fn validate_input(idx: usize, datatype: &str, shape: &[i64], data_len: usize) -> Result<(), String> {
    if datatype != "FP32" {
        return Err(format!(
            "input[{idx}]: datatype '{datatype}' unsupported (only FP32 is served)"
        ));
    }
    validate_shape(idx, shape, data_len)
}

/// TVM dtype (e.g. `"float32"`) → Open Inference v2 datatype (e.g. `"FP32"`).
pub fn tvm_to_v2_dtype(dt: &str) -> &'static str {
    match dt {
        "float32" => "FP32",
        "float64" => "FP64",
        "float16" => "FP16",
        "int64" => "INT64",
        "int32" => "INT32",
        "int16" => "INT16",
        "int8" => "INT8",
        "uint8" => "UINT8",
        "uint16" => "UINT16",
        "uint32" => "UINT32",
        "uint64" => "UINT64",
        "bool" => "BOOL",
        _ => "FP32",
    }
}
