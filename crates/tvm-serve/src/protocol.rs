//! Open Inference v2 (KServe) protocol types, REST/JSON subset. Tensor `data` is
//! carried as [`TensorData`], a typed buffer covering the native dtypes this
//! serve image supports (FP32/FP64, INT8/16/32/64, UINT8/16/32/64). FP16 and
//! BOOL are intentionally unsupported for now (FP16 needs an unsafe half path in
//! rust; add it as a follow-up).
use serde::{Deserialize, Serialize};

// Some v2 fields are accepted for protocol compatibility but not consumed:
// (outputs) all outputs are returned; (parameters) none are honored.
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
    /// v2 `data`: a flat or arbitrarily nested array of numbers. Parsed into a
    /// typed [`TensorData`] (per `datatype`) by [`TensorData::from_json`].
    pub data: serde_json::Value,
}

/// Typed tensor payload for the native dtypes this serve image supports. FP16
/// and BOOL are intentionally absent (FP16 deferred; BOOL is not a model I/O
/// here). Kept transport-agnostic (no tvm-ffi types) so both the REST and gRPC
/// handlers build and consume it; the worker maps it to/from a TVM `Tensor`.
#[derive(Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl TensorData {
    /// Element count.
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::F64(v) => v.len(),
            TensorData::I8(v) => v.len(),
            TensorData::I16(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I64(v) => v.len(),
            TensorData::U8(v) => v.len(),
            TensorData::U16(v) => v.len(),
            TensorData::U32(v) => v.len(),
            TensorData::U64(v) => v.len(),
        }
    }

    /// Parse a v2 request input's `data` (flat or nested arrays of numbers) into a
    /// typed buffer matching the v2 `datatype`. Integer types read JSON integers
    /// directly (so INT64 keeps full precision) with a float fallback, so `1.0` is
    /// accepted in an integer slot. FP16 is rejected (deferred).
    pub fn from_json(v: &serde_json::Value, datatype: &str) -> Result<TensorData, String> {
        let mut nums: Vec<&serde_json::Number> = Vec::new();
        collect_numbers(v, &mut nums)?;
        let td = match datatype {
            "FP32" => TensorData::F32(map_nums(&nums, |n| n.as_f64().map(|x| x as f32))?),
            "FP64" => TensorData::F64(map_nums(&nums, |n| n.as_f64())?),
            "INT8" => TensorData::I8(map_nums(&nums, |n| as_i64(n).map(|x| x as i8))?),
            "INT16" => TensorData::I16(map_nums(&nums, |n| as_i64(n).map(|x| x as i16))?),
            "INT32" => TensorData::I32(map_nums(&nums, |n| as_i64(n).map(|x| x as i32))?),
            "INT64" => TensorData::I64(map_nums(&nums, as_i64)?),
            "UINT8" => TensorData::U8(map_nums(&nums, |n| as_u64(n).map(|x| x as u8))?),
            "UINT16" => TensorData::U16(map_nums(&nums, |n| as_u64(n).map(|x| x as u16))?),
            "UINT32" => TensorData::U32(map_nums(&nums, |n| as_u64(n).map(|x| x as u32))?),
            "UINT64" => TensorData::U64(map_nums(&nums, as_u64)?),
            "FP16" => {
                return Err(
                    "datatype 'FP16' is not supported by this serve image (deferred)".to_string(),
                )
            }
            other => return Err(format!("unsupported datatype '{other}'")),
        };
        Ok(td)
    }

    /// Convert to a v2 response `data` array (flat, row-major numbers).
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            TensorData::F32(v) => serde_json::json!(v),
            TensorData::F64(v) => serde_json::json!(v),
            TensorData::I8(v) => serde_json::json!(v),
            TensorData::I16(v) => serde_json::json!(v),
            TensorData::I32(v) => serde_json::json!(v),
            TensorData::I64(v) => serde_json::json!(v),
            TensorData::U8(v) => serde_json::json!(v),
            TensorData::U16(v) => serde_json::json!(v),
            TensorData::U32(v) => serde_json::json!(v),
            TensorData::U64(v) => serde_json::json!(v),
        }
    }
}

// JSON integers larger than i64/u64 arrive as floats; the float fallback keeps
// `1.0` acceptable in an integer slot without silently mangling big integers.
fn as_i64(n: &serde_json::Number) -> Option<i64> {
    n.as_i64().or_else(|| n.as_f64().map(|f| f as i64))
}

fn as_u64(n: &serde_json::Number) -> Option<u64> {
    n.as_u64().or_else(|| n.as_f64().map(|f| f as u64))
}

// Recursively collect the leaf numbers of a (possibly nested) v2 data array.
fn collect_numbers<'a>(
    v: &'a serde_json::Value,
    out: &mut Vec<&'a serde_json::Number>,
) -> Result<(), String> {
    match v {
        serde_json::Value::Array(a) => {
            for e in a {
                collect_numbers(e, out)?;
            }
            Ok(())
        }
        serde_json::Value::Number(n) => {
            out.push(n);
            Ok(())
        }
        other => Err(format!("unsupported element in data: {other}")),
    }
}

fn map_nums<T>(
    nums: &[&serde_json::Number],
    f: impl Fn(&serde_json::Number) -> Option<T>,
) -> Result<Vec<T>, String> {
    let mut out = Vec::with_capacity(nums.len());
    for n in nums {
        out.push(f(n).ok_or_else(|| format!("number {n} not convertible to the target dtype"))?);
    }
    Ok(out)
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
    /// Flat row-major numbers, typed to match `datatype` (built from a
    /// [`TensorData`] via [`TensorData::to_json`]).
    pub data: serde_json::Value,
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

/// Validates one input tensor's shape against its data length: non-negative dims
/// and `product(shape) == data_len`, with an overflow-checked product. Guards
/// tensor construction against huge/overflowing allocations driven by a
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

/// The native v2 datatypes this serve image supports (FP16 and BOOL excluded).
pub fn is_supported_v2_dtype(dt: &str) -> bool {
    matches!(
        dt,
        "FP32"
            | "FP64"
            | "INT8"
            | "INT16"
            | "INT32"
            | "INT64"
            | "UINT8"
            | "UINT16"
            | "UINT32"
            | "UINT64"
    )
}

/// Validates one input tensor: a supported native datatype plus `validate_shape`.
/// Returns a human-readable error for a 400 / invalid_argument response.
pub fn validate_input(
    idx: usize,
    datatype: &str,
    shape: &[i64],
    data_len: usize,
) -> Result<(), String> {
    if !is_supported_v2_dtype(datatype) {
        return Err(format!(
            "input[{idx}]: datatype '{datatype}' unsupported (FP16/BOOL are not served)"
        ));
    }
    validate_shape(idx, shape, data_len)
}

/// TVM dtype (e.g. `"float32"`) → Open Inference v2 datatype (e.g. `"FP32"`).
pub fn tvm_to_v2_dtype(dt: &str) -> &'static str {
    match dt {
        "float32" => "FP32",
        "float64" => "FP64",
        "int64" => "INT64",
        "int32" => "INT32",
        "int16" => "INT16",
        "int8" => "INT8",
        "uint8" => "UINT8",
        "uint16" => "UINT16",
        "uint32" => "UINT32",
        "uint64" => "UINT64",
        _ => "FP32",
    }
}
