/*
 * SPDX-FileCopyrightText: © 2025 DSLab - Fondazione Bruno Kessler
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! Open Inference v2 (KServe) protocol types, REST/JSON subset. Tensor `data` is
//! carried as [`TensorData`] over the native dtypes this image supports; FP16/BOOL
//! are unsupported.
use serde::{Deserialize, Serialize};

// v2 `outputs`/`parameters` request fields are accepted but ignored (serde drops
// unknown fields); the server always returns every model output.
#[derive(Debug, Deserialize)]
pub struct InferRequest {
    #[serde(default)]
    pub id: String,
    pub inputs: Vec<RequestInput>,
}

#[derive(Debug, Deserialize)]
pub struct RequestInput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    /// v2 `data`: a flat or arbitrarily nested array of numbers. Parsed into a
    /// typed [`TensorData`] (per `datatype`) by [`TensorData::from_json`].
    pub data: serde_json::Value,
}

/// Typed tensor payload for the supported native dtypes. Transport-agnostic (no
/// tvm-ffi types) so REST and gRPC share it; the worker maps it to/from a TVM `Tensor`.
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
    pub fn datatype(&self) -> &'static str {
        match self {
            Self::F32(_) => "FP32",
            Self::F64(_) => "FP64",
            Self::I8(_) => "INT8",
            Self::I16(_) => "INT16",
            Self::I32(_) => "INT32",
            Self::I64(_) => "INT64",
            Self::U8(_) => "UINT8",
            Self::U16(_) => "UINT16",
            Self::U32(_) => "UINT32",
            Self::U64(_) => "UINT64",
        }
    }

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

    /// Parse a v2 input's `data` (flat or nested numbers) into a typed buffer per
    /// `datatype`. Integers keep full precision, including integral decimal and
    /// exponent notation such as `1.0`; FP16 is rejected.
    pub fn from_json(v: &serde_json::Value, datatype: &str) -> Result<TensorData, String> {
        let mut nums: Vec<&serde_json::Number> = Vec::new();
        collect_numbers(v, &mut nums)?;
        let td = match datatype {
            "FP32" => TensorData::F32(map_nums(&nums, |n| {
                n.as_f64().map(|x| x as f32).filter(|x| x.is_finite())
            })?),
            "FP64" => TensorData::F64(map_nums(&nums, |n| n.as_f64())?),
            "INT8" => TensorData::I8(map_nums(&nums, |n| {
                as_i64(n).and_then(|x| i8::try_from(x).ok())
            })?),
            "INT16" => TensorData::I16(map_nums(&nums, |n| {
                as_i64(n).and_then(|x| i16::try_from(x).ok())
            })?),
            "INT32" => TensorData::I32(map_nums(&nums, |n| {
                as_i64(n).and_then(|x| i32::try_from(x).ok())
            })?),
            "INT64" => TensorData::I64(map_nums(&nums, as_i64)?),
            "UINT8" => TensorData::U8(map_nums(&nums, |n| {
                as_u64(n).and_then(|x| u8::try_from(x).ok())
            })?),
            "UINT16" => TensorData::U16(map_nums(&nums, |n| {
                as_u64(n).and_then(|x| u16::try_from(x).ok())
            })?),
            "UINT32" => TensorData::U32(map_nums(&nums, |n| {
                as_u64(n).and_then(|x| u32::try_from(x).ok())
            })?),
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

// Accept integral decimal/exponent notation without rounding through f64.
fn as_i64(n: &serde_json::Number) -> Option<i64> {
    n.as_i64().or_else(|| i64::try_from(exact_integer(n)?).ok())
}

fn as_u64(n: &serde_json::Number) -> Option<u64> {
    n.as_u64().or_else(|| u64::try_from(exact_integer(n)?).ok())
}

fn exact_integer(n: &serde_json::Number) -> Option<i128> {
    let text = n.as_str();
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((text, "0"));
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.strip_prefix('-').unwrap_or(mantissa);
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Some(0);
    }
    let significant = digits.trim_end_matches('0');
    let shift = exponent
        .parse::<i64>()
        .ok()?
        .checked_sub(i64::try_from(fraction.len()).ok()?)?
        .checked_add(i64::try_from(digits.len() - significant.len()).ok()?)?;
    // A negative shift leaves a fractional part. More than 20 digits cannot fit
    // any supported integer dtype; bound the exponent before calculating it.
    if shift < 0 || shift > 20 || significant.len() > 20 - shift as usize {
        return None;
    }
    let value = significant.parse::<i128>().ok()? * 10_i128.pow(shift as u32);
    Some(if negative { -value } else { value })
}

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
    /// Quantization params for int8/uint8 models, in the v2 `parameters` map so a
    /// client can quantize its input and dequantize the output. Omitted for float
    /// models, which keeps the response byte-identical to before for those.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
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

/// Validates shape vs data length: non-negative dims and overflow-checked
/// `product(shape) == data_len`. Guards tensor construction from a malformed
/// client shape that would otherwise overflow/segfault the worker.
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
    let numel = usize::try_from(numel)
        .map_err(|_| format!("input[{idx}]: shape {shape:?} exceeds the address space"))?;
    if numel != data_len {
        return Err(format!(
            "input[{idx}]: shape {shape:?} implies {numel} elements but data has {data_len}"
        ));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_numeric_overflow_and_fractional_integers() {
        for (dtype, value) in [
            ("INT8", "128"),
            ("INT8", "-129"),
            ("UINT8", "256"),
            ("UINT8", "-1"),
            ("INT16", "32768"),
            ("UINT16", "65536"),
            ("INT32", "2147483648"),
            ("UINT32", "4294967296"),
            ("INT64", "9223372036854775808"),
            ("INT64", "-9223372036854775809"),
            ("INT64", "-9223372036854775809.0"),
            ("UINT64", "18446744073709551616"),
            ("INT64", "9007199254740991.5"),
            ("UINT64", "1.00000000000000001"),
            ("INT8", "1e-400"),
            ("INT8", "1.5"),
            ("UINT8", "1.5"),
            ("FP32", "1e100"),
            ("FP64", "1e400"),
        ] {
            let json = serde_json::from_str(&format!("[{value}]")).unwrap();
            assert!(
                TensorData::from_json(&json, dtype).is_err(),
                "{dtype} {value}"
            );
        }
    }

    #[test]
    fn preserves_integer_boundaries_and_accepts_integral_float_notation() {
        for (dtype, value) in [
            ("INT8", "-128"),
            ("INT8", "127"),
            ("UINT8", "255"),
            ("INT64", "-9223372036854775808"),
            ("INT64", "9223372036854775807"),
            ("UINT64", "18446744073709551615"),
            ("INT64", "9223372036854775807.0"),
            ("UINT64", "18446744073709551615.0"),
            ("INT8", "1.0"),
            ("UINT8", "1e1"),
            ("INT8", "-1.20e2"),
            ("UINT8", "1000e-1"),
            ("UINT8", "0e9999999999999999999999999999"),
        ] {
            let json: serde_json::Value = serde_json::from_str(&format!("[{value}]")).unwrap();
            assert!(
                TensorData::from_json(&json, dtype).is_ok(),
                "{dtype} {value}"
            );
        }
    }

    #[test]
    fn integer_decimal_notation_preserves_all_digits() {
        for (dtype, input, expected) in [
            ("INT64", "-9223372036854775808.0", "-9223372036854775808"),
            ("INT64", "9223372036854775807.0", "9223372036854775807"),
            ("UINT64", "1.8446744073709551615e19", "18446744073709551615"),
        ] {
            let json = serde_json::from_str(&format!("[{input}]")).unwrap();
            let actual = TensorData::from_json(&json, dtype).unwrap().to_json();
            assert_eq!(actual.to_string(), format!("[{expected}]"));
        }
    }
}
