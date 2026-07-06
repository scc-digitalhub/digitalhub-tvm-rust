//! GO/NO-GO proof-of-concept for the Rust TVM runtime.
//!
//! This is the equivalence check that justified the whole crate: load a Relax
//! `model.so`, run inference through Rust (`tvm-relax`), and confirm the output
//! is numerically identical to a reference produced by TVM's Python runtime on
//! the same model and input. It is a standalone example, not part of the served
//! binary.
//!
//! Expected files in <model_dir>: metadata.json, model.so, input.bin (f32 LE
//! row-major, shape = inputs[0].shape), expected.bin (reference f32 LE output).
//!
//! Usage: cargo run --example poc -- <model_dir>
use std::fs;
use std::path::Path;

use anyhow::{anyhow, ensure, Context, Result};
use tvm_ffi::Tensor;
use tvm_relax::{Metadata, RelaxModel};

/// Reads a raw little-endian f32 dump (the format the Python side writes with
/// `.tofile()`) into a flat Vec, verifying the byte count is 4-aligned.
fn read_f32_le(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() % 4 == 0,
        "{} is not a multiple of 4 bytes",
        path.display()
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let dir = Path::new(&dir);

    let meta = Metadata::from_file(&dir.join("metadata.json").to_string_lossy())?;
    let so = dir.join("model.so");
    let in_spec = meta
        .inputs
        .first()
        .ok_or_else(|| anyhow!("metadata has no inputs"))?;

    println!("== TVM Rust runtime PoC ==");
    println!("model.so : {}", so.display());
    println!("entry    : {}", meta.entry);
    println!(
        "input    : {} {:?} {}",
        in_spec.name, in_spec.shape, in_spec.dtype
    );

    let input_data = read_f32_le(&dir.join("input.bin"))?;
    let expected = read_f32_le(&dir.join("expected.bin"))?;

    // Wrap the flat input buffer as a Tensor with the shape from metadata.
    let input = Tensor::from_slice(&input_data, &in_spec.shape)
        .map_err(|e| anyhow!("Tensor::from_slice: {e:?}"))?;

    let model = RelaxModel::load(&so.to_string_lossy(), &meta.entry)?;

    // Time a single inference so the PoC also reports rough latency.
    let t0 = std::time::Instant::now();
    let out = model.run_single(&input)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    let out_slice = out
        .data_as_slice::<f32>()
        .map_err(|e| anyhow!("output data_as_slice: {e:?}"))?;

    println!("inference: {ms:.1} ms");
    println!("out shape: {:?}  (numel {})", out.shape(), out_slice.len());

    ensure!(
        out_slice.len() == expected.len(),
        "numel mismatch: rust={} reference={}",
        out_slice.len(),
        expected.len()
    );

    // Compare element-wise against the reference. GO/NO-GO is decided on the
    // absolute error; the relative error is reported for context only.
    let max_abs = out_slice
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let max_rel = out_slice
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1e-6))
        .fold(0f32, f32::max);

    println!("max abs diff vs reference: {max_abs:.3e}");
    println!("max rel diff vs reference: {max_rel:.3e}");

    let tol = 1e-3f32;
    if max_abs < tol {
        println!("\nGO - Rust output matches TVM Python reference (tol {tol:.0e})");
        Ok(())
    } else {
        Err(anyhow!(
            "NO-GO - max diff {max_abs:.3e} >= tolerance {tol:.0e}"
        ))
    }
}
