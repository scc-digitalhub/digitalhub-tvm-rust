/*
 * SPDX-FileCopyrightText: © 2025 DSLab - Fondazione Bruno Kessler
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thin Rust wrapper over `tvm-ffi` to load and run a compiled TVM Relax
//! `model.so` through the Relax VirtualMachine.
//!
//! No high-level Rust binding exists for the Relax VM, so it's driven by name
//! over the tvm-ffi C ABI, mirroring the C++ runtime:
//!
//! ```text
//!   lib  = Module::load_from_file("model.so")
//!   vm   = lib["vm_load_executable"]()
//!          vm["vm_initialization"](devtype, devid, alloc, …)
//!   out  = vm[entry](inputs…)
//! ```

use serde::Deserialize;
use tvm_ffi::collections::array::Array;
use tvm_ffi::{AnyView, Function, Module, Tensor};

/// `DLDeviceType::kDLCPU`. CPU-only for now.
const KDLCPU: i32 = 1;
/// `AllocatorType::kPooled`.
const ALLOC_POOLED: i32 = 2;

/// Local error type: `tvm-ffi` errors don't implement `std::error::Error`, so
/// we flatten them into a string we can carry through `?` and `anyhow`.
#[derive(Debug)]
pub enum Error {
    Tvm(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Tvm(m) => write!(f, "tvm-ffi: {m}"),
        }
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Adapter that maps a `tvm-ffi` result into our `Result`, so call sites can
/// use `?` uniformly.
fn ffi<T>(r: tvm_ffi::Result<T>) -> Result<T> {
    r.map_err(|e| Error::Tvm(format!("{e:?}")))
}

fn default_dtype() -> String {
    "float32".to_string()
}

/// One input or output tensor as described by `metadata.json`. `dtype`
/// defaults to `float32` when absent.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<i64>,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    /// Affine quantization params, present only for quantized models (TFLite int8).
    /// A client receiving int8 cannot map the values back to reals without them, so
    /// they travel with the model. Per-axis quantization yields more than one entry,
    /// indexed by `quantized_dimension`.
    #[serde(default)]
    pub scale: Vec<f64>,
    #[serde(default)]
    pub zero_point: Vec<i64>,
    #[serde(default)]
    pub quantized_dimension: Option<i64>,
}

/// The `metadata.json` sidecar emitted alongside `model.so`: it names the VM
/// entry function and the model's input/output signature.
#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    pub entry: String,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub tvm_version: String,
    #[serde(default)]
    pub tvm_git_commit: String,
}

impl Metadata {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("read {path}: {e}"))?;
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse {path}: {e}"))
    }

    pub fn validate_runtime(
        &self,
        runtime_version: &str,
        runtime_commit: &str,
        runtime_arch: &str,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            !runtime_version.is_empty() && runtime_version != "unknown",
            "serve image has no embedded TVM version"
        );
        anyhow::ensure!(
            !runtime_commit.is_empty() && runtime_commit != "unknown",
            "serve image has no embedded TVM source revision"
        );
        anyhow::ensure!(
            !self.tvm_version.is_empty(),
            "model metadata has no tvm_version"
        );
        anyhow::ensure!(
            self.tvm_version == runtime_version,
            "model TVM version {:?} is incompatible with serve image version {:?}",
            self.tvm_version,
            runtime_version
        );
        anyhow::ensure!(
            !self.tvm_git_commit.is_empty(),
            "model metadata has no tvm_git_commit; recompile it with an attested toolkit"
        );
        anyhow::ensure!(
            self.tvm_git_commit == runtime_commit,
            "model TVM revision {:?} is incompatible with serve image revision {:?}",
            self.tvm_git_commit,
            runtime_commit
        );
        validate_llvm_target(&self.target, runtime_arch)
    }
}

#[derive(Default, Deserialize)]
struct LlvmTarget {
    kind: String,
    #[serde(default)]
    mtriple: String,
}

fn validate_llvm_target(target_text: &str, runtime_arch: &str) -> anyhow::Result<()> {
    let target_text = target_text.trim();
    anyhow::ensure!(!target_text.is_empty(), "model metadata has no target");

    let target = if target_text.starts_with('{') {
        serde_json::from_str::<LlvmTarget>(target_text)
            .map_err(|e| anyhow::anyhow!("invalid TVM target {target_text:?}: {e}"))?
    } else {
        let mut fields = target_text.split_whitespace();
        let kind = fields.next().unwrap_or_default().to_string();
        let mtriple = fields
            .find_map(|field| field.strip_prefix("-mtriple="))
            .unwrap_or_default()
            .to_string();
        LlvmTarget { kind, mtriple }
    };

    anyhow::ensure!(
        target.kind == "llvm",
        "serve image supports LLVM CPU models, got target kind {:?}",
        target.kind
    );
    if target.mtriple.is_empty() {
        return Ok(());
    }

    let target_arch = target
        .mtriple
        .split('-')
        .next()
        .and_then(canonical_arch)
        .ok_or_else(|| anyhow::anyhow!("unsupported LLVM target triple {:?}", target.mtriple))?;
    let runtime_arch = canonical_arch(runtime_arch)
        .ok_or_else(|| anyhow::anyhow!("unsupported serve image architecture {runtime_arch:?}"))?;
    anyhow::ensure!(
        target_arch == runtime_arch,
        "model target triple {:?} is incompatible with runtime architecture {:?}",
        target.mtriple,
        runtime_arch
    );
    Ok(())
}

fn canonical_arch(arch: &str) -> Option<&'static str> {
    match arch.to_ascii_lowercase().as_str() {
        "x86_64" | "amd64" => Some("x86_64"),
        "aarch64" | "arm64" => Some("aarch64"),
        arch if arch.starts_with("arm") => Some("arm"),
        "riscv64" => Some("riscv64"),
        "powerpc64" | "powerpc64le" | "ppc64le" => Some("powerpc64"),
        "s390x" => Some("s390x"),
        _ => None,
    }
}

/// A loaded model ready for inference. `entry` borrows into the VM, which borrows
/// into the DSO, so all three are kept alive. None are `Send`/`Sync`, so a
/// `RelaxModel` lives on one dedicated inference thread.
pub struct RelaxModel {
    /// The compiled `model.so`; kept alive because the VM lives inside it.
    _lib: Module,
    /// The Relax VM instance created from the DSO.
    _vm: Module,
    /// The model's entry `PackedFunc`, resolved once at load time.
    entry: Function,
}

impl RelaxModel {
    /// Loads `model.so` and initializes the VM on CPU.
    pub fn load(so_path: &str, entry: &str) -> Result<Self> {
        let lib = ffi(Module::load_from_file(so_path))?;

        let loader = ffi(lib.get_function("vm_load_executable"))?;
        let vm: Module = ffi(loader.call_packed(&[]).and_then(|any| any.try_into()))?;

        // vm_initialization wants one (device_type, device_id, alloc_type) triple
        // per device: compute then host. Both are CPU, so the same triple twice.
        let init = ffi(vm.get_function("vm_initialization"))?;
        ffi(init.call_tuple((KDLCPU, 0i32, ALLOC_POOLED, KDLCPU, 0i32, ALLOC_POOLED)))?;

        // Resolve the entry PackedFunc once, so `run` is a direct call.
        let entry_fn = ffi(vm.get_function(entry))?;

        Ok(Self {
            _lib: lib,
            _vm: vm,
            entry: entry_fn,
        })
    }

    /// Multi-input / multi-output inference. Relax returns either a single
    /// Tensor or an Array of Tensors; we normalize both to a Vec.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        // Borrow each input as an AnyView (no copy) for the type-erased entry.
        let views: Vec<AnyView> = inputs.iter().map(AnyView::from).collect();
        let out = ffi(self.entry.call_packed(&views))?;

        // Single-output models return a bare Tensor; try that, else unpack an Array.
        if let Some(t) = AnyView::from(&out).try_as::<Tensor>() {
            return Ok(vec![t]);
        }
        let arr: Array<Tensor> = ffi(out.try_into())?;
        let mut tensors = Vec::with_capacity(arr.len());
        for i in 0..arr.len() {
            tensors.push(ffi(arr.get(i))?);
        }
        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// metadata.json carries the affine params of a quantized model, and serde must
    /// keep them: an unknown field would be dropped silently, leaving a client unable
    /// to interpret the int8 it receives.
    #[test]
    fn metadata_parses_quantization_params() {
        let raw = r#"{
            "entry": "main",
            "inputs":  [{"name":"images","shape":[1,224,224,3],"dtype":"int8",
                         "scale":[0.003921568859368563],"zero_point":[-128]}],
            "outputs": [{"name":"out","shape":[1,56,1029],"dtype":"int8",
                         "scale":[0.0172492116689682],"zero_point":[-35]}]
        }"#;
        let m: Metadata = serde_json::from_str(raw).unwrap();
        assert_eq!(m.inputs[0].scale, vec![0.003921568859368563]);
        assert_eq!(m.inputs[0].zero_point, vec![-128]);
        assert_eq!(m.outputs[0].zero_point, vec![-35]);
    }

    /// Per-axis quantization indexes the entries by an axis, which must survive too.
    #[test]
    fn metadata_parses_per_axis_quantization() {
        let raw = r#"{"entry":"main",
            "inputs":[{"name":"w","shape":[64,3,3,3],"dtype":"int8",
                       "scale":[0.1,0.2],"zero_point":[0,0],"quantized_dimension":3}],
            "outputs":[]}"#;
        let m: Metadata = serde_json::from_str(raw).unwrap();
        assert_eq!(m.inputs[0].scale.len(), 2);
        assert_eq!(m.inputs[0].quantized_dimension, Some(3));
    }

    /// A float model has no params: the vectors stay empty rather than zeroed, and it
    /// is that emptiness that tells "not quantized" from "quantized with scale 0".
    /// Also the non-regression on the old metadata.json, which had no such fields.
    #[test]
    fn metadata_float_model_has_no_quantization_params() {
        let raw = r#"{"entry":"main",
            "inputs":[{"name":"images","shape":[1,3,640,640],"dtype":"float32"}],
            "outputs":[{"name":"out","shape":[1,84,8400],"dtype":"float32"}]}"#;
        let m: Metadata = serde_json::from_str(raw).unwrap();
        assert!(m.inputs[0].scale.is_empty());
        assert!(m.inputs[0].zero_point.is_empty());
        assert_eq!(m.inputs[0].quantized_dimension, None);
        assert_eq!(m.outputs[0].dtype, "float32");
    }

    /// dtype keeps defaulting to float32 when metadata.json omits it.
    #[test]
    fn metadata_dtype_defaults_to_float32() {
        let raw = r#"{"entry":"main","inputs":[{"name":"x","shape":[1]}],"outputs":[]}"#;
        let m: Metadata = serde_json::from_str(raw).unwrap();
        assert_eq!(m.inputs[0].dtype, "float32");
    }

    #[test]
    fn metadata_accepts_matching_runtime_identity_and_target() {
        let raw = r#"{"entry":"main","inputs":[],"outputs":[],
            "tvm_version":"0.26.0","tvm_git_commit":"c7b458e",
            "target":"{\"kind\":\"llvm\",\"mtriple\":\"x86_64-pc-linux-gnu\"}"}"#;
        let m: Metadata = serde_json::from_str(raw).unwrap();
        m.validate_runtime("0.26.0", "c7b458e", "x86_64").unwrap();
    }

    #[test]
    fn metadata_rejects_incompatible_runtime_identity_and_target() {
        let mut m: Metadata = serde_json::from_str(
            r#"{"entry":"main","inputs":[],"outputs":[],
                "tvm_version":"0.26.0","tvm_git_commit":"c7b458e","target":"llvm"}"#,
        )
        .unwrap();

        assert!(m.validate_runtime("unknown", "c7b458e", "x86_64").is_err());
        assert!(m.validate_runtime("0.25.0", "c7b458e", "x86_64").is_err());
        assert!(m.validate_runtime("0.26.0", "different", "x86_64").is_err());
        m.target = "cuda".to_string();
        assert!(m.validate_runtime("0.26.0", "c7b458e", "x86_64").is_err());
        m.target = r#"{"kind":"llvm","mtriple":"aarch64-linux-gnu"}"#.to_string();
        assert!(m.validate_runtime("0.26.0", "c7b458e", "x86_64").is_err());
    }
}
