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
use std::io::Read;
use tvm_ffi::collections::array::Array;
use tvm_ffi::{AnyView, DLDevice, DLDeviceType, Function, Module, Shape, Tensor};

/// `AllocatorType::kPooled`.
const ALLOC_POOLED: i32 = 2;

/// Selected from the compiled artifact's target, never from a serving option.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
}

impl Backend {
    pub fn device(self) -> DLDevice {
        let kind = match self {
            Self::Cpu => DLDeviceType::kDLCPU,
            Self::Cuda => DLDeviceType::kDLCUDA,
        };
        // CUDA ordinal 0 is the first device visible to this container.
        DLDevice::new(kind, 0)
    }

    fn validate_build(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self != Self::Cuda || cfg!(feature = "cuda"),
            "CUDA model requires a CUDA-enabled serve image (build with feature cuda)"
        );
        Ok(())
    }

    fn validate_device(self) -> Result<()> {
        self.validate_build()
            .map_err(|e| Error::Tvm(e.to_string()))?;
        if self == Self::Cuda {
            Function::get_global("device_api.cuda").map_err(|_| {
                Error::Tvm(
                    "CUDA backend is not registered; libtvm_runtime_cuda.so is required".into(),
                )
            })?;
            let get_attr = ffi(Function::get_global("runtime.GetDeviceAttr"))?;
            // DeviceAttrKind::kExist. This reports driver/device availability without allocating.
            let exists: i64 = ffi(get_attr
                .call_tuple((2i32, 0i32, 0i32))
                .and_then(|any| any.try_into()))?;
            if exists == 0 {
                return Err(Error::Tvm(
                    "CUDA device 0 is unavailable; expose a supported NVIDIA GPU and driver to the container".into(),
                ));
            }
        }
        Ok(())
    }
}

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
    pub target_triple: String,
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
    ) -> anyhow::Result<Backend> {
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
        let backend = validate_target(&self.target, runtime_arch)?;
        validate_target_triple(&self.target_triple, runtime_arch)?;
        backend.validate_build()?;
        Ok(backend)
    }
}

#[derive(Default, Deserialize)]
struct Target {
    kind: String,
    #[serde(default)]
    mtriple: String,
    #[serde(default)]
    arch: String,
    #[serde(default)]
    host: Option<Box<Target>>,
}

fn validate_target(target_text: &str, runtime_arch: &str) -> anyhow::Result<Backend> {
    let target_text = target_text.trim();
    anyhow::ensure!(!target_text.is_empty(), "model metadata has no target");

    let target = if target_text.starts_with('{') {
        serde_json::from_str::<Target>(target_text)
            .map_err(|e| anyhow::anyhow!("invalid TVM target {target_text:?}: {e}"))?
    } else {
        let mut fields = target_text.split_whitespace();
        let kind = fields.next().unwrap_or_default().to_string();
        let mtriple = fields
            .find_map(|field| field.strip_prefix("-mtriple="))
            .unwrap_or_default()
            .to_string();
        Target {
            kind,
            mtriple,
            ..Target::default()
        }
    };

    match target.kind.as_str() {
        "llvm" => {
            validate_target_triple(&target.mtriple, runtime_arch)?;
            Ok(Backend::Cpu)
        }
        "cuda" => {
            anyhow::ensure!(
                matches!(canonical_arch(runtime_arch), Some("x86_64" | "aarch64")),
                "CUDA serve image supports only AMD64 and ARM64"
            );
            anyhow::ensure!(
                valid_cuda_arch(&target.arch),
                "CUDA model requires an explicit target arch such as sm_87 or sm_110"
            );
            let host = target.host.as_ref().ok_or_else(|| {
                anyhow::anyhow!("CUDA model requires a nested LLVM host target with mtriple")
            })?;
            anyhow::ensure!(host.kind == "llvm", "CUDA host target must use LLVM");
            anyhow::ensure!(
                !host.mtriple.is_empty(),
                "CUDA host target requires mtriple"
            );
            validate_target_triple(&host.mtriple, runtime_arch)?;
            Ok(Backend::Cuda)
        }
        _ => anyhow::bail!(
            "serve image supports LLVM CPU and optional CUDA models, got target kind {:?}",
            target.kind
        ),
    }
}

fn valid_cuda_arch(arch: &str) -> bool {
    let Some(compute) = arch.strip_prefix("sm_") else {
        return false;
    };
    let compute = if compute
        .as_bytes()
        .last()
        .is_some_and(u8::is_ascii_lowercase)
    {
        &compute[..compute.len() - 1]
    } else {
        compute
    };
    !compute.is_empty() && compute.bytes().all(|byte| byte.is_ascii_digit())
}

fn validate_target_triple(triple: &str, runtime_arch: &str) -> anyhow::Result<()> {
    if triple.is_empty() {
        return Ok(());
    }
    let (_, abi) = triple
        .split_once("-linux-")
        .ok_or_else(|| anyhow::anyhow!("serve image requires a Linux target, got {triple:?}"))?;
    anyhow::ensure!(
        abi.starts_with("gnu"),
        "serve image requires the GNU/glibc ABI, got {triple:?}"
    );
    let target_arch = triple
        .split('-')
        .next()
        .and_then(canonical_arch)
        .ok_or_else(|| anyhow::anyhow!("unsupported LLVM target triple {triple:?}"))?;
    let runtime_arch = canonical_arch(runtime_arch)
        .ok_or_else(|| anyhow::anyhow!("unsupported serve image architecture {runtime_arch:?}"))?;
    anyhow::ensure!(
        target_arch == runtime_arch,
        "model target triple {:?} is incompatible with runtime architecture {:?}",
        triple,
        runtime_arch
    );
    anyhow::ensure!(
        target_arch != "arm" || abi == "gnueabihf",
        "ARM32 serve image requires the hard-float ABI, got {triple:?}"
    );
    Ok(())
}

fn canonical_arch(arch: &str) -> Option<&'static str> {
    match arch.to_ascii_lowercase().as_str() {
        "x86_64" | "amd64" => Some("x86_64"),
        "aarch64" | "arm64" => Some("aarch64"),
        "arm" => Some("arm"),
        arch if arch.starts_with("armv") || arch.starts_with("thumb") => Some("arm"),
        "riscv64" => Some("riscv64"),
        "powerpc64" | "ppc64" => Some("powerpc64"),
        "powerpc64le" | "ppc64le" => Some("powerpc64le"),
        "s390x" => Some("s390x"),
        _ => None,
    }
}

/// Check the actual library before loading executable code.
pub fn validate_library(path: &str, runtime_arch: &str) -> anyhow::Result<()> {
    let mut header = [0u8; 20];
    std::fs::File::open(path)?.read_exact(&mut header)?;
    anyhow::ensure!(&header[..4] == b"\x7fELF", "invalid ELF library {path:?}");
    let (class, endian, machine) = match canonical_arch(runtime_arch) {
        Some("x86_64") => (2, 1, 62),
        Some("aarch64") => (2, 1, 183),
        Some("arm") => (1, 1, 40),
        Some("riscv64") => (2, 1, 243),
        Some("powerpc64le") => (2, 1, 21),
        Some("powerpc64") => (2, 2, 21),
        Some("s390x") => (2, 2, 22),
        _ => anyhow::bail!("unsupported serve image architecture {runtime_arch:?}"),
    };
    let actual = match header[5] {
        1 => u16::from_le_bytes([header[18], header[19]]),
        2 => u16::from_be_bytes([header[18], header[19]]),
        _ => anyhow::bail!("invalid ELF byte order"),
    };
    anyhow::ensure!(
        header[4] == class && header[5] == endian && actual == machine,
        "model ELF class/byte order/machine is incompatible with runtime architecture {runtime_arch:?}"
    );
    Ok(())
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
    backend: Backend,
    transfers: Option<TensorTransfers>,
}

impl RelaxModel {
    /// Loads `model.so` on the artifact's compute device, with a CPU host device.
    pub fn load(so_path: &str, entry: &str, backend: Backend) -> Result<Self> {
        // Fail before loading executable model code; never retry a CUDA artifact on CPU.
        backend.validate_device()?;
        let transfers = if backend == Backend::Cuda {
            Some(TensorTransfers::new()?)
        } else {
            None
        };
        let lib = ffi(Module::load_from_file(so_path))?;

        let loader = ffi(lib.get_function("vm_load_executable"))?;
        let vm: Module = ffi(loader.call_packed(&[]).and_then(|any| any.try_into()))?;

        // vm_initialization wants one (device_type, device_id, alloc_type) triple
        // per device: compute then host. CPU artifacts preserve the original two CPU triples.
        let init = ffi(vm.get_function("vm_initialization"))?;
        ffi(init.call_tuple((
            backend.device().device_type as i32,
            0i32,
            ALLOC_POOLED,
            DLDeviceType::kDLCPU as i32,
            0i32,
            ALLOC_POOLED,
        )))?;

        // Resolve the entry PackedFunc once, so `run` is a direct call.
        let entry_fn = vm.get_function(entry).map_err(|error| {
            Error::Tvm(format!("cannot resolve model entry {entry:?}: {error:?}"))
        })?;

        Ok(Self {
            _lib: lib,
            _vm: vm,
            entry: entry_fn,
            backend,
            transfers,
        })
    }

    /// Inputs begin as ordinary CPU tensors owned by the worker thread.
    pub fn input_from_cpu(&self, input: Tensor) -> Result<Tensor> {
        if input.device().device_type != DLDeviceType::kDLCPU {
            return Err(Error::Tvm("inference input must be staged on CPU".into()));
        }
        match &self.transfers {
            Some(transfers) => transfers.copy(&input, self.backend.device()),
            None => Ok(input),
        }
    }

    /// CPU readers never receive a device pointer. CUDA outputs are copied and synchronized.
    pub fn output_to_cpu(&self, output: &Tensor) -> Result<Tensor> {
        if output.device().device_type == DLDeviceType::kDLCPU {
            return Ok(output.clone());
        }
        if output.device() != self.backend.device() {
            return Err(Error::Tvm(format!(
                "unexpected output device {:?}",
                output.device()
            )));
        }
        match &self.transfers {
            Some(transfers) => transfers.copy(output, Backend::Cpu.device()),
            None => Err(Error::Tvm("CPU model returned a non-CPU tensor".into())),
        }
    }

    /// Multi-input / multi-output inference. Relax returns either a single
    /// Tensor or an Array of Tensors; we normalize both to a Vec.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        // Borrow each input as an AnyView (no copy) for the type-erased entry.
        let views: Vec<AnyView> = inputs.iter().map(AnyView::from).collect();
        let out = ffi(self.entry.call_packed(&views));
        // Finish outstanding CUDA work even when invocation reports an error, before the
        // worker drops the request's input tensors or begins another inference.
        let synced = match &self.transfers {
            Some(transfers) => transfers.synchronize(self.backend.device()),
            None => Ok(()),
        };
        let out = out?;
        synced?;

        // Single-output models return a bare Tensor; try that, else unpack an Array.
        if let Some(t) = AnyView::from(&out).try_as::<Tensor>() {
            return Ok(vec![t]);
        }
        let arr: Array<Tensor> = ffi(out.try_into())?;
        (0..arr.len()).map(|i| ffi(arr.get(i))).collect()
    }
}

/// TVM owns all device allocations and copies. These globals are part of the pinned
/// runtime; no CUDA SDK or compiler is needed to load a precompiled model.
struct TensorTransfers {
    alloc: Function,
    copy: Function,
    sync: Function,
}

impl TensorTransfers {
    fn new() -> Result<Self> {
        Ok(Self {
            alloc: ffi(Function::get_global("runtime.TVMTensorAllocWithScope"))?,
            copy: ffi(Function::get_global("runtime.TVMTensorCopyFromTo"))?,
            sync: ffi(Function::get_global("runtime.Device_StreamSync"))?,
        })
    }

    fn copy(&self, source: &Tensor, destination: DLDevice) -> Result<Tensor> {
        let shape = Shape::from(source.shape());
        let dtype = source.dtype();
        let target: Tensor = ffi(self
            .alloc
            .call_packed(&[
                AnyView::from(&shape),
                AnyView::from(&dtype),
                AnyView::from(&destination),
                AnyView::new(),
            ])
            .and_then(|any| any.try_into()))?;
        // CopyFromTo uses TVM's default stream. Synchronize before host staging or device
        // allocations can be released, including the error path after a native launch.
        let copied = ffi(self.copy.call_tuple((source, &target)));
        let device = if source.device().device_type != DLDeviceType::kDLCPU {
            source.device()
        } else {
            destination
        };
        let synced = self.synchronize(device);
        copied?;
        synced?;
        Ok(target)
    }

    fn synchronize(&self, device: DLDevice) -> Result<()> {
        ffi(self
            .sync
            .call_packed(&[AnyView::from(&device), AnyView::from(&0i64)]))?;
        Ok(())
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

    #[test]
    fn target_triple_checks_platform_and_recorded_architecture() {
        for triple in [
            "x86_64-apple-darwin",
            "x86_64-linux-musl",
            "aarch64-linux-gnu",
        ] {
            assert!(validate_target_triple(triple, "x86_64").is_err());
        }
        assert!(validate_target_triple("armv7l-unknown-linux-gnueabi", "arm").is_err());
        assert!(validate_target_triple("armv7l-unknown-linux-gnueabihf", "arm").is_ok());
        let m: Metadata = serde_json::from_str(
            r#"{"entry":"main","inputs":[],"outputs":[],"target":"llvm",
            "target_triple":"aarch64-linux-gnu","tvm_version":"0.26.0","tvm_git_commit":"c7b458e"}"#,
        ).unwrap();
        assert!(m.validate_runtime("0.26.0", "c7b458e", "x86_64").is_err());
    }

    #[test]
    fn cuda_target_requires_explicit_arch_and_matching_llvm_host() {
        for (host, arch) in [
            ("x86_64-linux-gnu", "x86_64"),
            ("aarch64-linux-gnu", "aarch64"),
        ] {
            for gpu in ["sm_87", "sm_110", "sm_90a"] {
                let target = serde_json::json!({"kind":"cuda", "arch":gpu,
                    "host":{"kind":"llvm", "mtriple":host}})
                .to_string();
                assert_eq!(validate_target(&target, arch).unwrap(), Backend::Cuda);
            }
        }
        for target in [
            "cuda -arch=sm_87",
            r#"{"kind":"cuda","host":{"kind":"llvm","mtriple":"x86_64-linux-gnu"}}"#,
            r#"{"kind":"cuda","arch":"sm_87"}"#,
            r#"{"kind":"cuda","arch":"sm_87","host":{"kind":"llvm"}}"#,
            r#"{"kind":"cuda","arch":"sm_87","host":{"kind":"c","mtriple":"x86_64-linux-gnu"}}"#,
            r#"{"kind":"cuda","arch":"sm_87","host":{"kind":"llvm","mtriple":"aarch64-linux-gnu"}}"#,
            r#"{"kind":"cuda","arch":"sm_87","host":{"kind":"llvm","mtriple":"x86_64-linux-musl"}}"#,
            r#"{"kind":"cuda","arch":"native","host":{"kind":"llvm","mtriple":"x86_64-linux-gnu"}}"#,
        ] {
            assert!(validate_target(target, "x86_64").is_err(), "{target}");
        }
        let arm32 = r#"{"kind":"cuda","arch":"sm_87","host":{"kind":"llvm","mtriple":"armv7-linux-gnueabihf"}}"#;
        assert!(validate_target(arm32, "arm").is_err());
    }

    #[test]
    fn cpu_target_validation_preserves_all_supported_serving_architectures() {
        for (triple, arch) in [
            ("x86_64-linux-gnu", "x86_64"),
            ("aarch64-linux-gnu", "aarch64"),
            ("armv7l-unknown-linux-gnueabihf", "arm"),
        ] {
            let target = format!("llvm -mtriple={triple}");
            assert_eq!(validate_target(&target, arch).unwrap(), Backend::Cpu);
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn cpu_build_rejects_cuda_before_loading_model_code() {
        let error = RelaxModel::load("/no/model.so", "main", Backend::Cuda)
            .err()
            .unwrap();
        assert!(error.to_string().contains("CUDA-enabled serve image"));
    }

    #[test]
    fn backend_uses_first_visible_device_and_cpu_host_is_distinct() {
        assert_eq!(
            Backend::Cpu.device(),
            DLDevice::new(DLDeviceType::kDLCPU, 0)
        );
        assert_eq!(
            Backend::Cuda.device(),
            DLDevice::new(DLDeviceType::kDLCUDA, 0)
        );
    }

    #[test]
    fn validates_actual_elf_before_loading() {
        let executable = std::env::current_exe().unwrap();
        let path = executable.to_str().unwrap();
        assert!(validate_library(path, std::env::consts::ARCH).is_ok());
        let foreign = if std::env::consts::ARCH == "x86_64" {
            "aarch64"
        } else {
            "x86_64"
        };
        assert!(validate_library(path, foreign).is_err());
        assert!(validate_library("/no/model.so", std::env::consts::ARCH).is_err());
    }
}
