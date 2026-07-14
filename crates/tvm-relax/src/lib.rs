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
}

/// The `metadata.json` sidecar emitted alongside `model.so`: it names the VM
/// entry function and the model's input/output signature.
#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    pub entry: String,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

impl Metadata {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let raw =
            std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("read {path}: {e}"))?;
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse {path}: {e}"))
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
        ffi(init.call_tuple((
            KDLCPU,
            0i32,
            ALLOC_POOLED,
            KDLCPU,
            0i32,
            ALLOC_POOLED,
        )))?;

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
