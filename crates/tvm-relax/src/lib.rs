//! Thin Rust wrapper over `tvm-ffi` to load and run a compiled TVM Relax
//! `model.so` through the Relax VirtualMachine.
//!
//! There is no high-level Rust binding for the Relax VM, so everything is done
//! by name over the tvm-ffi C ABI: we ask the loaded module for named
//! `PackedFunc`s (`vm_load_executable`, `vm_initialization`, and the model's
//! entry function) and call them with type-erased `Any`/`AnyView` values. The
//! four-step dance mirrors what the TVM C++ runtime does internally:
//!
//! ```text
//!   lib  = Module::load_from_file("model.so")   // the compiled DSO
//!   vm   = lib["vm_load_executable"]()          // instantiate the Relax VM module
//!          vm["vm_initialization"](devtype, devid, alloc, …)  // bind device + allocator
//!   out  = vm[entry](inputs…)                   // run the model's entry function
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
/// defaults to `float32` since this runtime is FP32-only.
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

/// A loaded model ready for inference. The handles are all reference-counted
/// TVM objects; we keep the DSO and the VM modules alive for the whole lifetime
/// of the model even though only `entry` is called directly, because the VM and
/// the entry function borrow into them.
///
/// None of these handles are `Send`/`Sync`, which is why the server owns a
/// `RelaxModel` on a single dedicated inference thread.
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

        // `vm_load_executable` is a PackedFunc exported by the DSO; calling it
        // with no args returns the Relax VM as another Module.
        let loader = ffi(lib.get_function("vm_load_executable"))?;
        let vm: Module = ffi(loader.call_packed(&[]).and_then(|any| any.try_into()))?;

        // vm_initialization takes one (device_type, device_id, alloc_type)
        // triple per device. The C++ runtime always passes the compute device
        // followed by the host device; both are CPU here, so we send the same
        // triple twice.
        let init = ffi(vm.get_function("vm_initialization"))?;
        ffi(init.call_tuple((
            KDLCPU,
            0i32,
            ALLOC_POOLED,
            KDLCPU,
            0i32,
            ALLOC_POOLED,
        )))?;

        // Resolve the entry PackedFunc once here rather than by name on every
        // `run`, so inference is a direct call.
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
        // The entry PackedFunc takes type-erased args, so borrow each input
        // Tensor as an AnyView (no copy) and pass the slice positionally.
        let views: Vec<AnyView> = inputs.iter().map(AnyView::from).collect();
        let out = ffi(self.entry.call_packed(&views))?;

        // Single-output models return a bare Tensor; try that first, otherwise
        // treat the result as an Array and unpack each element.
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

    /// Convenience for single-output models.
    pub fn run_single(&self, input: &Tensor) -> Result<Tensor> {
        self.run(std::slice::from_ref(input))?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Tvm("VM produced no output".to_string()))
    }
}
