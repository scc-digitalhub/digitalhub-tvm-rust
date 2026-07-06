//! Native linking for tvm-relax.
//!
//! tvm-ffi-sys already links libtvm_ffi.so (core FFI only). The Relax VM
//! (vm_load_executable, vm_initialization, memory manager, kernels) lives in
//! libtvm_runtime.so, so we link it here and add the rpath.
use std::env;

fn main() {
    let build_dir = env::var("TVM_BUILD_DIR")
        .unwrap_or_else(|_| "/home/ltrubbiani/tvm/src/tvm-current/build".to_string());
    let lib_dir = format!("{build_dir}/lib");

    println!("cargo:rustc-link-search=native={lib_dir}");

    // libtvm_runtime.so registers the `relax.VMExecutable` loader via a static
    // initializer. We never reference its symbols from Rust, so the linker's
    // default --as-needed would drop the DT_NEEDED entry entirely and the
    // initializer would never run ("loader not registered" at runtime). Wrap
    // just -ltvm_runtime in --no-as-needed to force the dependency, then restore
    // --as-needed so it doesn't affect any libraries linked afterwards.
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-L{lib_dir}");
    println!("cargo:rustc-link-arg=-ltvm_runtime");
    println!("cargo:rustc-link-arg=-Wl,--as-needed");

    // Bake the lib dir into the binary's rpath so libtvm_runtime.so is found at
    // runtime without needing LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");

    // Rebuild when the runtime lib itself changes (the tvm-current symlink may
    // be repointed to another TVM version under the same TVM_BUILD_DIR path).
    println!("cargo:rerun-if-changed={lib_dir}/libtvm_runtime.so");
    println!("cargo:rerun-if-env-changed=TVM_BUILD_DIR");
}
