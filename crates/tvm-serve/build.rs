//! Native linking for the tvm-serve binary. tvm-ffi-sys links libtvm_ffi.so; the
//! Relax VM lives in libtvm_runtime.so, linked below. `cargo:rustc-link-arg` does
//! NOT propagate from dependency rlibs (e.g. tvm-relax), so the link setup lives here.
use std::env;

fn main() {
    // Generate gRPC code from the KServe v2 proto using vendored protoc.
    if env::var("PROTOC").is_err() {
        if let Ok(p) = protoc_bin_vendored::protoc_bin_path() {
            env::set_var("PROTOC", p);
        }
    }
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&["proto/grpc_predict_v2.proto"], &["proto"])
        .expect("tonic_build: proto compilation failed");
    println!("cargo:rerun-if-changed=proto/grpc_predict_v2.proto");

    let build_dir = env::var("TVM_BUILD_DIR")
        .unwrap_or_else(|_| "/home/ltrubbiani/tvm/src/tvm-current/build".to_string());
    let lib_dir = format!("{build_dir}/lib");

    println!("cargo:rustc-link-search=native={lib_dir}");

    // libtvm_runtime.so registers the relax.VMExecutable loader via a static
    // initializer we never reference, so --no-as-needed forces the DT_NEEDED entry
    // (else "loader not registered" at runtime); restore --as-needed after.
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-L{lib_dir}");
    println!("cargo:rustc-link-arg=-ltvm_runtime");
    println!("cargo:rustc-link-arg=-Wl,--as-needed");

    // rpath so libtvm_runtime.so is found at runtime without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    // Rebuild if the runtime lib changes (e.g. tvm-current repointed) to avoid an ABI-stale link.
    println!("cargo:rerun-if-changed={lib_dir}/libtvm_runtime.so");
    println!("cargo:rerun-if-env-changed=TVM_BUILD_DIR");
}
