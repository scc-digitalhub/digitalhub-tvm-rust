//! Native linking for the tvm-serve binary. `cargo:rustc-link-arg` does NOT
//! propagate from dependencies to the binary crate, so the libtvm_runtime.so
//! force-load (--no-as-needed) and rpath must be repeated here. See
//! `crates/tvm-relax/build.rs` for the rationale.
use std::env;

fn main() {
    // Generate gRPC code from KServe v2 proto using vendored protoc.
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

    // force-load libtvm_runtime.so so its static initializers register the
    // relax.VMExecutable loader
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-L{lib_dir}");
    println!("cargo:rustc-link-arg=-ltvm_runtime");
    println!("cargo:rustc-link-arg=-Wl,--as-needed");

    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    // Rebuild when the actual runtime lib changes (e.g. the tvm-current symlink
    // is repointed to another TVM version under the SAME TVM_BUILD_DIR path),
    // so we never ship a binary linked against a stale/ABI-mismatched .so.
    println!("cargo:rerun-if-changed={lib_dir}/libtvm_runtime.so");
    println!("cargo:rerun-if-env-changed=TVM_BUILD_DIR");
}
