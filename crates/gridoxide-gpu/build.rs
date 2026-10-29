fn main() {
    #[cfg(feature = "cuda")]
    cudss::build();
}

/// Links cuDSS and the CUDA runtime and generates their FFI bindings, only
/// when the `cuda` feature is enabled — see `src/cudss.rs`.
///
/// The same shape as gridoxide's `opf-ipopt` and `opf-highs` (in the root
/// `build.rs`): nothing vendored, bindings generated against the install's own
/// headers, never hand-written. The old `gpu-nvidia` branch's three FFI bugs
/// were all hand-written signatures that did not match the headers.
///
/// - `CUDSS_DIR` (required): cuDSS's root, with `include/cudss.h` and
///   `lib/libcudss.so` — NVIDIA's redistributable archive unpacked, as
///   `probes/cudss_ubatch/run.sh` fetches it.
/// - `CUDA_HOME` (default `/usr/local/cuda`): the CUDA 12 toolkit, with
///   `include/cuda_runtime.h` and `lib64/libcudart.so`.
/// - `CUDA_INCLUDE_DIR` (default `$CUDA_HOME/include`): where
///   `cuda_runtime.h` is found. Pointing it at `probes/cudss_ubatch/stub`
///   lets `cargo check --features cuda` run on a machine without CUDA; only
///   linking needs the real toolkit.
///
/// Both library directories also go on the run path of this crate's own
/// tests and examples, so they run without `LD_LIBRARY_PATH`; a downstream
/// binary sets its own.
#[cfg(feature = "cuda")]
mod cudss {
    use std::env;
    use std::path::{Path, PathBuf};

    pub fn build() {
        let cudss_dir = PathBuf::from(env::var("CUDSS_DIR").unwrap_or_else(|_| {
            panic!(
                "the `cuda` feature needs CUDSS_DIR, the root of a cuDSS install with \
                 include/cudss.h and lib/libcudss.so (NVIDIA's redistributable archive \
                 unpacked; probes/cudss_ubatch/run.sh fetches a pinned one)"
            )
        }));
        let cuda_home =
            PathBuf::from(env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string()));
        let cuda_include = env::var("CUDA_INCLUDE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| cuda_home.join("include"));
        let cudss_include = cudss_dir.join("include");
        for header in [
            cudss_include.join("cudss.h"),
            cuda_include.join("cuda_runtime.h"),
        ] {
            if !header.is_file() {
                panic!(
                    "couldn't find {} — set CUDSS_DIR, CUDA_HOME or CUDA_INCLUDE_DIR",
                    header.display()
                );
            }
        }

        // Not checked for existence: `cargo check` against the stub headers
        // never links, and a missing library is the linker's clear error.
        for lib_dir in [cudss_dir.join("lib"), cuda_home.join("lib64")] {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
        println!("cargo:rustc-link-lib=dylib=cudss");
        println!("cargo:rustc-link-lib=dylib=cudart");

        generate_bindings(&cudss_include, &cuda_include);

        for var in ["CUDSS_DIR", "CUDA_HOME", "CUDA_INCLUDE_DIR"] {
            println!("cargo:rerun-if-env-changed={var}");
        }
        println!("cargo:rerun-if-changed={}", cudss_include.display());
    }

    fn generate_bindings(cudss_include: &Path, cuda_include: &Path) {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let mut builder = bindgen::Builder::default()
            .header_contents("cudss_wrapper.h", "#include <cudss.h>\n#include <cuda_runtime.h>\n")
            .clang_arg(format!("-I{}", cudss_include.display()))
            .clang_arg(format!("-I{}", cuda_include.display()))
            // cuDSS's whole C API, and of the CUDA runtime only what
            // `src/cudss.rs` calls.
            .allowlist_function("cudss.*")
            .allowlist_function(
                "cuda(Malloc|Free|Memcpy|MemcpyAsync|GetErrorString|Stream(Create|Destroy|Synchronize))",
            )
            .allowlist_type("cudss.*|cudaError.*|cudaDataType.*|cudaMemcpyKind|cudaStream_t")
            // The value types are CUDA's `CUDA_R_*`: cuDSS 0.8.0's
            // `CUDSS_R_*` enum equals them, and 1.0.0 makes `CUDSS_R_*`
            // macros over them that bindgen cannot expand.
            .allowlist_var("CUDSS_.*|CUDA_R_(32I|64F)")
            // Plain constants without the enum's name in front: the C names,
            // `CUDSS_STATUS_SUCCESS`, `cudaSuccess`.
            .default_enum_style(bindgen::EnumVariation::Consts)
            .prepend_enum_name(false)
            .layout_tests(false);

        // Same libclang-without-a-full-clang-toolchain fallback as the root
        // build script's bindgen invocations, kept here so this crate's
        // build stands alone.
        if let Some(gcc_builtin_include) = find_gcc_builtin_include() {
            builder = builder.clang_arg(format!("-I{}", gcc_builtin_include.display()));
        }

        builder
            .generate()
            .expect("failed to generate cuDSS FFI bindings")
            .write_to_file(out_dir.join("cudss_bindings.rs"))
            .expect("failed to write cuDSS FFI bindings");
    }

    fn find_gcc_builtin_include() -> Option<PathBuf> {
        let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let output = std::process::Command::new(cc)
            .arg("-print-file-name=include")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        path.is_dir().then_some(path)
    }
}
