fn main() {
    #[cfg(feature = "cuda")]
    cudss::build();
}

/// Links cuDSS and the CUDA runtime, compiles the Newton step's kernels
/// (`cuda/kernels.cu`) with nvcc, and generates the FFI bindings of all
/// three, only when the `cuda` feature is enabled — see `src/cuda.rs`.
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
///   linking needs the real toolkit. (Compiling the kernels needs nvcc,
///   which a check without CUDA skips: set `GRIDOXIDE_GPU_NO_NVCC`.)
/// - `CUDA_ARCH` (default `sm_80`, the A100): the architecture the kernels
///   are compiled for; a mismatch is a runtime "no kernel image" error.
/// - nvcc is `$CUDA_HOME/bin/nvcc` if it exists, else `nvcc` on the path.
///
/// Both library directories also go on the run path of this crate's own
/// tests and examples, so they run without `LD_LIBRARY_PATH`. A link
/// argument does not reach a dependent, so they are also handed on as
/// `DEP_CUDSS_LIB_DIRS`, for a dependent's build script to put on its own.
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

        require_cudss_1(&cudss_include.join("cudss.h"));

        // Not checked for existence: `cargo check` against the stub headers
        // never links, and a missing library is the linker's clear error.
        let lib_dirs = [cudss_dir.join("lib"), cuda_home.join("lib64")];
        for lib_dir in &lib_dirs {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
        // For a dependent's own run path, as `DEP_CUDSS_LIB_DIRS` (this
        // package `links = "cudss"`): gridoxide's `gpu` feature puts them on
        // the Python extension's.
        let joined: Vec<String> = lib_dirs.iter().map(|d| d.display().to_string()).collect();
        println!("cargo:lib_dirs={}", joined.join(":"));
        if env::var_os("GRIDOXIDE_GPU_NO_NVCC").is_none() {
            compile_kernels(&cuda_home, &cuda_include);
        }
        println!("cargo:rustc-link-lib=dylib=cudss");
        println!("cargo:rustc-link-lib=dylib=cudart");

        generate_bindings(&cudss_include, &cuda_include);

        for var in [
            "CUDSS_DIR",
            "CUDA_HOME",
            "CUDA_INCLUDE_DIR",
            "CUDA_ARCH",
            "GRIDOXIDE_GPU_NO_NVCC",
        ] {
            println!("cargo:rerun-if-env-changed={var}");
        }
        println!("cargo:rerun-if-changed={}", cudss_include.display());
        println!("cargo:rerun-if-changed=cuda/kernels.cu");
        println!("cargo:rerun-if-changed=cuda/kernels.h");
    }

    /// cuDSS 0.8.0 computes wrong solutions in a uniform batch (NaN for some
    /// systems from index 216 on, measured on an A100; `plans/GPU_BATCH_PLAN.md`),
    /// so this crate needs 1.0 or later, read from the header's
    /// `CUDSS_VERSION_MAJOR`.
    fn require_cudss_1(header: &Path) {
        let text = std::fs::read_to_string(header)
            .unwrap_or_else(|e| panic!("couldn't read {}: {e}", header.display()));
        let major = text
            .lines()
            .find_map(|l| l.trim().strip_prefix("#define CUDSS_VERSION_MAJOR"))
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or_else(|| panic!("{} has no CUDSS_VERSION_MAJOR", header.display()));
        if major < 1 {
            panic!(
                "{} is cuDSS {major}.x: gridoxide-gpu needs cuDSS 1.0 or later, as 0.8.0 \
                 computes wrong solutions in a uniform batch (see plans/GPU_BATCH_PLAN.md)",
                header.display()
            );
        }
    }

    fn compile_kernels(cuda_home: &Path, cuda_include: &Path) {
        // cc's CUDA mode finds nvcc through the NVCC variable (else `nvcc` on
        // the path) and takes `Build::compiler` for the *host* compiler nvcc
        // wraps; setting nvcc there instead, as the old gpu-nvidia branch
        // first did, drops cc's -Xcompiler wrapping of the host flags.
        let nvcc = cuda_home.join("bin/nvcc");
        if env::var_os("NVCC").is_none() && nvcc.is_file() {
            // SAFETY: a build script is single-threaded.
            unsafe { env::set_var("NVCC", &nvcc) };
        }
        let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
        // The shared CUDA runtime, the one libcudss loads: cc's default, the
        // static one, would put a second runtime into the process.
        cc::Build::new()
            .cuda(true)
            .cudart("shared")
            .file("cuda/kernels.cu")
            .include(cuda_include)
            .flag(format!("-arch={arch}"))
            .compile("gridoxide_gpu_kernels");
    }

    fn generate_bindings(cudss_include: &Path, cuda_include: &Path) {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let mut builder = bindgen::Builder::default()
            .header_contents(
                "cudss_wrapper.h",
                "#include <cudss.h>\n#include <cuda_runtime.h>\n#include \"kernels.h\"\n",
            )
            .clang_arg(format!("-I{}", cudss_include.display()))
            .clang_arg(format!("-I{}", cuda_include.display()))
            .clang_arg("-Icuda")
            // cuDSS's whole C API, the kernels' launchers, and of the CUDA
            // runtime only what `src/cuda.rs` and `src/cudss.rs` call.
            .allowlist_function("cudss.*|go_.*")
            .allowlist_function(
                "cuda(Malloc|Free|MallocHost|FreeHost|Memcpy|MemcpyAsync|GetErrorString|Stream(Create|Destroy|Synchronize|WaitEvent)|Event(CreateWithFlags|Record|Synchronize|Destroy))",
            )
            .allowlist_type("cudss.*|cudaError.*|cudaDataType.*|cudaMemcpyKind|cudaStream_t|cudaEvent_t")
            // The value types are CUDA's `CUDA_R_*`: cuDSS 1.0 makes
            // `CUDSS_R_*` macros over them, which bindgen cannot expand.
            .allowlist_var("CUDSS_.*|CUDA_R_(32I|64F)|cudaEventDisableTiming")
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
