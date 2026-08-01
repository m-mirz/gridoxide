fn main() {
    #[cfg(feature = "klu")]
    klu::build();
    #[cfg(feature = "pardiso")]
    pardiso::build();
}

/// Compiles the vendored SuiteSparse KLU solver (`vendor/suitesparse/`) and
/// generates its FFI bindings, only when the `klu` feature is enabled — see
/// `src/sparse_klu.rs` and `docs/src/solvers/backends.md` for why this
/// exists. `vendor/suitesparse/PROVENANCE.md` documents exactly what was
/// vendored, from where, and its licensing (KLU and BTF are
/// LGPL-2.1-or-later, which is why this whole integration is opt-in).
#[cfg(feature = "klu")]
mod klu {
    use std::env;
    use std::path::{Path, PathBuf};

    pub fn build() {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let vendor = manifest_dir.join("vendor/suitesparse");

        if env::var_os("CARGO_FEATURE_KLU_DYNAMIC").is_some() {
            // Link a system-installed libklu.so instead of compiling the
            // vendored source statically, preserving the LGPL relinking
            // right for anyone who needs strict compliance.
            println!("cargo:rustc-link-lib=dylib=klu");
        } else {
            build_static(&vendor);
        }

        generate_bindings(&vendor);

        println!("cargo:rerun-if-changed=vendor/suitesparse");
    }

    /// Compiles the int32 (non-`DLONG`) real and complex KLU variant, plus
    /// its AMD/COLAMD/BTF/SuiteSparse_config dependencies. gridoxide's
    /// networks are nowhere near the multi-billion-entry range `DLONG`
    /// (SuiteSparse's 64-bit index build) exists for, so only the plain
    /// `int32_t`-indexed functions (`klu_*`/`klu_z_*`, not `klu_l_*`/
    /// `klu_zl_*`) are compiled and bound.
    fn build_static(vendor: &Path) {
        let mut build = cc::Build::new();
        build
            .include(vendor.join("SuiteSparse_config"))
            .include(vendor.join("AMD/Include"))
            .include(vendor.join("COLAMD/Include"))
            .include(vendor.join("BTF/Include"))
            .include(vendor.join("KLU/Include"))
            .warnings(false);

        build.file(vendor.join("SuiteSparse_config/SuiteSparse_config.c"));

        for f in [
            "amd_1", "amd_2", "amd_aat", "amd_control", "amd_defaults", "amd_dump", "amd_info",
            "amd_order", "amd_post_tree", "amd_postorder", "amd_preprocess", "amd_valid", "amd_version",
        ] {
            build.file(vendor.join(format!("AMD/Source/{f}.c")));
        }

        for f in ["colamd", "colamd_version"] {
            build.file(vendor.join(format!("COLAMD/Source/{f}.c")));
        }

        for f in ["btf_maxtrans", "btf_order", "btf_strongcomp", "btf_version"] {
            build.file(vendor.join(format!("BTF/Source/{f}.c")));
        }

        for f in [
            // real (int32)
            "klu", "klu_analyze", "klu_analyze_given", "klu_defaults", "klu_diagnostics", "klu_dump",
            "klu_extract", "klu_factor", "klu_free_numeric", "klu_free_symbolic", "klu_kernel",
            "klu_memory", "klu_refactor", "klu_scale", "klu_solve", "klu_sort", "klu_tsolve", "klu_version",
            // complex (int32)
            "klu_z", "klu_z_diagnostics", "klu_z_dump", "klu_z_extract", "klu_z_factor",
            "klu_z_free_numeric", "klu_z_kernel", "klu_z_refactor", "klu_z_scale", "klu_z_solve",
            "klu_z_sort", "klu_z_tsolve",
        ] {
            build.file(vendor.join(format!("KLU/Source/{f}.c")));
        }

        build.compile("klu_vendored");
    }

    fn generate_bindings(vendor: &Path) {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let mut builder = bindgen::Builder::default()
            .header(vendor.join("KLU/Include/klu.h").to_str().unwrap())
            .clang_arg(format!("-I{}", vendor.join("SuiteSparse_config").display()))
            .clang_arg(format!("-I{}", vendor.join("AMD/Include").display()))
            .clang_arg(format!("-I{}", vendor.join("COLAMD/Include").display()))
            .clang_arg(format!("-I{}", vendor.join("BTF/Include").display()))
            .clang_arg(format!("-I{}", vendor.join("KLU/Include").display()))
            .allowlist_function("klu_.*")
            .allowlist_type("klu_.*")
            .allowlist_var("KLU_.*");

        // Only libclang.so is installed in some environments (this one
        // included), not the full clang toolchain with its own bundled
        // freestanding headers (stddef.h, stdarg.h, ...), which libclang's
        // preprocessor still needs. Fall back to gcc's equivalent builtin
        // include directory when found — harmless to add even when a full
        // clang toolchain is already present, since these are freestanding
        // headers with no ABI-specific content.
        if let Some(gcc_builtin_include) = find_gcc_builtin_include() {
            builder = builder.clang_arg(format!("-I{}", gcc_builtin_include.display()));
        }

        let bindings = builder.generate().expect("failed to generate KLU FFI bindings");

        bindings
            .write_to_file(out_dir.join("klu_bindings.rs"))
            .expect("failed to write KLU FFI bindings");
    }

    fn find_gcc_builtin_include() -> Option<PathBuf> {
        let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let output = std::process::Command::new(cc).arg("-print-file-name=include").output().ok()?;
        if !output.status.success() {
            return None;
        }
        let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        path.is_dir().then_some(path)
    }
}

/// Links a locally-installed Intel oneMKL's PARDISO sparse direct solver as
/// a fifth, opt-in `JacobianBackend` (`solver::JacobianBackend::Pardiso`,
/// `src/sparse_pardiso.rs`) — see `docs/src/solvers/backends.md` for why
/// this exists. Unlike `klu`, nothing is vendored: MKL is
/// proprietary, so this only locates and dynamically links a system install
/// (via the `MKLROOT` env var, the same variable Intel's own
/// `setvars.sh` sets) and generates FFI bindings from *that install's own*
/// `mkl_pardiso.h` — no MKL header or source is copied into this repo.
#[cfg(feature = "pardiso")]
mod pardiso {
    use std::env;
    use std::path::PathBuf;

    pub fn build() {
        let mkl_root = env::var("MKLROOT").expect(
            "the `pardiso` feature needs Intel oneMKL installed locally; set MKLROOT \
             (e.g. `source /opt/intel/oneapi/setvars.sh`) before building",
        );
        let mkl_root = PathBuf::from(mkl_root);

        // oneAPI 2024+ puts libmkl_rt.so directly under `lib/`, with
        // `lib/intel64` kept only as a symlink to `lib` for backward
        // compatibility; older oneAPI releases used `lib/intel64` as the
        // real directory instead. Probe both, preferring the newer layout.
        let lib_dir = [mkl_root.join("lib"), mkl_root.join("lib/intel64")]
            .into_iter()
            .find(|p| p.join("libmkl_rt.so").is_file())
            .unwrap_or_else(|| {
                panic!(
                    "couldn't find libmkl_rt.so under {}/lib or {}/lib/intel64 — \
                     is MKLROOT ({}) a valid oneMKL install?",
                    mkl_root.display(),
                    mkl_root.display(),
                    mkl_root.display()
                )
            });

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        // The Single Dynamic Library: one link target, LP64 (32-bit
        // MKL_INT — plenty for gridoxide's network sizes) and sequential
        // threading by default, avoiding MKL's own thread pool interacting
        // unpredictably with any caller-level parallelism (e.g. a batch of
        // scenarios run concurrently). Override via MKL's own env vars
        // (MKL_INTERFACE_LAYER/MKL_THREADING_LAYER) if a different
        // interface/threading layer is needed.
        println!("cargo:rustc-link-lib=dylib=mkl_rt");

        generate_bindings(&mkl_root);

        println!("cargo:rerun-if-env-changed=MKLROOT");
    }

    fn generate_bindings(mkl_root: &std::path::Path) {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let include_dir = mkl_root.join("include");
        let mut builder = bindgen::Builder::default()
            .header(include_dir.join("mkl_pardiso.h").to_str().unwrap())
            .clang_arg(format!("-I{}", include_dir.display()))
            .allowlist_function("pardiso.*")
            .allowlist_type("_MKL_DSS_HANDLE_t|MKL_INT");

        // Same libclang-without-a-full-clang-toolchain fallback `klu`'s own
        // bindgen invocation needs (duplicated rather than shared, since
        // `mod klu` only exists under the separate `klu` feature and this
        // module must build without it) — see `klu::find_gcc_builtin_include`
        // for why this exists.
        if let Some(gcc_builtin_include) = find_gcc_builtin_include() {
            builder = builder.clang_arg(format!("-I{}", gcc_builtin_include.display()));
        }

        let bindings = builder.generate().expect("failed to generate PARDISO FFI bindings");

        bindings
            .write_to_file(out_dir.join("pardiso_bindings.rs"))
            .expect("failed to write PARDISO FFI bindings");
    }

    fn find_gcc_builtin_include() -> Option<PathBuf> {
        let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let output = std::process::Command::new(cc).arg("-print-file-name=include").output().ok()?;
        if !output.status.success() {
            return None;
        }
        let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        path.is_dir().then_some(path)
    }
}
