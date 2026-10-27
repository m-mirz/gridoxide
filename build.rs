fn main() {
    // Without at least one `rerun-if` line cargo reruns this script whenever
    // *any* file in the package changes — which, in a build with none of the
    // features below, makes every edit to a test rebuild the library and
    // relink all of the test binaries. Each feature declares its own inputs.
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "opf-highs")]
    highs::build();
    #[cfg(feature = "opf-ipopt")]
    ipopt::build();
    #[cfg(feature = "smallsignal-arpack")]
    arpack::build();
    #[cfg(feature = "capi")]
    capi::build();
}

/// Generates `include/gridoxide.h` from the C ABI in `src/capi/`, only when
/// the `capi` feature is enabled.
///
/// **The mirror image of the `bindgen` modules** — the ones below, and `klu`
/// and `pardiso` in `crates/gridoxide-sparse/build.rs`. They run `bindgen` over
/// a system header so Rust can call a C library; this runs `cbindgen` over our
/// own source so a C library can call us. Same shape, opposite direction.
///
/// Two departures from those, both deliberate:
///
/// - **The output goes into the source tree, not `$OUT_DIR`.** A header in
///   `$OUT_DIR` is unreachable for the case this feature exists to serve — a
///   C++ project consuming a prebuilt `libgridoxide.a` with no cargo anywhere
///   in its build. So `include/gridoxide.h` is generated *and committed*, and
///   `tests/capi_test.rs` fails if the committed copy has drifted.
/// - **A read-only source tree is a warning, not an error.** Generating into
///   the tree is a convenience for developers; it must not be the thing that
///   stops a vendored or sandboxed build from compiling at all.
#[cfg(feature = "capi")]
mod capi {
    use std::env;
    use std::path::PathBuf;

    pub fn build() {
        let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let header = manifest.join("include/gridoxide.h");

        let generated = match generate(&manifest).map(inject_abi_version) {
            Some(text) => text,
            None => {
                println!(
                    "cargo:warning=cbindgen could not parse src/capi; \
                          include/gridoxide.h was left untouched"
                );
                return;
            }
        };

        // Written only when it actually changed. Rewriting an identical file
        // would bump its mtime on every build, which for a header means
        // recompiling every C++ translation unit that includes it.
        let current = std::fs::read_to_string(&header).unwrap_or_default();
        if current == generated {
            println!("cargo:rerun-if-changed=src/capi");
            return;
        }
        if let Some(parent) = header.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if std::fs::write(&header, &generated).is_err() {
            println!(
                "cargo:warning=couldn't write {} (read-only tree?); the committed header \
                 may be stale — run `cargo build --features capi` in a writable checkout",
                header.display()
            );
        }
        println!("cargo:rerun-if-changed=src/capi");
    }

    /// Adds `GRIDOXIDE_ABI_VERSION` to the header, taken from the Rust
    /// constant rather than written twice.
    ///
    /// cbindgen could export it as a constant, but only by exporting *every*
    /// `pub const` in the crate — `WYE`, `RANK_TOLERANCE` and a dozen others
    /// that are not part of this ABI. Injecting the one that is keeps the
    /// header honest and makes drift between the two impossible by
    /// construction.
    fn inject_abi_version(header: String) -> String {
        let version = include_str!("src/capi/mod.rs")
            .lines()
            .find_map(|line| {
                line.trim()
                    .strip_prefix("pub const ABI_VERSION: u32 = ")
                    .and_then(|rest| rest.trim_end_matches(';').parse::<u32>().ok())
            })
            .expect("src/capi/mod.rs must declare `pub const ABI_VERSION: u32 = <n>;`");
        // Written as a literal multi-line string rather than with `\`
        // continuations, which strip leading whitespace and so ate the aligned
        // `*` out of the block comment.
        let define = format!(
            r#"
/**
 * The ABI this header describes. Compare it against `gridoxide_abi_version()`
 * at run time when linking a prebuilt library, so a layout mismatch fails
 * early instead of quietly corrupting a struct.
 */
#define GRIDOXIDE_ABI_VERSION {version}
"#
        );
        match header.find("#define GRIDOXIDE_H") {
            Some(at) => {
                let end = at + "#define GRIDOXIDE_H".len();
                format!("{}{}{}", &header[..end], define, &header[end..])
            }
            None => header,
        }
    }

    /// Parses `src/capi/` and nothing else. `with_crate` would read every
    /// module regardless of `#[cfg]`, so a hand-written `extern "C"` block
    /// elsewhere (the BTF/AMD test oracle's, ARPACK's) or a second type named
    /// like the first (`rao::Limits` and `dynamics::models::Limits`) landed in
    /// the header as if it were this ABI.
    fn generate(manifest: &std::path::Path) -> Option<String> {
        let config = cbindgen::Config::from_root_or_default(manifest);
        cbindgen::Builder::new()
            .with_src(manifest.join("src/capi/mod.rs"))
            .with_config(config)
            .generate()
            .ok()
            .map(|bindings| {
                let mut buffer = Vec::new();
                bindings.write(&mut buffer);
                String::from_utf8_lossy(&buffer).into_owned()
            })
    }
}

/// Links a system ARPACK for the `smallsignal-arpack` cross-check.
///
/// **The one module here that does not run `bindgen`,** and the reason is worth
/// stating rather than leaving as an inconsistency. ARPACK's whole C surface
/// here is two functions — `znaupd_c` and `zneupd_c`, the ISO_C_BINDING
/// wrappers arpack-ng exports — and they are declared by hand in
/// `src/dynamics/arpack.rs`. Generating two declarations would mean requiring
/// `arpack.h`, which distributions place inconsistently (Debian's
/// `libarpack2-dev` puts it at `/usr/include/arpack/arpack.h`, others at the
/// include root, and the runtime package ships none), to save writing eighteen
/// argument types once. The declarations are checked the only way that
/// matters: by the cross-check test, which disagrees loudly with our own
/// solver if a single argument is wrong.
///
/// Needs a linkable `libarpack.so` — the unversioned symlink, which comes from
/// the development package (`apt install libarpack2-dev`). Set `ARPACK_ROOT` if
/// it lives somewhere other than `/usr`.
#[cfg(feature = "smallsignal-arpack")]
mod arpack {
    use std::env;
    use std::path::PathBuf;

    pub fn build() {
        let root = PathBuf::from(env::var("ARPACK_ROOT").unwrap_or_else(|_| "/usr".to_string()));
        let lib_dir = [
            root.join("lib")
                .join(env::var("CARGO_CFG_TARGET_ARCH").map_or_else(
                    |_| "x86_64-linux-gnu".to_string(),
                    |arch| format!("{arch}-linux-gnu"),
                )),
            root.join("lib"),
            root.join("lib64"),
            root.clone(),
        ]
        .into_iter()
        .find(|p| p.join("libarpack.so").is_file())
        .unwrap_or_else(|| {
            panic!(
                "couldn't find `libarpack.so` under {} — the `smallsignal-arpack` \
                 feature needs a local ARPACK *with its development files* (on \
                 Debian/Ubuntu: `apt install libarpack2-dev`; the runtime package \
                 alone ships only `libarpack.so.2`, which the linker will not \
                 take). Set ARPACK_ROOT if it is installed elsewhere.",
                root.display()
            )
        });

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=arpack");
        println!("cargo:rerun-if-env-changed=ARPACK_ROOT");
    }
}

/// Links a system IPOPT install and generates its FFI bindings, only when the
/// `opf-ipopt` feature is enabled — see `src/opf/ipopt.rs`.
///
/// The third instance of the same shape as `pardiso` (in
/// `crates/gridoxide-sparse/build.rs`) and [`highs`]: nothing
/// vendored, nothing compiled from source, bindings generated against the
/// install's own header. IPOPT is EPL-2.0, and this arrangement is what keeps
/// that off `Cargo.toml`'s `license` field — the crate redistributes no
/// IPOPT-derived code, it only links a library the user installed. The same
/// argument the `pardiso` feature makes for proprietary oneMKL.
#[cfg(feature = "opf-ipopt")]
mod ipopt {
    use std::env;
    use std::path::PathBuf;

    pub fn build() {
        let root = PathBuf::from(env::var("IPOPT_ROOT").unwrap_or_else(|_| "/usr".to_string()));

        // Unlike HiGHS — whose Ubuntu `.pc` has a doubled prefix in every path
        // and is therefore unusable — IPOPT's is correct here. It is still not
        // used, for consistency and one fewer build dependency: `pkg-config`
        // is not installed on every machine that can build this crate, and
        // probing two directories is not the part of this file worth
        // outsourcing. The paths below match what `ipopt.pc` reports.
        let include_dir = root.join("include/coin-or");
        let header = include_dir.join("IpStdCInterface.h");
        if !header.is_file() {
            panic!(
                "couldn't find {} — the `opf-ipopt` feature needs a local IPOPT \
                 *with its development files* (on Debian/Ubuntu: `apt install \
                 coinor-libipopt-dev`). Set IPOPT_ROOT if it is installed \
                 somewhere other than {}.",
                header.display(),
                root.display()
            );
        }

        let lib_dir = [
            root.join("lib")
                .join(env::var("CARGO_CFG_TARGET_ARCH").map_or_else(
                    |_| "x86_64-linux-gnu".to_string(),
                    |arch| format!("{arch}-linux-gnu"),
                )),
            root.join("lib"),
            root.join("lib64"),
        ]
        .into_iter()
        .find(|p| p.join("libipopt.so").is_file())
        .unwrap_or_else(|| {
            panic!(
                "found IPOPT headers under {} but no `libipopt.so` beside them — \
                 the unversioned symlink comes from the development package",
                include_dir.display()
            )
        });

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=ipopt");
        // IPOPT is C++ internally, as HiGHS is.
        println!("cargo:rustc-link-lib=dylib=stdc++");

        generate_bindings(&include_dir, &header);

        println!("cargo:rerun-if-env-changed=IPOPT_ROOT");
        println!("cargo:rerun-if-changed={}", header.display());
    }

    fn generate_bindings(include_dir: &std::path::Path, header: &std::path::Path) {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let mut builder = bindgen::Builder::default()
            .header(header.to_str().unwrap())
            .clang_arg(format!("-I{}", include_dir.display()))
            // `ipopt.pc` puts this in its own Cflags; the header's `bool`
            // handling depends on it.
            .clang_arg("-DHAVE_CSTDDEF")
            .allowlist_function("(Create|Free|Add|Open|Close|Set|Get)?Ipopt.*|SetIntermediateCallback")
            .allowlist_type(
                "Ipopt.*|Eval_.*|Intermediate_CB|ip(index|number)|UserDataPtr|ApplicationReturnStatus",
            )
            .allowlist_var("IPOPT_.*");

        // Same libclang-without-a-full-clang-toolchain fallback the other
        // bindgen invocations need; kept per-module rather than shared, as
        // they all are, so each feature's module stands alone.
        if let Some(gcc_builtin_include) = find_gcc_builtin_include() {
            builder = builder.clang_arg(format!("-I{}", gcc_builtin_include.display()));
        }

        // `ipindex` and `ipnumber` are typedefs whose width the install
        // chooses (`int`/`int64_t`, `double`/`float`). Reading them from the
        // installed header is the whole reason these bindings are generated:
        // a hard-coded guess would compile and then misread every index array
        // crossed.
        let bindings = builder
            .generate()
            .expect("failed to generate IPOPT FFI bindings");

        bindings
            .write_to_file(out_dir.join("ipopt_bindings.rs"))
            .expect("failed to write IPOPT FFI bindings");
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

/// Links a system HiGHS install and generates its FFI bindings, only when the
/// `opf-highs` feature is enabled — see `src/opf/highs.rs`.
///
/// Deliberately the same shape as `pardiso` in `crates/gridoxide-sparse/build.rs`:
/// nothing vendored, nothing
/// compiled from source, bindings generated against the install's own header.
/// The differences from that module are only where HiGHS's layout differs.
#[cfg(feature = "opf-highs")]
mod highs {
    use std::env;
    use std::path::PathBuf;

    pub fn build() {
        // `/usr` is the default because that is where every distribution
        // package lands; `HIGHS_ROOT` covers a local build or a non-standard
        // prefix, playing the role `MKLROOT` plays for `pardiso`.
        let root = PathBuf::from(env::var("HIGHS_ROOT").unwrap_or_else(|_| "/usr".to_string()));

        // NOT pkg-config, and not by oversight. HiGHS ships `highs.pc.in`, but
        // Ubuntu's built `highs.pc` (1.12.0+ds1-3ubuntu1) has a doubled
        // prefix in every path — `libdir=/usr/usr/lib/x86_64-linux-gnu`,
        // `includedir=/usr/usr/include/highs` — so it points nowhere. Probing
        // the standard locations directly is both correct here and one fewer
        // build dependency. Revisit only if that packaging bug is fixed *and*
        // something needs a prefix this probe cannot find.
        let include_dir = root.join("include/highs");
        let header = include_dir.join("interfaces/highs_c_api.h");
        if !header.is_file() {
            panic!(
                "couldn't find {} — the `opf-highs` feature needs a local HiGHS \
                 *with its development files* (on Debian/Ubuntu: `apt install \
                 libhighs-dev`; the runtime `libhighs1` package alone is not \
                 enough, as it ships neither the headers nor the `libhighs.so` \
                 link symlink). Set HIGHS_ROOT if it is installed somewhere \
                 other than {}.",
                header.display(),
                root.display()
            );
        }

        // Distributions put the shared library in the multiarch directory;
        // a from-source install uses a flat `lib/`. Probe both, as `pardiso`
        // probes `lib` and `lib/intel64`.
        let lib_dir = [
            root.join("lib")
                .join(env::var("CARGO_CFG_TARGET_ARCH").map_or_else(
                    |_| "x86_64-linux-gnu".to_string(),
                    |arch| format!("{arch}-linux-gnu"),
                )),
            root.join("lib"),
            root.join("lib64"),
        ]
        .into_iter()
        .find(|p| p.join("libhighs.so").is_file())
        .unwrap_or_else(|| {
            panic!(
                "found HiGHS headers under {} but no `libhighs.so` beside them — \
                 `libhighs.so.1` alone is not linkable, and the unversioned \
                 symlink comes from the development package (`libhighs-dev`)",
                include_dir.display()
            )
        });

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=highs");
        // HiGHS is C++ internally even though the API crossed here is pure C,
        // so the C++ standard library has to come along.
        println!("cargo:rustc-link-lib=dylib=stdc++");

        generate_bindings(&include_dir, &header);

        println!("cargo:rerun-if-env-changed=HIGHS_ROOT");
        println!("cargo:rerun-if-changed={}", header.display());
    }

    fn generate_bindings(include_dir: &std::path::Path, header: &std::path::Path) {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let mut builder = bindgen::Builder::default()
            .header(header.to_str().unwrap())
            // The header's own includes (`util/HighsInt.h`, `HConfig.h`) are
            // written relative to the `highs/` directory, not to itself.
            .clang_arg(format!("-I{}", include_dir.display()))
            .allowlist_function("Highs_.*")
            .allowlist_type("Highs.*")
            .allowlist_var("k?Highs.*");

        // Same libclang-without-a-full-clang-toolchain fallback the other two
        // bindgen invocations need; see `klu::find_gcc_builtin_include` in
        // `crates/gridoxide-sparse/build.rs`.
        if let Some(gcc_builtin_include) = find_gcc_builtin_include() {
            builder = builder.clang_arg(format!("-I{}", gcc_builtin_include.display()));
        }

        // `HighsInt` is `int64_t` or `int` depending on whether the build
        // defined `HIGHSINT64`, and `HConfig.h` records which. Letting bindgen
        // read that is the whole reason this is generated rather than
        // hand-written: hard-coding the wrong width would not fail to compile,
        // it would silently misread every index array handed across.
        let bindings = builder
            .generate()
            .expect("failed to generate HiGHS FFI bindings");

        bindings
            .write_to_file(out_dir.join("highs_bindings.rs"))
            .expect("failed to write HiGHS FFI bindings");
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
