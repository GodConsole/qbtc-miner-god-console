// build.rs
//
// Optional CUDA kernel compilation + optional OpenCL host compilation.
//
// Both are best-effort: if the toolchain / headers are missing or compilation
// fails for any reason, a CPU stub providing the same symbols is compiled
// instead, so the crate always links. The miner detects "zero devices" at
// runtime and falls back to CPU.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    println!("cargo:rerun-if-changed=cuda/qbtc_sha256_miner.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=OPENCL_INCLUDE");
    println!("cargo:rerun-if-env-changed=OPENCL_LIB");
    println!("cargo:rerun-if-env-changed=QBTC_CUDART_STATIC");

    // =================================================================
    // CUDA
    // =================================================================
    match find_cuda_path() {
        Some(cuda_path) => {
            if let Err(e) = build_cuda_kernel(&out_dir, &cuda_path) {
                println!("cargo:warning=CUDA build failed; falling back to the CPU stub.");
                for line in e.lines() {
                    println!("cargo:warning=  {}", line);
                }
                build_cuda_cpu_stub(&out_dir);
            }
        }
        None => {
            println!(
                "cargo:warning=No CUDA toolkit found (set CUDA_PATH to override). \
                 Building without CUDA support."
            );
            build_cuda_cpu_stub(&out_dir);
        }
    }

    // =================================================================
    // OpenCL (same best-effort pattern)
    // =================================================================
    println!("cargo:rerun-if-changed=cuda/qbtc_sha256_miner.cl");
    println!("cargo:rerun-if-changed=cuda/qbtc_sha256_miner_opencl.c");

    let cl_files_present = Path::new("cuda/qbtc_sha256_miner.cl").exists()
        && Path::new("cuda/qbtc_sha256_miner_opencl.c").exists();

    if !cl_files_present {
        println!(
            "cargo:warning=OpenCL source files not found; building OpenCL stub."
        );
        build_opencl_cpu_stub(&out_dir);
        return;
    }

    // Embed the .cl source into a generated header the C host can #include.
    generate_cl_header(&out_dir);

    match find_opencl_include() {
        Some(inc) => {
            if let Err(e) = build_opencl_kernel(&out_dir, &inc) {
                println!("cargo:warning=OpenCL build failed: {}", e);
                build_opencl_cpu_stub(&out_dir);
            }
        }
        None => {
            println!(
                "cargo:warning=OpenCL headers (CL/cl.h) not found; building OpenCL stub."
            );
            build_opencl_cpu_stub(&out_dir);
        }
    }
}

fn is_windows() -> bool {
    env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
}

// =====================================================================
// CUDA helpers
// =====================================================================

fn nvcc_name() -> &'static str {
    if is_windows() { "nvcc.exe" } else { "nvcc" }
}

fn newest_matching_dir(parent: &str, prefix: &str, nvcc: &str) -> Option<String> {
    let mut candidates: Vec<PathBuf> = fs::read_dir(parent)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(prefix))
                .unwrap_or(false)
        })
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .rev()
        .find(|p| p.join("bin").join(nvcc).exists())
        .map(|p| p.to_string_lossy().into_owned())
}

fn find_cuda_path() -> Option<String> {
    let nvcc = nvcc_name();

    if let Ok(p) = env::var("CUDA_PATH") {
        if Path::new(&p).join("bin").join(nvcc).exists() {
            return Some(p);
        }
    }

    if is_windows() {
        return newest_matching_dir(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            "v",
            nvcc,
        );
    }

    for base in ["/usr/local/cuda", "/opt/cuda", "/usr"] {
        if Path::new(base).join("bin").join(nvcc).exists() {
            return Some(base.to_string());
        }
    }
    newest_matching_dir("/usr/local", "cuda-", nvcc)
}

fn build_cuda_kernel(out_dir: &str, cuda_path: &str) -> Result<(), String> {
    let windows = is_windows();
    let nvcc = Path::new(cuda_path).join("bin").join(nvcc_name());
    let obj = PathBuf::from(out_dir).join(if windows { "qbtc_cuda.obj" } else { "qbtc_cuda.o" });
    let lib = PathBuf::from(out_dir).join(if windows {
        "qbtc_cuda_kernel.lib"
    } else {
        "libqbtc_cuda_kernel.a"
    });

    let target = env::var("TARGET").unwrap_or_default();
    let mut nvcc_cmd = Command::new(&nvcc);

    let mut cl_env = Vec::new();
    if windows {
        if let Some(t) = cc::windows_registry::find_tool(&target, "cl.exe") {
            cl_env = t
                .get_envs()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect();
            for (k, v) in &cl_env {
                nvcc_cmd.env(k, v);
            }
        }
    }

    let help_text = Command::new(&nvcc)
        .envs(cl_env.iter().cloned())
        .arg("--help")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();

    let mut supported_archs: Vec<u32> = Command::new(&nvcc)
        .envs(cl_env.iter().cloned())
        .arg("--list-gpu-arch")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|line| line.trim().strip_prefix("compute_")?.parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_default();

    if supported_archs.is_empty() && !help_text.is_empty() {
        for token in help_text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            if let Some(num_str) = token.strip_prefix("compute_") {
                if let Ok(arch) = num_str.parse::<u32>() {
                    if !supported_archs.contains(&arch) {
                        supported_archs.push(arch);
                    }
                }
            }
        }
        supported_archs.sort_unstable();
    }

    let desired_archs = [61u32, 70, 75, 80, 86, 89, 90, 120];
    let active_archs: Vec<u32> = if supported_archs.is_empty() {
        vec![61, 75, 86]
    } else {
        let filtered: Vec<u32> = desired_archs
            .iter()
            .copied()
            .filter(|a| supported_archs.contains(a))
            .collect();
        if filtered.is_empty() { vec![61, 75, 86] } else { filtered }
    };

    nvcc_cmd.args([
        "cuda/qbtc_sha256_miner.cu",
        "-c",
        "-o",
        &obj.to_string_lossy(),
        "-O3",
    ]);

    if help_text.is_empty() || help_text.contains("Wno-deprecated-gpu-targets") {
        nvcc_cmd.arg("-Wno-deprecated-gpu-targets");
    }
    if windows || help_text.contains("allow-unsupported-compiler") {
        nvcc_cmd.arg("-allow-unsupported-compiler");
    }

    for arch in &active_archs {
        nvcc_cmd.args([
            "--generate-code",
            &format!("arch=compute_{0},code=sm_{0}", arch),
        ]);
    }
    if let Some(&min_arch) = active_archs.first() {
        nvcc_cmd.args([
            "--generate-code",
            &format!("arch=compute_{0},code=compute_{0}", min_arch),
        ]);
    }
    if let Some(&max_arch) = active_archs.last() {
        if Some(&max_arch) != active_archs.first() {
            nvcc_cmd.args([
                "--generate-code",
                &format!("arch=compute_{0},code=compute_{0}", max_arch),
            ]);
        }
    }

    if windows {
        nvcc_cmd.args(["-Xcompiler", "/MD"]);
    } else {
        nvcc_cmd.args(["-Xcompiler", "-fPIC"]);
    }

    let output = nvcc_cmd
        .output()
        .map_err(|e| format!("could not run {}: {}", nvcc.display(), e))?;
    if !output.status.success() {
        return Err(format!(
            "nvcc failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    if windows {
        let mut lib_cmd = cc::windows_registry::find_tool(&target, "lib.exe")
            .map(|t| t.to_command())
            .unwrap_or_else(|| Command::new("lib.exe"));
        let status = lib_cmd
            .arg(format!("/OUT:{}", lib.display()))
            .arg(&obj)
            .status()
            .map_err(|e| {
                format!(
                    "could not run lib.exe: {} (no MSVC C++ build tools found, \
                     and lib.exe is not on PATH)",
                    e
                )
            })?;
        if !status.success() {
            return Err(format!("lib.exe failed: {}", status));
        }
        println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_path);
    } else {
        let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
        let status = Command::new(&ar)
            .args(["crs"])
            .arg(&lib)
            .arg(&obj)
            .status()
            .map_err(|e| format!("could not run {}: {}", ar, e))?;
        if !status.success() {
            return Err(format!("{} failed: {}", ar, status));
        }
        for sub in ["lib64", "lib", "lib/x86_64-linux-gnu"] {
            let dir = Path::new(cuda_path).join(sub);
            if dir.exists() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=qbtc_cuda_kernel");
    // QBTC_CUDART_STATIC=1 links the CUDA runtime statically, so the .exe also starts on
    // machines without the CUDA runtime DLL (AMD / Intel users).
    if env::var_os("QBTC_CUDART_STATIC").is_some() {
        println!("cargo:rustc-link-lib=static=cudart_static");
        if !windows {
            println!("cargo:rustc-link-lib=dylib=rt");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
    println!("cargo:warning=CUDA kernel compiled; GPU mining available.");
    Ok(())
}

// Stub providing all CUDA symbols declared in src/gpu_miner.rs, so the crate
// links and runs on machines without a CUDA toolkit.
fn build_cuda_cpu_stub(out_dir: &str) {
    const STUB: &str = r#"#include <stdint.h>

int qbtc_cuda_mine_on(int dev, const uint8_t* h120, uint64_t tu64,
                      uint64_t start, uint64_t bsz, uint64_t* out,
                      volatile int* sf) {
    (void)dev; (void)h120; (void)tu64; (void)start;
    (void)bsz; (void)out; (void)sf;
    return -1;
}

int qbtc_cuda_device_count(int* n) {
    if (n) *n = 0;
    return -1;
}

int qbtc_cuda_init_device(int dev) {
    (void)dev;
    return -1;
}

const char* qbtc_cuda_device_name(int dev) {
    (void)dev;
    return "CUDA_NOT_AVAILABLE";
}

void qbtc_cuda_shutdown(void) {}
"#;
    let stub = PathBuf::from(out_dir).join("qbtc_cuda_stub.c");
    fs::write(&stub, STUB).expect("could not write the CUDA stub");
    cc::Build::new().file(&stub).compile("qbtc_cuda_kernel");
}

// =====================================================================
// OpenCL
// =====================================================================

// Locations where CL/cl.h might live. CUDA toolkit ships it, so if CUDA is
// present we usually find it there. Otherwise try vendor SDKs and system paths.
//
// Override: set OPENCL_INCLUDE to a directory that contains CL/cl.h (for example the
// Khronos OpenCL-Headers checkout or the vcpkg include dir) to build without the CUDA toolkit.
fn find_opencl_include() -> Option<PathBuf> {
    if let Ok(p) = env::var("OPENCL_INCLUDE") {
        let p = PathBuf::from(p);
        if p.join("CL").join("cl.h").exists() {
            return Some(p);
        }
    }

    // 1. CUDA toolkit ships CL/cl.h under <cuda>/include
    if let Some(cuda_path) = find_cuda_path() {
        let inc = Path::new(&cuda_path).join("include");
        if inc.join("CL").join("cl.h").exists() {
            return Some(inc);
        }
    }

    if is_windows() {
        // 2. Vendor SDKs
        for base in [
            r"C:\Program Files (x86)\Intel\OpenCL SDK\include",
            r"C:\Program Files\Intel\OpenCL SDK\include",
            r"C:\Program Files (x86)\AMD APP SDK\include",
            r"C:\Program Files\AMD APP SDK\include",
        ] {
            let p = Path::new(base);
            if p.join("CL").join("cl.h").exists() {
                return Some(p.to_path_buf());
            }
        }
    } else {
        // 3. Linux / BSD system includes
        for base in ["/usr/include", "/usr/local/include", "/opt/rocm/include"] {
            let p = Path::new(base);
            if p.join("CL").join("cl.h").exists() {
                return Some(p.to_path_buf());
            }
        }
    }

    None
}

// Windows: CUDA toolkit ships a redirect stub OpenCL.lib under <cuda>/lib/x64
// which loads the real driver OpenCL.dll at runtime. Linux finds libOpenCL.so
// through the standard linker search path.
//
// Override: set OPENCL_LIB to the directory that contains OpenCL.lib (Windows) or
// libOpenCL.so (Linux) when building without the CUDA toolkit.
fn find_opencl_lib_dir() -> Option<PathBuf> {
    if let Ok(p) = env::var("OPENCL_LIB") {
        return Some(PathBuf::from(p));
    }
    if let Some(cuda_path) = find_cuda_path() {
        let lib = Path::new(&cuda_path).join("lib").join("x64");
        if lib.join("OpenCL.lib").exists() {
            return Some(lib);
        }
    }
    None
}

fn build_opencl_kernel(out_dir: &str, include_dir: &Path) -> Result<(), String> {
    let mut build = cc::Build::new();
    build.file("cuda/qbtc_sha256_miner_opencl.c");
    build.include(out_dir);      // qbtc_cl_source.h generated above
    build.include(include_dir);  // CL/cl.h
    build.opt_level(3);

    build.compile("qbtc_cl_host");

    if let Some(lib_dir) = find_opencl_lib_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=OpenCL");

    println!("cargo:warning=OpenCL host compiled; OpenCL mining available.");
    Ok(())
}

// Emits .cl source into OUT_DIR/qbtc_cl_source.h as a static C string.
fn generate_cl_header(out_dir: &str) {
    let cl_src = match fs::read_to_string("cuda/qbtc_sha256_miner.cl") {
        Ok(s) => s,
        Err(_) => return,
    };
    let hdr_path = PathBuf::from(out_dir).join("qbtc_cl_source.h");
    let mut out = String::with_capacity(cl_src.len() * 2 + 64);
    out.push_str("#pragma once\nstatic const char* QBTC_CL_SOURCE =\n\"");
    for ch in cl_src.chars() {
        match ch {
            '\r' => {}
            '\\' => out.push_str("\\\\"),
            '"'  => out.push_str("\\\""),
            '\n' => out.push_str("\\n\"\n\""),
            _    => out.push(ch),
        }
    }
    out.push_str("\";\n");
    fs::write(&hdr_path, out).expect("could not write qbtc_cl_source.h");
}

// Stub providing all OpenCL symbols declared in src/gpu_miner.rs.
fn build_opencl_cpu_stub(out_dir: &str) {
    const STUB: &str = r#"#include <stdint.h>

int qbtc_cl_mine_on(int dev, const uint8_t* h120, uint64_t t,
                    uint64_t s, uint64_t b, uint64_t* out,
                    volatile int* sf) {
    (void)dev; (void)h120; (void)t; (void)s; (void)b; (void)out; (void)sf;
    return -1;
}
int qbtc_cl_device_count(int* n) { if (n) *n = 0; return -1; }
int qbtc_cl_init_device(int dev) { (void)dev; return -1; }
const char* qbtc_cl_device_name(int dev) { (void)dev; return "OPENCL_NOT_AVAILABLE"; }
void qbtc_cl_shutdown(void) {}
"#;
    let stub = PathBuf::from(out_dir).join("qbtc_cl_stub.c");
    fs::write(&stub, STUB).expect("could not write OpenCL stub");
    cc::Build::new().file(&stub).compile("qbtc_cl_host");
}