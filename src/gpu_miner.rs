// src/gpu_miner.rs
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::Ordering;
use crate::block::{Block, BlockHeader};
use crate::crypto::merkle;
use crate::transaction::Transaction;

macro_rules! gpu_log {
    ($($arg:tt)*) => {
        if !crate::miner::MUTE_CONSOLE_LOGS.load(Ordering::Relaxed) {
            tracing::info!($($arg)*);
        }
    };
}
macro_rules! gpu_warn {
    ($($arg:tt)*) => {
        if !crate::miner::MUTE_CONSOLE_LOGS.load(Ordering::Relaxed) {
            tracing::warn!($($arg)*);
        }
    };
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GpuBackend { Cuda, OpenCl }

// GPU API selection (first match wins):
//   QBTC_GPU_BACKEND=opencl|cl     explicit API choice
//   QBTC_MINER_BACKEND=opencl|cl   shorthand, also switches the miner to GPU mode
//   otherwise CUDA (and OpenCL as an automatic fallback, see gpu_init_auto)
fn api_is_explicit() -> bool {
    let api = std::env::var("QBTC_GPU_BACKEND").unwrap_or_default().to_lowercase();
    let miner = std::env::var("QBTC_MINER_BACKEND").unwrap_or_default().to_lowercase();
    matches!(api.as_str(), "cuda" | "opencl" | "cl") || matches!(miner.as_str(), "opencl" | "cl")
}

pub fn selected_backend() -> GpuBackend {
    let api = std::env::var("QBTC_GPU_BACKEND").unwrap_or_default().to_lowercase();
    let miner = std::env::var("QBTC_MINER_BACKEND").unwrap_or_default().to_lowercase();
    match (api.as_str(), miner.as_str()) {
        ("opencl", _) | ("cl", _) => GpuBackend::OpenCl,
        ("cuda", _) => GpuBackend::Cuda,
        (_, "opencl") | (_, "cl") => GpuBackend::OpenCl,
        _ => GpuBackend::Cuda,
    }
}

extern "C" {
    // CUDA
    fn qbtc_cuda_mine_on(device_id: i32, header_120: *const u8, target_u64: u64,
                         start_nonce: u64, batch_size: u64,
                         found_nonce_out: *mut u64, stop_flag_ptr: *mut i32) -> i32;
    fn qbtc_cuda_device_count(n: *mut i32) -> i32;
    fn qbtc_cuda_init_device(dev: i32) -> i32;
    fn qbtc_cuda_device_name(dev: i32) -> *const std::os::raw::c_char;
    fn qbtc_cuda_shutdown();

    // OpenCL
    fn qbtc_cl_mine_on(device_id: i32, header_120: *const u8, target_u64: u64,
                       start_nonce: u64, batch_size: u64,
                       found_nonce_out: *mut u64, stop_flag_ptr: *mut i32) -> i32;
    fn qbtc_cl_device_count(n: *mut i32) -> i32;
    fn qbtc_cl_init_device(dev: i32) -> i32;
    fn qbtc_cl_device_name(dev: i32) -> *const std::os::raw::c_char;
    fn qbtc_cl_shutdown();
}

fn be_count(be: GpuBackend, n: *mut i32) -> i32 {
    unsafe { match be {
        GpuBackend::Cuda   => qbtc_cuda_device_count(n),
        GpuBackend::OpenCl => qbtc_cl_device_count(n),
    } }
}
fn be_init(be: GpuBackend, dev: i32) -> i32 {
    unsafe { match be {
        GpuBackend::Cuda   => qbtc_cuda_init_device(dev),
        GpuBackend::OpenCl => qbtc_cl_init_device(dev),
    } }
}
fn be_name(be: GpuBackend, dev: i32) -> String {
    unsafe {
        let p = match be {
            GpuBackend::Cuda   => qbtc_cuda_device_name(dev),
            GpuBackend::OpenCl => qbtc_cl_device_name(dev),
        };
        if p.is_null() { "Unknown".into() }
        else { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
    }
}
pub fn be_shutdown(be: GpuBackend) {
    unsafe { match be {
        GpuBackend::Cuda   => qbtc_cuda_shutdown(),
        GpuBackend::OpenCl => qbtc_cl_shutdown(),
    } }
}
fn be_mine(be: GpuBackend, dev: i32, hdr: *const u8, target: u64,
           start: u64, batch: u64, found: *mut u64, stop: *mut i32) -> i32 {
    unsafe { match be {
        GpuBackend::Cuda   => qbtc_cuda_mine_on(dev, hdr, target, start, batch, found, stop),
        GpuBackend::OpenCl => qbtc_cl_mine_on  (dev, hdr, target, start, batch, found, stop),
    } }
}

// Random per-process nonce base so several miners do not scan the same range.
static NONCE_BASE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
pub fn process_nonce_base() -> u64 {
    *NONCE_BASE.get_or_init(|| {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64);
        h.write_u32(std::process::id());
        h.finish() & 0x00FF_FFFF_FFFF_FFFF
    })
}

pub struct GpuStopFlag { inner: Box<i32> }
impl GpuStopFlag {
    pub fn new() -> Self { Self { inner: Box::new(0) } }
    pub fn as_ptr(&mut self) -> *mut i32 { self.inner.as_mut() as *mut i32 }
    pub fn stop(&mut self) { *self.inner = 1; }
    pub fn is_stopped(&self) -> bool { *self.inner != 0 }
}

// Chooses the GPU API and brings its devices up. When the API was not chosen
// explicitly and CUDA finds no usable device (AMD / Intel machine, no NVIDIA
// driver), OpenCL is tried before giving up, so users need no extra switches.
pub fn gpu_init_auto() -> (GpuBackend, Vec<i32>) {
    let be = selected_backend();
    let devices = gpu_init_all(be);
    if !devices.is_empty() || be != GpuBackend::Cuda || api_is_explicit() {
        return (be, devices);
    }
    gpu_log!("[INFO] Miner: no CUDA device found, trying OpenCL...");
    let cl = gpu_init_all(GpuBackend::OpenCl);
    if cl.is_empty() { (be, devices) } else { (GpuBackend::OpenCl, cl) }
}

pub fn gpu_init_all(backend: GpuBackend) -> Vec<i32> {
    let mut count: i32 = 0;
    if be_count(backend, &mut count) != 0 || count <= 0 {
        gpu_warn!("[WARN] Miner: {:?} device enumeration failed (n={})", backend, count);
        return Vec::new();
    }
    let wanted: Option<Vec<i32>> = std::env::var("QBTC_GPU_DEVICES").or_else(|_| std::env::var("QBTC_CUDA_DEVICES")).ok().map(|raw| {
        raw.split(',').filter_map(|p| p.trim().parse::<i32>().ok())
            .filter(|d| *d >= 0 && *d < count).collect()
    });
    let candidates = match wanted {
        Some(l) if !l.is_empty() => l,
        _ => (0..count).collect(),
    };
    let mut ready = Vec::new();
    for dev in candidates {
        if be_init(backend, dev) == 0 {
            gpu_log!("[INFO] Miner: {:?} device {} ready: {}", backend, dev, be_name(backend, dev));
            ready.push(dev);
        } else {
            gpu_warn!("[WARN] Miner: {:?} device {} init failed", backend, dev);
        }
    }
    if ready.is_empty() {
        gpu_warn!("[WARN] Miner: no {:?} device initialized", backend);
    } else {
        gpu_log!("[INFO] Miner: mining on {} {:?} device(s): {:?}", ready.len(), backend, ready);
    }
    ready
}

pub fn build_header_120(timestamp: u64, previous_hash: &[u8; 32], merkle_root: &[u8; 32],
                        commit_merkle_root: &[u8; 32], nonce: u64, target: u64) -> [u8; 120] {
    let mut buf = [0u8; 120];
    buf[0..8].copy_from_slice(&timestamp.to_be_bytes());
    buf[8..40].copy_from_slice(previous_hash);
    buf[40..72].copy_from_slice(merkle_root);
    buf[72..104].copy_from_slice(commit_merkle_root);
    buf[104..112].copy_from_slice(&nonce.to_be_bytes());
    buf[112..120].copy_from_slice(&target.to_be_bytes());
    buf
}

#[derive(Clone)]
pub struct GpuBlockData {
    pub previous_hash: [u8; 32],
    pub merkle_root: [u8; 32],
    pub commit_merkle_root: [u8; 32],
    pub target: u64,
    pub compare_target: u64,
    pub fixed_timestamp: Option<u64>,
    pub pool_mode: bool,
    pub nonce_base: u64,
    pub nonce_min: Option<u64>,
    pub nonce_max: Option<u64>,
    pub transactions: Vec<Transaction>,
}
impl GpuBlockData {
    pub fn from_template(previous_hash: [u8; 32], transactions: Vec<Transaction>, target: u64) -> Self {
        let tx_ids: Vec<[u8; 32]> = transactions.iter().map(|tx| tx.calculate_id()).collect();
        let witness_hashes: Vec<[u8; 32]> = transactions.iter().map(|tx| tx.calculate_witness_hash()).collect();
        let merkle_root = merkle::build_merkle_root(tx_ids);
        let commit_merkle_root = merkle::build_merkle_root(witness_hashes);
        Self {
            previous_hash, merkle_root, commit_merkle_root, target,
            compare_target: target,
            fixed_timestamp: None,
            pool_mode: false,
            nonce_base: process_nonce_base(),
            nonce_min: None,
            nonce_max: None,
            transactions,
        }
    }
    pub fn from_pool_job(previous_hash: [u8; 32], job: &crate::miner::PoolJob) -> Self {
        Self {
            previous_hash,
            merkle_root: job.merkle_root,
            commit_merkle_root: job.commit_merkle_root,
            target: job.header_target,
            compare_target: job.share_target,
            fixed_timestamp: Some(job.timestamp),
            pool_mode: true,
            nonce_base: job.nonce_min.unwrap_or_else(process_nonce_base),
            nonce_min: job.nonce_min,
            nonce_max: job.nonce_max,
            transactions: Vec::new(),
        }
    }
    pub fn to_block_with_nonce(&self, nonce: u64, timestamp: u64) -> Block {
        Block {
            header: BlockHeader { timestamp, previous_hash: self.previous_hash, merkle_root: self.merkle_root, commit_merkle_root: self.commit_merkle_root, nonce, target: self.target },
            transactions: self.transactions.clone(),
        }
    }
}

pub enum GpuScanResult {
    Found(u64),
    Exhausted,
    Cancelled,
    Failed(i32),
}

fn describe_gpu_failure(code: i32) -> &'static str {
    match code {
        -1 => "kernel launch or sync failed",
        -2 => "H2D memcpy of params failed",
        -3 => "H2D memcpy of found-sentinel failed",
        -4 => "H2D memcpy of stop-flag failed",
        -5 => "D2H memcpy of result failed",
        -6 => "invalid device id",
        -7 => "device not initialized / cudaSetDevice failed",
        _ => "unknown failure",
    }
}

pub fn gpu_scan_on(backend: GpuBackend, device_id: i32, header: &[u8; 120],
                   target: u64, start_nonce: u64, batch_size: u64,
                   stop: &mut GpuStopFlag) -> GpuScanResult {
    if stop.is_stopped() { return GpuScanResult::Cancelled; }
    let mut found: u64 = 0;
    let ret = be_mine(backend, device_id, header.as_ptr(), target, start_nonce, batch_size,
                      &mut found, stop.as_ptr());
    match ret {
        0 => { gpu_log!("[INFO] Miner: {:?} found nonce: {}", backend, found); GpuScanResult::Found(found) }
        1 => GpuScanResult::Exhausted,
        code => {
            gpu_warn!("[WARN] Miner: {:?} batch failed (code={}, {})", backend, code, describe_gpu_failure(code));
            GpuScanResult::Failed(code)
        }
    }
}

pub fn run_gpu_mine_loop_on(
    backend: GpuBackend,
    device_id: i32,
    slot: u64,
    device_count: u64,
    gpu_data: GpuBlockData,
    result_tx: crossbeam_channel::Sender<Block>,
    stop_rx: crossbeam_channel::Receiver<()>,
) {
    let fixed_timestamp = match gpu_data.fixed_timestamp {
        Some(t) => t,
        None => SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
    };
    let header = build_header_120(fixed_timestamp, &gpu_data.previous_hash, &gpu_data.merkle_root, &gpu_data.commit_merkle_root, 0, gpu_data.target);
    let batch_size: u64 = std::env::var("QBTC_CUDA_BATCH_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000_000u64);
    let mut stop = GpuStopFlag::new();
    let devices = device_count.max(1);
    let stride = batch_size.saturating_mul(devices);
    let mut start_nonce: u64 = gpu_data.nonce_base.wrapping_add(batch_size.saturating_mul(slot % devices));
    let mut hash_count: u64 = 0;
    let mining_started = std::time::Instant::now();
    let mut last_progress_print = mining_started;
    let mut hashes_since_print: u64 = 0;
    const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
    gpu_log!(
        "[INFO] Miner: {:?} GPU{} scan started (batch_size={}, start_nonce={}, stride={}, timestamp={})",
        backend, device_id, batch_size, start_nonce, stride, fixed_timestamp
    );
    if slot == 0 && std::env::var_os("QBTC_CUDA_DEBUG").is_some() {
        print_reference_hash_for_debug(&header);
    }
    let mut consecutive_failures: u32 = 0;
    let verbose_progress = std::env::var_os("QBTC_CUDA_DEBUG").is_some();
    loop {
        if stop_rx.try_recv().is_ok() { stop.stop(); break; }
        let batch_started = std::time::Instant::now();
        if let Some(max) = gpu_data.nonce_max {
            if start_nonce > max {
                start_nonce = gpu_data.nonce_min.unwrap_or(0);
            }
        }
        let scan_result = gpu_scan_on(backend, device_id, &header, gpu_data.compare_target, start_nonce, batch_size, &mut stop);
        let batch_elapsed = batch_started.elapsed();
        match scan_result {
            GpuScanResult::Found(winning_nonce) => {
                consecutive_failures = 0;
                let candidate = gpu_data.to_block_with_nonce(winning_nonce, fixed_timestamp);
                let verified = if gpu_data.pool_mode {
                    crate::miner::pool_share_valid(&candidate, gpu_data.compare_target)
                } else {
                    crate::consensus::ConsensusEngine::verify_proof_of_work(&candidate, gpu_data.target)
                };
                if verified && gpu_data.pool_mode {
                    gpu_log!("[INFO] Miner [GPU{}]: share found nonce={}", device_id, winning_nonce);
                    let _ = result_tx.send(candidate);
                    start_nonce = start_nonce.wrapping_add(stride);
                    continue;
                }
                if verified {
                    gpu_log!("[INFO] Miner [GPU{}]: Proof-of-Work puzzle solved nonce={}", device_id, winning_nonce);
                    let hash = candidate.calculate_hash();
                    let hash_hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
                    gpu_log!("[INFO] Miner: Absolute Block Hash: 0x{}", hash_hex);
                    let _ = result_tx.send(candidate);
                    loop { if stop_rx.try_recv().is_ok() { return; } std::thread::sleep(std::time::Duration::from_millis(100)); }
                } else {
                    let dbg_hash = candidate.calculate_hash();
                    let dbg_prefix = u64::from_be_bytes(dbg_hash[0..8].try_into().unwrap());
                    gpu_warn!(
                        "[WARN] Miner: GPU{} reported a nonce that failed CPU re-verification \
                         (nonce={} hash_prefix={} target={} ts={}); discarding and continuing",
                        device_id, winning_nonce, dbg_prefix, gpu_data.target, fixed_timestamp
                    );
                    start_nonce = start_nonce.wrapping_add(stride);
                    continue;
                }
            }
            GpuScanResult::Exhausted => {
                consecutive_failures = 0;
                hash_count += batch_size;
                hashes_since_print += batch_size;
                let since_print = last_progress_print.elapsed();
                if since_print >= PROGRESS_INTERVAL {
                    let mh_s = (hashes_since_print as f64 / since_print.as_secs_f64()) / 1_000_000.0;
                    let total_elapsed = mining_started.elapsed().as_secs_f64();
                    crate::stats::STATS.gpu_rate(device_id, mh_s);
                    gpu_log!(
                        "[MINING] GPU{} progress: {} hashes total | {:.1} MH/s | elapsed {:.0}s | nonce={} | last_batch={:.3}s",
                        device_id, hash_count, mh_s, total_elapsed, start_nonce, batch_elapsed.as_secs_f64()
                    );
                    last_progress_print = std::time::Instant::now();
                    hashes_since_print = 0;
                }
                if verbose_progress && hash_count % 500_000_000 < batch_size {
                    gpu_log!("[INFO] Miner: GPU{} {} million hashes calculated.", device_id, hash_count / 1_000_000);
                }
                start_nonce = start_nonce.wrapping_add(stride);
                if start_nonce == 0 { gpu_warn!("[WARN] Miner: GPU{} exhausted the entire u64 nonce space.", device_id); break; }
                if stop.is_stopped() || stop_rx.try_recv().is_ok() { break; }
            }
            GpuScanResult::Cancelled => break,
            GpuScanResult::Failed(code) => {
                consecutive_failures += 1;
                if consecutive_failures >= 5 {
                    tracing::error!("[ERROR] Miner: GPU{} has failed {} batches in a row (last code={}). Check GPU/driver health.", device_id, consecutive_failures, code);
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
                if stop.is_stopped() || stop_rx.try_recv().is_ok() { break; }
            }
        }
    }
}

pub fn print_reference_hash_for_debug(header: &[u8; 120]) {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(header);
    let rust_hash: [u8; 32] = hasher.finalize().into();
    let rust_prefix = u64::from_be_bytes(rust_hash[0..8].try_into().unwrap());
    gpu_log!("[DEBUG] Miner: reference hash prefix={}", rust_prefix);
    gpu_log!("[DEBUG] Miner: header_hex={}", header.iter().map(|b| format!("{:02x}",b)).collect::<String>());
}

pub fn run_bench(seconds: u64) {
    let (be, devices) = gpu_init_auto();
    if devices.is_empty() {
        println!("[BENCH] No usable {:?} device.", be);
        return;
    }
    let seconds = seconds.max(1);
    let batch_size: u64 = std::env::var("QBTC_CUDA_BATCH_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000_000u64);
    let header = build_header_120(1_700_000_000, &[0x11u8; 32], &[0x22u8; 32], &[0x33u8; 32], 0, 0);
    let n = devices.len() as u64;
    println!("[BENCH] {} {:?} device(s), batch_size={}, 2 s warm-up + {} s measured", devices.len(), be, batch_size, seconds);

    let mut handles = Vec::new();
    for (slot, dev) in devices.iter().copied().enumerate() {
        handles.push(std::thread::spawn(move || {
            let mut stop = GpuStopFlag::new();
            let stride = batch_size.saturating_mul(n);
            let mut start = batch_size.saturating_mul(slot as u64);
            let warm = std::time::Instant::now();
            while warm.elapsed() < std::time::Duration::from_secs(2) {
                match gpu_scan_on(be, dev, &header, 0, start, batch_size, &mut stop) {
                    GpuScanResult::Exhausted => {}
                    _ => return (dev, 0u64, 1.0f64),
                }
                start = start.wrapping_add(stride);
            }
            let begin = std::time::Instant::now();
            let mut hashes = 0u64;
            while begin.elapsed() < std::time::Duration::from_secs(seconds) {
                match gpu_scan_on(be, dev, &header, 0, start, batch_size, &mut stop) {
                    GpuScanResult::Exhausted => hashes += batch_size,
                    _ => break,
                }
                start = start.wrapping_add(stride);
            }
            (dev, hashes, begin.elapsed().as_secs_f64().max(0.001))
        }));
    }

    let mut total = 0.0f64;
    for h in handles {
        if let Ok((dev, hashes, secs)) = h.join() {
            let mh = hashes as f64 / secs / 1_000_000.0;
            total += mh;
            println!("[BENCH] GPU{} {:.1} MH/s", dev, mh);
        }
    }
    println!("[BENCH] TOTAL {:.1} MH/s", total);
    be_shutdown(be);
}