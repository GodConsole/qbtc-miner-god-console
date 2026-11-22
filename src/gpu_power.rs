// src/gpu_power.rs
//
// Safe GPU power ceiling:  `--max-power <watts>`  (or env QBTC_MAX_POWER).
//
// Works per GPU, identified by its exact PCI address (taken from the CUDA /
// OpenCL driver itself, so the limit can never land on the wrong card), and
// does not depend on the card or driver being "official": every device is
// probed with every available method and the best working one is used.
//
//  1. HARDWARE CAP (only when the driver accepts and confirms it):
//       NVIDIA   NVML, falling back to `nvidia-smi -pl`
//       Linux    hwmon power1_cap / power1_max (AMD, Intel, anything that exposes it)
//     The cap is only ever LOWERED, is verified by read-back, and the original
//     value is restored by `shutdown()`.
//
//  2. SOFTWARE GOVERNOR (works without admin rights, and on cards/drivers that
//     refuse a hardware cap: mobile chips on desktop boards, modified drivers,
//     Windows AMD). The real power sensor is read every 250 ms and the mining
//     loop of that GPU sleeps between batches (duty cycle) so the average draw
//     stays at the limit. >15% over the limit pauses the GPU completely.
//     Sensors: NVML, nvidia-smi, hwmon (power or energy counter), AMD ADL.
//
//  3. A device with neither a verified cap nor a working sensor does NOT mine
//     (the limit could not be honoured) unless you opt in to the unverified
//     fixed duty cycle `--power-blind-duty <percent>`.
//
// The limit is a ceiling on AVERAGE power as the sensor reports it (NVIDIA:
// board power; AMD Linux: hwmon; AMD Windows: ASIC power).

use crate::gpu_miner::GpuBackend;
use libloading::Library;
use std::ffi::{c_char, c_void, CString};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
#[cfg(target_os = "linux")]
use std::path::{Path, PathBuf};

const SAMPLE_MS: u64 = 250;
const SENSOR_LOST_SAMPLES: u32 = 8; // 2 s without a valid reading
const MIN_DUTY: f64 = 0.05;
const STALE_MS: u64 = 5_000; // sampler silent for 5 s => pause
const HW_BACKSTOP: f64 = 1.10; // governor ceiling when a hardware cap is active
const TRIP: f64 = 1.15; // instant pause above limit * TRIP

static LIMIT_W: OnceLock<f64> = OnceLock::new();
static BLIND_DUTY: OnceLock<f64> = OnceLock::new();

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Set the ceiling. Call once at start-up, before any GPU is initialised.
pub fn configure(watts: f64) -> Result<(), String> {
    if !watts.is_finite() || !(5.0..=2000.0).contains(&watts) {
        return Err(format!("--max-power must be between 5 and 2000 watts (got {watts})"));
    }
    LIMIT_W.set(watts).map_err(|_| "--max-power was already configured".to_string())
}

/// Opt-in last resort for cards with no usable sensor and no hardware cap:
/// a fixed duty cycle (percent of full speed). It is NOT verified against real
/// power draw.
pub fn configure_blind_duty(percent: u32) -> Result<(), String> {
    if !(5..=100).contains(&percent) {
        return Err("--power-blind-duty must be between 5 and 100 (percent)".into());
    }
    BLIND_DUTY.set(percent as f64 / 100.0).map_err(|_| "--power-blind-duty was already configured".to_string())
}

pub fn configured() -> Option<f64> {
    LIMIT_W.get().copied()
}

/// (device index, device name, PCI address reported by the driver)
pub type DeviceRef = (i32, String, Option<String>);

/// Called by `gpu_init_all` with every device that initialised. Returns the
/// devices that may mine under the configured limit (all of them when no limit
/// is configured).
pub fn enforce(backend: GpuBackend, devices: Vec<DeviceRef>) -> Vec<i32> {
    let Some(&want) = LIMIT_W.get() else {
        return devices.into_iter().map(|(d, _, _)| d).collect();
    };
    let is_cuda = backend == GpuBackend::Cuda;
    let blind = BLIND_DUTY.get().copied();
    log("info", format!("max power {:.0} W requested for {} GPU(s)", want, devices.len()));

    let mut kept = Vec::new();
    let mut mons: Vec<Monitored> = Vec::new();
    for (dev, name, pci) in devices.clone() {
        if state().throttles.iter().any(|(k, _)| *k == (is_cuda, dev)) {
            kept.push(dev); // already vetted by an earlier call
            continue;
        }
        let unique_name = devices.iter().filter(|d| d.1 == name).count() == 1;
        match prepare(is_cuda, &name, pci, unique_name, want, blind) {
            Ok(p) => {
                log("ok", format!("GPU{dev} ({name}): {}", p.summary));
                let throttle = Arc::new(match p.fixed_duty {
                    Some(d) => Throttle::fixed(d),
                    None => Throttle::new(p.sensor.is_some()),
                });
                {
                    let mut st = state();
                    st.throttles.push(((is_cuda, dev), throttle.clone()));
                    if let Some(r) = p.restore {
                        st.restores.push((format!("GPU{dev}"), r));
                    }
                }
                if let Some(sensor) = p.sensor {
                    mons.push(Monitored {
                        sensor,
                        ctl: p.controller,
                        throttle,
                        label: format!("GPU{dev}"),
                        was_paused: false,
                    });
                }
                kept.push(dev);
            }
            Err(e) => log("err", format!("GPU{dev} ({name}) will NOT mine: {e}")),
        }
    }
    if !mons.is_empty() {
        let s = spawn_sampler(mons);
        state().samplers.push(s);
    }
    kept
}

/// Throttle for one device's mining loop (None when no limit is active for it).
pub fn throttle(backend: GpuBackend, dev: i32) -> Option<Arc<Throttle>> {
    let is_cuda = backend == GpuBackend::Cuda;
    state().throttles.iter().find(|(k, _)| *k == (is_cuda, dev)).map(|(_, t)| t.clone())
}

/// Stop the governor and restore the original hardware limits. Idempotent.
/// (If the process is killed instead, a lowered limit simply stays lowered,
/// which is the safe direction; it resets on reboot / driver reload.)
pub fn shutdown() {
    let (samplers, restores) = {
        let mut st = state();
        (std::mem::take(&mut st.samplers), std::mem::take(&mut st.restores))
    };
    for (flag, _) in &samplers {
        flag.store(true, Ordering::Relaxed);
    }
    for (_, h) in samplers {
        let _ = h.join();
    }
    for (label, r) in restores.iter().rev() {
        match r.apply() {
            Ok(()) => log("info", format!("{label}: original power limit restored")),
            Err(e) => log("warn", format!("{label}: could not restore power limit: {e}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Throttle: what a mining thread talks to
// ---------------------------------------------------------------------------

pub struct Throttle {
    duty_milli: AtomicU32,
    paused: AtomicBool,
    tick_ms: AtomicU64,
    watched: bool, // false => hardware cap only, nothing to wait for
    fixed: bool,   // blind fixed duty cycle: no sampler behind it
}

impl Throttle {
    fn new(watched: bool) -> Self {
        Self {
            duty_milli: AtomicU32::new(1000),
            paused: AtomicBool::new(false),
            tick_ms: AtomicU64::new(now_ms()),
            watched,
            fixed: false,
        }
    }
    fn fixed(duty: f64) -> Self {
        Self {
            duty_milli: AtomicU32::new((duty.clamp(MIN_DUTY, 1.0) * 1000.0).round() as u32),
            paused: AtomicBool::new(false),
            tick_ms: AtomicU64::new(now_ms()),
            watched: true,
            fixed: true,
        }
    }
    fn publish(&self, c: &Controller) {
        self.duty_milli.store((c.duty * 1000.0).round() as u32, Ordering::Relaxed);
        self.paused.store(c.paused, Ordering::Relaxed);
        self.tick_ms.store(now_ms(), Ordering::Relaxed);
    }
    fn blocked(&self) -> bool {
        if !self.watched || self.fixed {
            return false;
        }
        self.paused.load(Ordering::Relaxed)
            || now_ms().saturating_sub(self.tick_ms.load(Ordering::Relaxed)) > STALE_MS
    }

    /// Call at the top of every batch iteration. Sleeps for the duty-cycle gap
    /// (proportional to the previous batch time) and blocks while the GPU is
    /// paused. Returns false if a stop request arrived (caller must stop).
    pub fn wait(&self, stop: &crossbeam_channel::Receiver<()>, last_batch: Duration) -> bool {
        if !self.watched {
            return true;
        }
        let duty = (self.duty_milli.load(Ordering::Relaxed) as f64 / 1000.0).clamp(MIN_DUTY, 1.0);
        if duty < 1.0 && !last_batch.is_zero() {
            let nap = last_batch.mul_f64(1.0 / duty - 1.0).min(Duration::from_secs(30));
            if !sleep_or_stop(stop, nap) {
                return false;
            }
        }
        while self.blocked() {
            if !sleep_or_stop(stop, Duration::from_millis(100)) {
                return false;
            }
        }
        true
    }
}

fn sleep_or_stop(stop: &crossbeam_channel::Receiver<()>, d: Duration) -> bool {
    matches!(stop.recv_timeout(d), Err(crossbeam_channel::RecvTimeoutError::Timeout))
}

// ---------------------------------------------------------------------------
// Controller: pure logic, unit-tested
// ---------------------------------------------------------------------------

pub struct Controller {
    limit: f64,
    pub duty: f64,
    pub paused: bool,
    missing: u32,
    ema: Option<f64>,
    fail_closed: bool, // pause when the sensor is lost (no hardware cap behind us)
}

impl Controller {
    pub fn new(limit: f64, fail_closed: bool) -> Self {
        Self { limit, duty: 1.0, paused: false, missing: 0, ema: None, fail_closed }
    }

    pub fn update(&mut self, reading: Option<f64>) {
        // A reading of 0 W is treated as "no reading": a stuck sensor must not
        // look like a cool card.
        let p = match reading {
            Some(p) if p.is_finite() && p > 0.0 => p,
            _ => {
                self.missing = self.missing.saturating_add(1);
                if self.fail_closed && self.missing >= SENSOR_LOST_SAMPLES {
                    self.paused = true;
                }
                return;
            }
        };
        self.missing = 0;
        let ema = match self.ema {
            Some(e) => e + 0.1 * (p - e), // ~2.5 s time constant
            None => p,
        };
        self.ema = Some(ema);
        if p > self.limit * TRIP {
            self.paused = true;
        } else if self.paused && ema <= self.limit * 0.95 {
            self.paused = false;
        }
        // Idle power while paused says nothing about the load, so the duty
        // cycle only adapts while mining.
        if !self.paused {
            let ratio = (self.limit * 0.97 / ema).clamp(0.5, 1.5);
            self.duty = (self.duty * ratio.powf(0.25)).clamp(MIN_DUTY, 1.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Sampler thread
// ---------------------------------------------------------------------------

struct Monitored {
    sensor: Sensor,
    ctl: Controller,
    throttle: Arc<Throttle>,
    label: String,
    was_paused: bool,
}

fn spawn_sampler(mut mons: Vec<Monitored>) -> (Arc<AtomicBool>, JoinHandle<()>) {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let handle = std::thread::Builder::new()
        .name("gpu-power".into())
        .spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                for m in mons.iter_mut() {
                    let w = m.sensor.watts();
                    m.ctl.update(w);
                    m.throttle.publish(&m.ctl);
                    if m.ctl.paused != m.was_paused {
                        m.was_paused = m.ctl.paused;
                        if m.ctl.paused {
                            log("warn", format!("{}: paused to stay under the power limit (sensor {:?} W)", m.label, w.map(|v| v.round())));
                        } else {
                            log("info", format!("{}: power back under the limit, resuming", m.label));
                        }
                    }
                }
                let mut waited = 0;
                while waited < SAMPLE_MS && !flag.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(50));
                    waited += 50;
                }
            }
        })
        .expect("failed to start gpu-power thread");
    (stop, handle)
}

// ---------------------------------------------------------------------------
// Global state, logging helpers
// ---------------------------------------------------------------------------

struct State {
    throttles: Vec<((bool, i32), Arc<Throttle>)>,
    restores: Vec<(String, Restore)>,
    samplers: Vec<(Arc<AtomicBool>, JoinHandle<()>)>,
}
static STATE: Mutex<State> = Mutex::new(State { throttles: Vec::new(), restores: Vec::new(), samplers: Vec::new() });
fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

fn log(level: &'static str, msg: String) {
    if !crate::miner::MUTE_CONSOLE_LOGS.load(Ordering::Relaxed) {
        println!("[{}] GPU power: {}", level.to_uppercase(), msg);
    }
    crate::stats::STATS.event(level, msg);
}

// ---------------------------------------------------------------------------
// Identity helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Vendor {
    Nvidia,
    Other,
}

#[derive(Clone, Debug)]
struct Identity {
    name: String,
    pci: String, // 0000:01:00.0
}

// "00000000:01:00.0" / "0000:01:00.0" -> "0000:01:00.0" (lowercase)
fn normalize_pci(s: &str) -> String {
    let s = s.trim().to_lowercase();
    if s.len() == 16 && s.as_bytes()[8] == b':' { s[4..].to_string() } else { s }
}

// nvidia-smi / NVML like the 8-digit-domain form
fn pci8(pci: &str) -> String {
    if pci.len() == 12 { format!("0000{pci}") } else { pci.to_string() }
}

fn norm_name(s: &str) -> String {
    let n: String = s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
    n.trim_start_matches("nvidia").to_string()
}

fn names_equal(a: &str, b: &str) -> bool {
    let (a, b) = (norm_name(a), norm_name(b));
    !a.is_empty() && a == b
}

// Only a hint for which probe to try first; every device is tried with all.
fn vendor_hint(name: &str) -> Vendor {
    let s = name.to_lowercase();
    const NV: [&str; 9] = ["nvidia", "geforce", "quadro", "tesla", "rtx", "gtx", "titan", "p106", "p104"];
    if NV.iter().any(|k| s.contains(k)) { Vendor::Nvidia } else { Vendor::Other }
}

fn quiet(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    #[cfg(not(windows))]
    {
        let _ = cmd;
    }
}

// nvidia-smi with a hard timeout (a broken driver can hang it forever).
fn run_smi(args: &[&str], timeout_s: u64) -> Option<String> {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        #[allow(unused_mut)]
        let mut names = vec!["nvidia-smi".to_string()];
        #[cfg(windows)]
        names.push("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe".to_string());
        for n in names {
            let mut c = Command::new(&n);
            c.args(&args);
            quiet(&mut c);
            if let Ok(o) = c.output() {
                if o.status.success() {
                    let _ = tx.send(Some(String::from_utf8_lossy(&o.stdout).into_owned()));
                    return;
                }
            }
        }
        let _ = tx.send(None);
    });
    rx.recv_timeout(Duration::from_secs(timeout_s)).ok().flatten()
}

fn parse_smi(text: &str) -> Vec<Identity> {
    text.lines()
        .filter_map(|line| {
            let mut p = line.splitn(3, ',');
            let _index = p.next()?.trim().parse::<usize>().ok()?;
            let raw = p.next()?.trim().to_lowercase();
            let name = p.next()?.trim().to_string();
            if raw.is_empty() || name.is_empty() {
                return None;
            }
            Some(Identity { name, pci: normalize_pci(&raw) })
        })
        .collect()
}

fn smi_identities() -> Vec<Identity> {
    run_smi(&["--query-gpu=index,pci.bus_id,name", "--format=csv,noheader,nounits"], 8)
        .map(|t| parse_smi(&t))
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Per-device preparation
// ---------------------------------------------------------------------------

struct Prepared {
    sensor: Option<Sensor>,
    controller: Controller,
    restore: Option<Restore>,
    fixed_duty: Option<f64>,
    summary: String,
}

struct CapResult {
    enforced: bool, // the card now runs at or below the requested ceiling
    restore: Option<Restore>,
    note: String,
}

struct Probe {
    sensor: Option<Sensor>,
    cap: Result<CapResult, String>,
}

fn prepare(is_cuda: bool, name: &str, pci: Option<String>, unique_name: bool, want: f64, blind: Option<f64>) -> Result<Prepared, String> {
    let hint = if is_cuda { Vendor::Nvidia } else { vendor_hint(name) };
    let pci = match pci.map(|p| normalize_pci(&p)).filter(|p| !p.is_empty()) {
        Some(p) => p,
        None => {
            // The driver gave no PCI address. For NVIDIA a single unambiguous
            // name match against nvidia-smi is still safe.
            if !(unique_name && hint == Vendor::Nvidia) {
                return Err("the driver does not report this GPU's PCI address".into());
            }
            let m: Vec<Identity> = smi_identities().into_iter().filter(|i| names_equal(&i.name, name)).collect();
            if m.len() == 1 { m[0].pci.clone() } else { return Err("cannot identify this GPU's PCI address (driver and nvidia-smi both unclear)".into()) }
        }
    };

    // Try every method; the likely vendor first. A probe fails when the device
    // is simply not that vendor's / the interface is missing.
    let order = if hint == Vendor::Nvidia { [true, false] } else { [false, true] };
    let mut errors: Vec<String> = Vec::new();
    let mut probe: Option<Probe> = None;
    for nvidia in order {
        let r = if nvidia { probe_nvidia(&pci, want) } else { probe_system(&pci, want) };
        match r {
            Ok(p) => {
                probe = Some(p);
                break;
            }
            Err(e) => errors.push(e),
        }
    }
    let Some(probe) = probe else {
        let why = format!("no power interface found ({})", errors.join("; "));
        return match blind {
            Some(d) => Ok(blind_prepared(d, why)),
            None => Err(format!("{why}; try --power-blind-duty <percent> for an unverified fixed limit")),
        };
    };

    let sensor = probe.sensor.and_then(live);
    let (hw_ok, restore, note) = match probe.cap {
        Ok(c) => (c.enforced, c.restore, c.note),
        Err(e) => (false, None, format!("hardware cap unavailable ({e})")),
    };
    if !hw_ok && sensor.is_none() {
        if let Some(r) = &restore {
            let _ = r.apply();
        }
        return match blind {
            Some(d) => Ok(blind_prepared(d, note)),
            None => Err(format!(
                "{note}; no usable power sensor either, so {want:.0} W can be neither enforced nor verified (opt in to an unverified fixed limit with --power-blind-duty <percent>)"
            )),
        };
    }
    let limit = if hw_ok { want * HW_BACKSTOP } else { want };
    let summary = format!(
        "{}; software governor {}",
        note,
        if sensor.is_some() { "active" } else { "not needed (no sensor)" }
    );
    Ok(Prepared { sensor, controller: Controller::new(limit, !hw_ok), restore, fixed_duty: None, summary })
}

fn blind_prepared(duty: f64, why: String) -> Prepared {
    Prepared {
        sensor: None,
        controller: Controller::new(f64::MAX, false),
        restore: None,
        fixed_duty: Some(duty),
        summary: format!("{why}; running a fixed {:.0}% duty cycle (UNVERIFIED, no power measurement)", duty * 100.0),
    }
}

// A sensor must give a positive reading (energy counters need a second sample).
fn live(s: Sensor) -> Option<Sensor> {
    let mut w = s.watts();
    if w.is_none() {
        std::thread::sleep(Duration::from_millis(350));
        w = s.watts();
    }
    if w.map_or(false, |v| v > 0.0) { Some(s) } else { None }
}

// ---------------------------------------------------------------------------
// Sensors and restore actions
// ---------------------------------------------------------------------------

enum Sensor {
    Nvml(Arc<Nvml>, Dev),
    Smi(SmiSensor),
    #[cfg(target_os = "linux")]
    Hwmon(HwmonSensor),
    #[cfg(windows)]
    Adl(adl::Adl),
}

impl Sensor {
    fn watts(&self) -> Option<f64> {
        match self {
            Sensor::Nvml(nv, d) => nv_get(nv.usage, *d).map(|mw| mw as f64 / 1000.0),
            Sensor::Smi(s) => s.watts(),
            #[cfg(target_os = "linux")]
            Sensor::Hwmon(h) => h.watts(),
            #[cfg(windows)]
            Sensor::Adl(a) => a.watts(),
        }
    }
}

enum Restore {
    Nvml { nv: Arc<Nvml>, dev: Dev, orig_mw: u32 },
    Smi { pci: String, orig_w: f64 },
    #[cfg(target_os = "linux")]
    Sysfs { path: PathBuf, orig: String },
}

impl Restore {
    fn apply(&self) -> Result<(), String> {
        match self {
            Restore::Nvml { nv, dev, orig_mw } => {
                let rc = unsafe { (nv.set_limit)(dev.0, *orig_mw) };
                if rc == 0 { Ok(()) } else { Err(format!("NVML error {rc}")) }
            }
            Restore::Smi { pci, orig_w } => run_smi(&["-i", pci.as_str(), "-pl", &format!("{orig_w}")], 8)
                .map(|_| ())
                .ok_or_else(|| "nvidia-smi could not restore the limit".to_string()),
            #[cfg(target_os = "linux")]
            Restore::Sysfs { path, orig } => std::fs::write(path, orig).map_err(|e| e.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// NVIDIA: NVML (desktop and mobile chips, official or modified drivers, as long
// as the driver ships NVML), nvidia-smi as fallback
// ---------------------------------------------------------------------------

type NvHandle = *mut c_void;
#[derive(Clone, Copy)]
struct Dev(NvHandle);
unsafe impl Send for Dev {}
unsafe impl Sync for Dev {}

struct Nvml {
    _lib: Library,
    by_pci: unsafe extern "C" fn(*const c_char, *mut NvHandle) -> i32,
    usage: unsafe extern "C" fn(NvHandle, *mut u32) -> i32,
    get_limit: unsafe extern "C" fn(NvHandle, *mut u32) -> i32,
    constraints: unsafe extern "C" fn(NvHandle, *mut u32, *mut u32) -> i32,
    set_limit: unsafe extern "C" fn(NvHandle, u32) -> i32,
}
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

static NVML: OnceLock<Result<Arc<Nvml>, String>> = OnceLock::new();
fn nvml() -> Result<Arc<Nvml>, String> {
    NVML.get_or_init(|| unsafe { nvml_load() }).clone()
}

unsafe fn nvml_load() -> Result<Arc<Nvml>, String> {
    #[cfg(windows)]
    let names = ["nvml.dll", "C:\\Windows\\System32\\nvml.dll", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll"];
    #[cfg(not(windows))]
    let names = ["libnvidia-ml.so.1", "libnvidia-ml.so"];
    let lib = names.iter().find_map(|n| Library::new(n).ok()).ok_or("NVML library not found")?;
    let init = *lib.get::<unsafe extern "C" fn() -> i32>(b"nvmlInit_v2\0").map_err(|e| e.to_string())?;
    let by_pci = *lib
        .get::<unsafe extern "C" fn(*const c_char, *mut NvHandle) -> i32>(b"nvmlDeviceGetHandleByPciBusId_v2\0")
        .map_err(|e| e.to_string())?;
    let usage = *lib.get::<unsafe extern "C" fn(NvHandle, *mut u32) -> i32>(b"nvmlDeviceGetPowerUsage\0").map_err(|e| e.to_string())?;
    let get_limit = *lib
        .get::<unsafe extern "C" fn(NvHandle, *mut u32) -> i32>(b"nvmlDeviceGetPowerManagementLimit\0")
        .map_err(|e| e.to_string())?;
    let constraints = *lib
        .get::<unsafe extern "C" fn(NvHandle, *mut u32, *mut u32) -> i32>(b"nvmlDeviceGetPowerManagementLimitConstraints\0")
        .map_err(|e| e.to_string())?;
    let set_limit = *lib
        .get::<unsafe extern "C" fn(NvHandle, u32) -> i32>(b"nvmlDeviceSetPowerManagementLimit\0")
        .map_err(|e| e.to_string())?;
    if init() != 0 {
        return Err("NVML initialization failed".into());
    }
    Ok(Arc::new(Nvml { _lib: lib, by_pci, usage, get_limit, constraints, set_limit }))
}

fn nv_open(nv: &Nvml, pci: &str) -> Result<Dev, String> {
    let c = CString::new(pci).map_err(|e| e.to_string())?;
    let mut h: NvHandle = std::ptr::null_mut();
    let rc = unsafe { (nv.by_pci)(c.as_ptr(), &mut h) };
    if rc != 0 || h.is_null() {
        return Err(format!("NVML cannot open the GPU at {pci} (code {rc})"));
    }
    Ok(Dev(h))
}

fn nv_get(f: unsafe extern "C" fn(NvHandle, *mut u32) -> i32, d: Dev) -> Option<u32> {
    let mut v = 0u32;
    if unsafe { f(d.0, &mut v) } == 0 { Some(v) } else { None }
}

fn nvidia_cap(nv: &Arc<Nvml>, d: Dev, want_w: f64) -> Result<CapResult, String> {
    let cur = nv_get(nv.get_limit, d).ok_or("the driver does not report a power limit")?;
    let (mut min, mut max) = (0u32, 0u32);
    if unsafe { (nv.constraints)(d.0, &mut min, &mut max) } != 0 {
        return Err("the driver does not expose power-limit constraints".into());
    }
    let want = (want_w * 1000.0).round() as u32;
    if want >= cur {
        return Ok(CapResult {
            enforced: true,
            restore: None,
            note: format!("card power limit is {:.0} W, already within the ceiling", cur as f64 / 1000.0),
        });
    }
    let target = want.max(min);
    let rc = unsafe { (nv.set_limit)(d.0, target) };
    if rc != 0 {
        return Err(match rc {
            4 => "no permission, run as Administrator/root".to_string(),
            3 => "the driver reports power-limit changes as not supported".to_string(),
            _ => format!("NVML error {rc}"),
        });
    }
    let restore = Restore::Nvml { nv: nv.clone(), dev: d, orig_mw: cur };
    match nv_get(nv.get_limit, d) {
        Some(r) if r.abs_diff(target) <= 1000 => {}
        _ => {
            let _ = restore.apply();
            return Err("the driver did not confirm the new limit; reverted".into());
        }
    }
    let note = if target <= want {
        format!("hardware power limit {:.0} W -> {:.0} W (verified)", cur as f64 / 1000.0, target as f64 / 1000.0)
    } else {
        format!("card minimum is {:.0} W (above the requested ceiling); set to the minimum", min as f64 / 1000.0)
    };
    Ok(CapResult { enforced: target <= want, restore: Some(restore), note })
}

struct SmiQ {
    draw: Option<f64>,
    limit: Option<f64>,
    min: Option<f64>,
}

fn smi_query(pci: &str, timeout_s: u64) -> Option<SmiQ> {
    let t = run_smi(&["-i", pci, "--query-gpu=power.draw,power.limit,power.min_limit", "--format=csv,noheader,nounits"], timeout_s)?;
    let f: Vec<Option<f64>> = t.trim().split(',').map(|x| x.trim().parse::<f64>().ok()).collect();
    if f.len() != 3 {
        return None;
    }
    Some(SmiQ { draw: f[0], limit: f[1], min: f[2] })
}

struct SmiSensor {
    pci: String,
    cache: Mutex<(Option<Instant>, Option<f64>)>,
}

impl SmiSensor {
    fn new(pci: String) -> Self {
        Self { pci, cache: Mutex::new((None, None)) }
    }
    // nvidia-smi is a process spawn, so it is read at most once a second.
    fn watts(&self) -> Option<f64> {
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = c.0 {
            if t.elapsed() < Duration::from_millis(1000) {
                return c.1;
            }
        }
        let v = smi_query(&self.pci, 2).and_then(|q| q.draw);
        *c = (Some(Instant::now()), v);
        v
    }
}

fn smi_cap(pci: &str, q: &SmiQ, want_w: f64) -> Result<CapResult, String> {
    let cur = q.limit.ok_or("nvidia-smi reports no power limit")?;
    let min = q.min.unwrap_or(0.0);
    if want_w >= cur {
        return Ok(CapResult {
            enforced: true,
            restore: None,
            note: format!("card power limit is {:.0} W, already within the ceiling", cur),
        });
    }
    let target = want_w.max(min).round();
    run_smi(&["-i", pci, "-pl", &format!("{target}")], 8)
        .ok_or("nvidia-smi could not set the power limit (needs Administrator/root, or the driver does not support it)")?;
    let restore = Restore::Smi { pci: pci.to_string(), orig_w: cur };
    match smi_query(pci, 5).and_then(|q| q.limit) {
        Some(r) if (r - target).abs() <= 1.0 => {}
        _ => {
            let _ = restore.apply();
            return Err("nvidia-smi did not confirm the new power limit; reverted".into());
        }
    }
    let note = if target <= want_w {
        format!("hardware power limit {:.0} W -> {:.0} W via nvidia-smi (verified)", cur, target)
    } else {
        format!("card minimum is {:.0} W (above the requested ceiling); set to the minimum", min)
    };
    Ok(CapResult { enforced: target <= want_w, restore: Some(restore), note })
}

fn probe_nvidia(pci: &str, want: f64) -> Result<Probe, String> {
    match nvml() {
        Ok(nv) => {
            let d = nv_open(&nv, pci)?;
            let sensor = match nv_get(nv.usage, d).filter(|mw| *mw > 0) {
                Some(_) => Some(Sensor::Nvml(nv.clone(), d)),
                // NVML has no power reading on this chip: nvidia-smi sometimes does.
                None => smi_query(&pci8(pci), 5)
                    .and_then(|q| q.draw)
                    .filter(|w| *w > 0.0)
                    .map(|_| Sensor::Smi(SmiSensor::new(pci8(pci)))),
            };
            Ok(Probe { sensor, cap: nvidia_cap(&nv, d, want) })
        }
        Err(e) => {
            let p = pci8(pci);
            let q = smi_query(&p, 5).ok_or_else(|| format!("{e}; nvidia-smi unavailable"))?;
            let sensor = q.draw.filter(|w| *w > 0.0).map(|_| Sensor::Smi(SmiSensor::new(p.clone())));
            let cap = smi_cap(&p, &q, want);
            Ok(Probe { sensor, cap })
        }
    }
}

// ---------------------------------------------------------------------------
// Linux: hwmon (AMD, Intel, any driver that exposes power); energy counters as
// a fallback sensor
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
struct HwmonSensor {
    dir: PathBuf,
    energy: Mutex<Option<(Instant, u64)>>,
}

#[cfg(target_os = "linux")]
impl HwmonSensor {
    fn new(dir: PathBuf) -> Self {
        Self { dir, energy: Mutex::new(None) }
    }
    fn watts(&self) -> Option<f64> {
        let direct = read_u64(&self.dir.join("power1_average"))
            .filter(|v| *v > 0)
            .or_else(|| read_u64(&self.dir.join("power1_input")).filter(|v| *v > 0));
        if let Some(u) = direct {
            return Some(u as f64 / 1e6);
        }
        // energy1_input is in microjoules; watts = delta energy / delta time.
        let e = read_u64(&self.dir.join("energy1_input"))?;
        let now = Instant::now();
        let mut last = self.energy.lock().unwrap_or_else(|x| x.into_inner());
        let out = match *last {
            Some((t, prev)) if e >= prev && now.duration_since(t) >= Duration::from_millis(200) => {
                Some((e - prev) as f64 / 1e6 / now.duration_since(t).as_secs_f64())
            }
            _ => None,
        };
        if out.is_some() || last.is_none() {
            *last = Some((now, e));
        }
        out
    }
}

#[cfg(target_os = "linux")]
fn find_hwmon(pci: &str) -> Option<PathBuf> {
    let dir = PathBuf::from("/sys/bus/pci/devices").join(pci).join("hwmon");
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| ["power1_average", "power1_input", "energy1_input", "power1_cap", "power1_max"].iter().any(|f| p.join(f).exists()))
}

#[cfg(target_os = "linux")]
fn read_u64(p: &Path) -> Option<u64> {
    std::fs::read_to_string(p).ok()?.trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn hwmon_cap(hw: &Path, want_w: f64) -> Result<CapResult, String> {
    let cap = ["power1_cap", "power1_max"]
        .iter()
        .map(|f| hw.join(f))
        .find(|p| read_u64(p).is_some())
        .ok_or("no writable power cap is exposed by this driver")?;
    let cur = read_u64(&cap).ok_or("power cap unreadable")?;
    let min = read_u64(&hw.join("power1_cap_min")).unwrap_or(0);
    let want = (want_w * 1e6).round() as u64;
    if want >= cur {
        return Ok(CapResult {
            enforced: true,
            restore: None,
            note: format!("card power cap is {:.0} W, already within the ceiling", cur as f64 / 1e6),
        });
    }
    let target = want.max(min);
    let orig = std::fs::read_to_string(&cap).map_err(|e| e.to_string())?.trim().to_string();
    std::fs::write(&cap, target.to_string()).map_err(|e| format!("{e} (run as root)"))?;
    let restore = Restore::Sysfs { path: cap.clone(), orig };
    match read_u64(&cap) {
        Some(r) if r.abs_diff(target) <= 1_000_000 => {}
        _ => {
            let _ = restore.apply();
            return Err("the driver did not confirm the new power cap; reverted".into());
        }
    }
    let note = if target <= want {
        format!("hardware power cap {:.0} W -> {:.0} W (verified)", cur as f64 / 1e6, target as f64 / 1e6)
    } else {
        format!("card minimum is {:.0} W (above the requested ceiling); set to the minimum", min as f64 / 1e6)
    };
    Ok(CapResult { enforced: target <= want, restore: Some(restore), note })
}

#[cfg(target_os = "linux")]
fn probe_system(pci: &str, want: f64) -> Result<Probe, String> {
    let hw = find_hwmon(pci).ok_or("no hwmon power interface for this PCI device")?;
    let cap = hwmon_cap(&hw, want);
    Ok(Probe { sensor: Some(Sensor::Hwmon(HwmonSensor::new(hw))), cap })
}

#[cfg(windows)]
fn probe_system(pci: &str, _want: f64) -> Result<Probe, String> {
    let a = adl::Adl::open(pci)?;
    Ok(Probe {
        sensor: Some(Sensor::Adl(a)),
        cap: Err("Windows AMD: no driver-side cap in this build, enforcing by software governor on ASIC power".into()),
    })
}

#[cfg(not(any(target_os = "linux", windows)))]
fn probe_system(_pci: &str, _want: f64) -> Result<Probe, String> {
    Err("non-NVIDIA power control is not supported on this OS".into())
}

// AMD Windows: ADL PMLog power sensor (ASIC power, watts).
#[cfg(windows)]
mod adl {
    use libloading::Library;
    use std::ffi::c_void;
    use std::sync::{Arc, OnceLock};

    type Ctx = *mut c_void;
    type Alloc = unsafe extern "system" fn(i32) -> *mut c_void;

    #[repr(C)]
    struct AdapterInfo {
        size: i32,
        index: i32,
        udid: [u8; 256],
        bus: i32,
        device: i32,
        function: i32,
        vendor: i32,
        name: [u8; 256],
        display: [u8; 256],
        present: i32,
        exist: i32,
        driver: [u8; 256],
        driver_ext: [u8; 256],
        pnp: [u8; 256],
        os_index: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Sensor {
        supported: i32,
        value: i32,
    }
    #[repr(C)]
    struct PmLog {
        size: i32,
        sensors: [Sensor; 256],
    }

    struct Malloc {
        _lib: Library,
        f: unsafe extern "C" fn(usize) -> *mut c_void,
    }
    static MALLOC: OnceLock<Malloc> = OnceLock::new();
    unsafe extern "system" fn alloc(n: i32) -> *mut c_void {
        if n <= 0 {
            std::ptr::null_mut()
        } else {
            MALLOC.get().map(|m| (m.f)(n as usize)).unwrap_or(std::ptr::null_mut())
        }
    }

    struct Lib {
        _lib: Library,
        ctx: Ctx,
        number: unsafe extern "C" fn(Ctx, *mut i32) -> i32,
        info: unsafe extern "C" fn(Ctx, *mut AdapterInfo, i32) -> i32,
        query: unsafe extern "C" fn(Ctx, i32, *mut PmLog) -> i32,
    }
    unsafe impl Send for Lib {}
    unsafe impl Sync for Lib {}

    static LIB: OnceLock<Result<Arc<Lib>, String>> = OnceLock::new();
    fn lib() -> Result<Arc<Lib>, String> {
        LIB.get_or_init(|| unsafe { load() }).clone()
    }

    unsafe fn load() -> Result<Arc<Lib>, String> {
        if MALLOC.get().is_none() {
            let crt = Library::new("msvcrt.dll").map_err(|e| e.to_string())?;
            let f = *crt.get::<unsafe extern "C" fn(usize) -> *mut c_void>(b"malloc\0").map_err(|e| e.to_string())?;
            let _ = MALLOC.set(Malloc { _lib: crt, f });
        }
        let l = Library::new("atiadlxx.dll").or_else(|_| Library::new("atiadlxy.dll")).map_err(|e| format!("ADL library not found: {e}"))?;
        let create = *l
            .get::<unsafe extern "C" fn(Alloc, i32, *mut Ctx) -> i32>(b"ADL2_Main_Control_Create\0")
            .map_err(|e| e.to_string())?;
        let number = *l
            .get::<unsafe extern "C" fn(Ctx, *mut i32) -> i32>(b"ADL2_Adapter_NumberOfAdapters_Get\0")
            .map_err(|e| e.to_string())?;
        let info = *l
            .get::<unsafe extern "C" fn(Ctx, *mut AdapterInfo, i32) -> i32>(b"ADL2_Adapter_AdapterInfo_Get\0")
            .map_err(|e| e.to_string())?;
        let query = *l
            .get::<unsafe extern "C" fn(Ctx, i32, *mut PmLog) -> i32>(b"ADL2_New_QueryPMLogData_Get\0")
            .map_err(|e| e.to_string())?;
        let mut ctx: Ctx = std::ptr::null_mut();
        if create(alloc, 1, &mut ctx) != 0 || ctx.is_null() {
            return Err("ADL initialization failed".into());
        }
        Ok(Arc::new(Lib { _lib: l, ctx, number, info, query }))
    }

    pub struct Adl {
        lib: Arc<Lib>,
        index: i32,
    }

    impl Adl {
        pub fn open(pci: &str) -> Result<Self, String> {
            let lib = lib()?;
            let (_, rest) = pci.split_once(':').ok_or("invalid PCI address")?;
            let (bus, rest) = rest.split_once(':').ok_or("invalid PCI address")?;
            let (dev, fun) = rest.split_once('.').ok_or("invalid PCI address")?;
            let bus = i32::from_str_radix(bus, 16).map_err(|e| e.to_string())?;
            let dev = i32::from_str_radix(dev, 16).map_err(|e| e.to_string())?;
            let fun = i32::from_str_radix(fun, 16).map_err(|e| e.to_string())?;
            unsafe {
                let mut n = 0i32;
                if (lib.number)(lib.ctx, &mut n) != 0 || n <= 0 || n > 128 {
                    return Err("ADL reports no adapters".into());
                }
                let mut v: Vec<AdapterInfo> = (0..n).map(|_| std::mem::zeroed()).collect();
                for a in v.iter_mut() {
                    a.size = std::mem::size_of::<AdapterInfo>() as i32;
                }
                if (lib.info)(lib.ctx, v.as_mut_ptr(), (v.len() * std::mem::size_of::<AdapterInfo>()) as i32) != 0 {
                    return Err("ADL adapter info failed".into());
                }
                let a = v
                    .iter()
                    .find(|a| a.present != 0 && a.bus == bus && a.device == dev && a.function == fun)
                    .ok_or("no ADL adapter matches this PCI address")?;
                Ok(Self { lib: lib.clone(), index: a.index })
            }
        }

        pub fn watts(&self) -> Option<f64> {
            unsafe {
                let mut m: PmLog = std::mem::zeroed();
                m.size = std::mem::size_of::<PmLog>() as i32;
                if (self.lib.query)(self.lib.ctx, self.index, &mut m) != 0 {
                    return None;
                }
                let s = m.sensors[23]; // ADL_PMLOG_ASIC_POWER
                if s.supported != 0 && s.value > 0 { Some(s.value as f64) } else { None }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (pure logic only; no GPU needed)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stays_at_full_speed_when_well_under_the_limit() {
        let mut c = Controller::new(150.0, true);
        for _ in 0..200 {
            c.update(Some(100.0));
        }
        assert!(!c.paused);
        assert!(c.duty > 0.99);
    }

    #[test]
    fn closed_loop_converges_below_the_limit() {
        // GPU model: 20 W idle, 220 W at full duty.
        let limit = 120.0;
        let mut c = Controller::new(limit, true);
        let mut p = 220.0;
        for _ in 0..600 {
            c.update(Some(p));
            p = 20.0 + 200.0 * c.duty;
        }
        assert!(!c.paused);
        assert!(p <= limit, "power {p} above limit");
        assert!(c.duty > 0.3, "throttled far too hard: {}", c.duty);
    }

    #[test]
    fn hard_trip_pauses_and_hysteresis_resumes() {
        let mut c = Controller::new(100.0, true);
        c.update(Some(130.0));
        assert!(c.paused);
        for _ in 0..200 {
            c.update(Some(30.0));
        }
        assert!(!c.paused);
    }

    #[test]
    fn lost_or_zero_sensor_pauses_when_fail_closed() {
        let mut c = Controller::new(100.0, true);
        for _ in 0..SENSOR_LOST_SAMPLES {
            c.update(None);
        }
        assert!(c.paused);
        let mut z = Controller::new(100.0, true);
        for _ in 0..SENSOR_LOST_SAMPLES {
            z.update(Some(0.0));
        }
        assert!(z.paused);
    }

    #[test]
    fn lost_sensor_does_not_pause_behind_a_hardware_cap() {
        let mut c = Controller::new(110.0, false);
        for _ in 0..100 {
            c.update(None);
        }
        assert!(!c.paused);
    }

    #[test]
    fn limit_below_idle_power_stays_paused() {
        let mut c = Controller::new(10.0, true);
        for _ in 0..300 {
            c.update(Some(25.0));
        }
        assert!(c.paused);
    }

    #[test]
    fn parses_nvidia_smi_and_pci_forms() {
        let v = parse_smi("0, 00000000:01:00.0, NVIDIA GeForce RTX 3060 Laptop GPU\n1, 00000000:0B:00.0, NVIDIA P106-100\n");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].pci, "0000:01:00.0");
        assert_eq!(v[1].pci, "0000:0b:00.0");
        assert_eq!(normalize_pci("0000:01:00.0"), "0000:01:00.0");
        assert_eq!(normalize_pci("00000000:0B:00.0"), "0000:0b:00.0");
        assert_eq!(pci8("0000:01:00.0"), "00000000:01:00.0");
    }

    #[test]
    fn names_must_match_exactly_after_normalisation() {
        assert!(names_equal("NVIDIA GeForce RTX 3060", "GeForce RTX 3060"));
        assert!(!names_equal("GeForce RTX 3060", "GeForce RTX 3060 Ti"));
        assert!(!names_equal("", ""));
    }

    #[test]
    fn blind_throttle_never_blocks_and_keeps_its_duty() {
        let t = Throttle::fixed(0.5);
        assert!(!t.blocked());
        assert_eq!(t.duty_milli.load(Ordering::Relaxed), 500);
    }

    #[test]
    fn rejects_absurd_limits() {
        assert!(configure(f64::NAN).is_err());
        assert!(configure(1.0).is_err());
        assert!(configure(5000.0).is_err());
        assert!(configure_blind_duty(0).is_err());
        assert!(configure_blind_duty(101).is_err());
    }
}
