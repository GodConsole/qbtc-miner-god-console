// src/gpu_control.rs
//
// Cross-vendor GPU power limit management for QBTC miner.
//
// Backends, in order:
//   1. NVIDIA     : nvidia-smi -i <idx> -pl <watts>
//                   (tries mobile vBIOS too; -pl often works even when
//                    power.min_limit/max_limit report "N/A")
//   2. AMD Linux  : /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap
//                   fallback: power_dpm_force_performance_level=low|auto
//   3. AMD Windows: ADL2_Overdrive8_Setting_Set (atiadlxx.dll)
//
// Percent mode (--power-percent N) is the safer choice on mobile chips whose
// vBIOS locks the absolute wattage: it derives the target from the default
// TDP reported by the driver, not the current cap.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor { Nvidia, Amd, Intel, Unknown }

#[derive(Debug, Clone)]
pub struct GpuSnapshot {
    pub index: i32,
    pub vendor: Vendor,
    pub name: String,
    pub pci: Option<String>,
    pub power_default_w: Option<f64>,
    pub power_min_w: Option<f64>,
    pub power_max_w: Option<f64>,
    pub power_current_w: Option<f64>,
}

#[derive(Debug)]
pub struct PowerReport {
    pub index: i32,
    pub vendor: Vendor,
    pub name: String,
    pub requested_w: Option<f64>,
    pub applied_w: Option<f64>,
    pub success: bool,
    pub note: String,
}

pub struct PowerLimiter {
    /// Snapshot taken before any write, kept for diagnostics / manual reset.
    pub snapshots: Vec<GpuSnapshot>,
}

impl PowerLimiter {
    pub fn new() -> Self {
        Self { snapshots: enumerate_gpus() }
    }

    /// Apply --max-power W (absolute) and/or --power-percent P to every
    /// eligible GPU. On mobile chips that reject -pl we fall back
    /// automatically and report it in `note`.
    pub fn apply_all(&self, watts: Option<u32>, percent: Option<u32>) -> Vec<PowerReport> {
        let mut reports = Vec::new();
        reports.extend(apply_nvidia(&self.snapshots, watts, percent));
        reports.extend(apply_amd_linux(&self.snapshots, watts, percent));
        #[cfg(target_os = "windows")]
        reports.extend(adl::apply_amd_windows(&self.snapshots, watts, percent));
        reports
    }

    pub fn print_report(&self, reports: &[PowerReport]) {
        if reports.is_empty() {
            println!("[INFO] GPU: no power-limit-capable devices detected.");
            return;
        }
        for r in reports {
            if r.success {
                match (r.requested_w, r.applied_w) {
                    (Some(req), Some(app)) =>
                        println!("[INFO] GPU {} ({}): power limit {:.0}W → read back {:.0}W",
                                 r.index, r.name, req, app),
                    (Some(req), None) =>
                        println!("[INFO] GPU {} ({}): power limit set to {:.0}W (driver did not echo)",
                                 r.index, r.name, req),
                    _ => println!("[INFO] GPU {} ({}): {}", r.index, r.name, r.note),
                }
            } else {
                println!("[WARN] GPU {} ({}): power limit NOT applied — {}",
                         r.index, r.name, r.note);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Enumeration
// ---------------------------------------------------------------------------

pub fn enumerate_gpus() -> Vec<GpuSnapshot> {
    let mut out = Vec::new();
    out.extend(snapshot_nvidia());
    out.extend(snapshot_amd_linux());
    #[cfg(target_os = "windows")]
    out.extend(adl::snapshot_amd_windows());
    out
}

fn snapshot_nvidia() -> Vec<GpuSnapshot> {
    let out = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,pci.bus_id,power.default_limit,power.min_limit,power.max_limit,power.limit",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let mut list = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let p: Vec<&str> = line.split(',').map(str::trim).collect();
        if p.len() < 7 { continue; }
        let index: i32 = match p[0].parse() { Ok(i) => i, Err(_) => continue };
        let parse = |s: &str| -> Option<f64> {
            if s.eq_ignore_ascii_case("N/A") || s.is_empty() { None } else { s.parse().ok() }
        };
        list.push(GpuSnapshot {
            index,
            vendor: Vendor::Nvidia,
            name: p[1].to_string(),
            pci: Some(p[2].to_string()),
            power_default_w: parse(p[3]),
            power_min_w: parse(p[4]),
            power_max_w: parse(p[5]),
            power_current_w: parse(p[6]),
        });
    }
    list
}

fn snapshot_amd_linux() -> Vec<GpuSnapshot> {
    let mut list = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/drm") else { return list; };
    for e in entries.flatten() {
        let fname = e.file_name().to_string_lossy().to_string();
        if !fname.starts_with("card") || fname.contains('-') { continue; }
        let dev = e.path().join("device");
        if !dev.exists() { continue; }
        let vendor = fs::read_to_string(dev.join("vendor")).unwrap_or_default();
        if vendor.trim() != "0x1002" { continue; }
        let Some(hwmon) = find_amdgpu_hwmon(&dev) else { continue; };
        let read_f = |name: &str| -> Option<f64> {
            fs::read_to_string(hwmon.join(name)).ok()?.trim().parse::<f64>().ok().map(|v| v / 1e6)
        };
        list.push(GpuSnapshot {
            index: -1,
            vendor: Vendor::Amd,
            name: fname,
            pci: Some(dev.file_name().unwrap_or_default().to_string_lossy().to_string()),
            power_default_w: read_f("power1_cap"), // reference for percent
            power_min_w: read_f("power1_cap_min"),
            power_max_w: read_f("power1_cap_max"),
            power_current_w: read_f("power1_average").or_else(|| read_f("power1_input")),
        });
    }
    list
}

fn find_amdgpu_hwmon(device: &Path) -> Option<PathBuf> {
    let hw = device.join("hwmon");
    let entries = fs::read_dir(&hw).ok()?;
    for e in entries.flatten() {
        let p = e.path();
        if let Ok(n) = fs::read_to_string(p.join("name")) {
            if n.trim() == "amdgpu" { return Some(p); }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// NVIDIA backend
// ---------------------------------------------------------------------------

fn apply_nvidia(snaps: &[GpuSnapshot], watts: Option<u32>, percent: Option<u32>) -> Vec<PowerReport> {
    let mut reports = Vec::new();
    for s in snaps.iter().filter(|s| s.vendor == Vendor::Nvidia) {
        let target_w = match (watts, percent) {
            (Some(w), _) => Some(w as f64),
            (_, Some(p)) => s.power_default_w.or(s.power_current_w).map(|base| base * p as f64 / 100.0),
            _ => continue,
        };
        let Some(tw) = target_w else { continue; };

        if let (Some(lo), Some(hi)) = (s.power_min_w, s.power_max_w) {
            if tw < lo - 0.5 || tw > hi + 0.5 {
                reports.push(PowerReport {
                    index: s.index, vendor: Vendor::Nvidia, name: s.name.clone(),
                    requested_w: Some(tw), applied_w: None, success: false,
                    note: format!("{:.0}W outside driver range [{:.0}W..{:.0}W]", tw, lo, hi),
                });
                continue;
            }
        }

        // On mobile MXM/desktop-with-mobile-die cards, min/max often show N/A
        // yet -pl still works on some driver versions. Try it anyway.
        let idx_s = s.index.to_string();
        let w_s = format!("{:.0}", tw);
        let attempt = Command::new("nvidia-smi")
            .args(["-i", &idx_s, "-pl", &w_s])
            .output();

        match attempt {
            Ok(o) if o.status.success() => {
                let applied = read_nv_watts(s.index);
                reports.push(PowerReport {
                    index: s.index, vendor: Vendor::Nvidia, name: s.name.clone(),
                    requested_w: Some(tw), applied_w: applied, success: true,
                    note: "nvidia-smi -pl accepted".into(),
                });
            }
            Ok(o) => {
                let err = String::from_utf8_lossy(&o.stderr)
                    .lines().next().unwrap_or("unknown").to_string();
                let hint = if s.power_min_w.is_none() && s.power_max_w.is_none() {
                    " (mobile vBIOS may lock power; try --power-percent)"
                } else { "" };
                reports.push(PowerReport {
                    index: s.index, vendor: Vendor::Nvidia, name: s.name.clone(),
                    requested_w: Some(tw), applied_w: None, success: false,
                    note: format!("nvidia-smi rejected: {}{}", err, hint),
                });
            }
            Err(e) => reports.push(PowerReport {
                index: s.index, vendor: Vendor::Nvidia, name: s.name.clone(),
                requested_w: Some(tw), applied_w: None, success: false,
                note: format!("spawn error: {}", e),
            }),
        }
    }
    reports
}

fn read_nv_watts(index: i32) -> Option<f64> {
    let out = Command::new("nvidia-smi")
        .args(["-i", &index.to_string(),
               "--query-gpu=power.limit",
               "--format=csv,noheader,nounits"])
        .output().ok()?;
    if !out.status.success() { return None; }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

// ---------------------------------------------------------------------------
// AMD Linux backend
// ---------------------------------------------------------------------------

fn apply_amd_linux(snaps: &[GpuSnapshot], watts: Option<u32>, percent: Option<u32>) -> Vec<PowerReport> {
    let mut reports = Vec::new();
    for s in snaps.iter().filter(|s| s.vendor == Vendor::Amd) {
        // Re-derive the hwmon path for this device
        let Some(dev) = PathBuf::from("/sys/class/drm").join(&s.name).join("device").canonicalize().ok() else { continue; };
        let Some(hwmon) = find_amdgpu_hwmon(&dev) else { continue; };

        let target_w = match (watts, percent) {
            (Some(w), _) => Some(w as f64),
            (_, Some(p)) => s.power_default_w.or(s.power_current_w).map(|b| b * p as f64 / 100.0),
            _ => continue,
        };
        let Some(tw) = target_w else { continue; };

        if let (Some(lo), Some(hi)) = (s.power_min_w, s.power_max_w) {
            if tw < lo - 0.5 || tw > hi + 0.5 {
                reports.push(PowerReport {
                    index: -1, vendor: Vendor::Amd, name: s.name.clone(),
                    requested_w: Some(tw), applied_w: None, success: false,
                    note: format!("{:.0}W outside [{:.0}W..{:.0}W]", tw, lo, hi),
                });
                continue;
            }
        }

        let cap_path = hwmon.join("power1_cap");
        let value_uw = (tw * 1e6).round() as u64;
        match fs::write(&cap_path, value_uw.to_string()) {
            Ok(_) => {
                let applied = fs::read_to_string(&cap_path).ok()
                    .and_then(|v| v.trim().parse::<f64>().ok()).map(|v| v / 1e6);
                reports.push(PowerReport {
                    index: -1, vendor: Vendor::Amd, name: s.name.clone(),
                    requested_w: Some(tw), applied_w: applied, success: true,
                    note: format!("sysfs power1_cap written ({} -> {})", cap_path.display(), value_uw),
                });
            }
            Err(e) => {
                // Mobile APUs often lack power1_cap; coarse fallback.
                let lowered = tw < s.power_current_w.unwrap_or(0.0) - 1.0;
                let level = if lowered { "low" } else { "auto" };
                let perf = dev.join("power_dpm_force_performance_level");
                let fallback_ok = fs::write(&perf, level).is_ok();
                let note = if fallback_ok {
                    format!("power1_cap write failed ({}); fell back to DPM={}", e, level)
                } else {
                    format!("power1_cap write failed ({}); mobile vBIOS likely locked, use --power-percent or driver reload", e)
                };
                reports.push(PowerReport {
                    index: -1, vendor: Vendor::Amd, name: s.name.clone(),
                    requested_w: Some(tw), applied_w: None, success: fallback_ok, note,
                });
            }
        }
    }
    reports
}

// ---------------------------------------------------------------------------
// AMD Windows backend (ADL Overdrive8)
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
mod adl {
    use super::*;
    use libloading::Library;
    use std::ffi::c_void;

    type Context = *mut c_void;
    type AdlMainControlCreate = unsafe extern "C" fn(
        Option<unsafe extern "C" fn(i32, *const i8)>, i32, *mut Context) -> i32;
    type AdlMainControlDestroy = unsafe extern "C" fn(Context) -> i32;
    type AdlAdapterNumberGet = unsafe extern "C" fn(Context, *mut i32) -> i32;
    type AdlAdapterInfoGet = unsafe extern "C" fn(Context, *mut AdapterInfo, i32) -> i32;
    type AdlOd8InitGet = unsafe extern "C" fn(Context, i32, *mut Od8Init) -> i32;
    type AdlOd8Set = unsafe extern "C" fn(Context, i32, *mut Od8Set, *mut Od8Init) -> i32;

    #[repr(C)]
    #[derive(Clone, Copy)]
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
    #[derive(Clone, Copy, Default)]
    struct Od8SingleInit {
        feature_id: i32,
        default_value: i32,
        min_value: i32,
        max_value: i32,
        step: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Od8Init {
        count: i32,
        capabilities: i32,
        table: [Od8SingleInit; 32],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Od8SingleSet {
        feature_id: i32,
        value: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Od8Set {
        count: i32,
        table: [Od8SingleSet; 32],
    }

    const ADL_OD8_POWER_LIMIT: i32 = 3; // % of default TDP

    struct AdlCtx {
        _lib: Library,
        ctx: Context,
        destroy: AdlMainControlDestroy,
        number: AdlAdapterNumberGet,
        info: AdlAdapterInfoGet,
        od8_init: AdlOd8InitGet,
        od8_set: AdlOd8Set,
    }

    fn load() -> Option<AdlCtx> {
        unsafe {
            let lib = Library::new("atiadlxx.dll").ok()?;
            let create: AdlMainControlCreate = *lib.get(b"ADL2_Main_Control_Create\0").ok()?;
            let destroy: AdlMainControlDestroy = *lib.get(b"ADL2_Main_Control_Destroy\0").ok()?;
            let number: AdlAdapterNumberGet = *lib.get(b"ADL2_Adapter_NumberOfAdapters_Get\0").ok()?;
            let info: AdlAdapterInfoGet = *lib.get(b"ADL2_Adapter_AdapterInfo_Get\0").ok()?;
            let od8_init: AdlOd8InitGet = *lib.get(b"ADL2_Overdrive8_Init_Setting_Get\0").ok()?;
            let od8_set: AdlOd8Set = *lib.get(b"ADL2_Overdrive8_Setting_Set\0").ok()?;
            let mut ctx = std::ptr::null_mut();
            if create(None, 1, &mut ctx) != 0 || ctx.is_null() { return None; }
            Some(AdlCtx { _lib: lib, ctx, destroy, number, info, od8_init, od8_set })
        }
    }

    pub fn snapshot_amd_windows() -> Vec<GpuSnapshot> {
        let Some(ctx) = load() else { return Vec::new(); };
        let mut list = Vec::new();
        unsafe {
            let mut n = 0;
            if (ctx.number)(ctx.ctx, &mut n) != 0 || n <= 0 || n > 128 { (ctx.destroy)(ctx.ctx); return list; }
            let mut adapters = vec![std::mem::zeroed::<AdapterInfo>(); n as usize];
            for a in &mut adapters { a.size = std::mem::size_of::<AdapterInfo>() as i32; }
            if (ctx.info)(ctx.ctx, adapters.as_mut_ptr(), (n as usize * std::mem::size_of::<AdapterInfo>()) as i32) != 0 {
                (ctx.destroy)(ctx.ctx);
                return list;
            }
            for a in &adapters {
                if a.present == 0 || a.exist == 0 { continue; }
                let name = String::from_utf8_lossy(&a.name).trim_end_matches('\0').to_string();
                let pci = format!("{:04x}:{:02x}:{:02x}.{}", 0, a.bus, a.device, a.function);
				let (min_pct, max_pct, default_pct) = {
					let mut init = std::mem::zeroed::<Od8Init>();
					if (ctx.od8_init)(ctx.ctx, a.index, &mut init) == 0 {
						if let Some(row) = init.table.iter().find(|r| r.feature_id == ADL_OD8_POWER_LIMIT) {
							(row.min_value as f64, row.max_value as f64, row.default_value as f64)
						} else { (0.0, 0.0, 0.0) }
					} else { (0.0, 0.0, 0.0) }
				};
                list.push(GpuSnapshot {
                    index: a.index,
                    vendor: Vendor::Amd,
                    name,
                    pci: Some(pci),
                    power_default_w: Some(default_pct),
                    power_min_w: Some(min_pct),
                    power_max_w: Some(max_pct),
                    power_current_w: None,
                });
            }
        }
        unsafe { (ctx.destroy)(ctx.ctx); }
        list
    }

    pub fn apply_amd_windows(snaps: &[GpuSnapshot], watts: Option<u32>, percent: Option<u32>) -> Vec<PowerReport> {
        let Some(ctx) = load() else {
            // No ADL: still report so user knows why nothing happened.
            return snaps.iter().filter(|s| s.vendor == Vendor::Amd).map(|s| PowerReport {
                index: s.index, vendor: Vendor::Amd, name: s.name.clone(),
                requested_w: None, applied_w: None, success: false,
                note: "atiadlxx.dll not available (AMD driver missing?)".into(),
            }).collect();
        };

        let mut reports = Vec::new();
        for s in snaps.iter().filter(|s| s.vendor == Vendor::Amd) {
            // Percent is native to OD8 (feature = ADL_OD8_POWER_LIMIT, value = % of default).
            let pct: Option<i32> = match (watts, percent) {
                (_, Some(p)) => Some(p as i32),
                (Some(w), None) => s.power_default_w.map(|d| ((w as f64) * 100.0 / d.max(1.0)) as i32),
                _ => None,
            };
            let Some(p) = pct else { continue; };

            if let (Some(lo), Some(hi)) = (s.power_min_w, s.power_max_w) {
                if (p as f64) < lo || (p as f64) > hi {
                    reports.push(PowerReport {
                        index: s.index, vendor: Vendor::Amd, name: s.name.clone(),
                        requested_w: watts.map(|w| w as f64), applied_w: None, success: false,
                        note: format!("{}% outside OD8 range [{:.0}%..{:.0}%] (mobile vBIOS may be locked)",
                                      p, lo, hi),
                    });
                    continue;
                }
            }

            let ok = unsafe {
                let mut init = std::mem::zeroed::<Od8Init>();
                if (ctx.od8_init)(ctx.ctx, s.index, &mut init) != 0 { false }
                else {
                    let mut set = std::mem::zeroed::<Od8Set>();
                    set.count = 1;
                    set.table[0].feature_id = ADL_OD8_POWER_LIMIT;
                    set.table[0].value = p;
                    (ctx.od8_set)(ctx.ctx, s.index, &mut set, &mut init) == 0
                }
            };

            reports.push(PowerReport {
                index: s.index, vendor: Vendor::Amd, name: s.name.clone(),
                requested_w: watts.map(|w| w as f64),
                applied_w: None,
                success: ok,
                note: if ok { format!("ADL OD8 power limit set to {}% of default", p) }
                      else { format!("ADL OD8 rejected {}% (min {:.0}% max {:.0}%); mobile vBIOS may be locked",
                                     p, s.power_min_w.unwrap_or(0.0), s.power_max_w.unwrap_or(0.0)) },
            });
        }
        unsafe { (ctx.destroy)(ctx.ctx); }
        reports
    }
}