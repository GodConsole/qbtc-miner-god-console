// src/stats.rs
//
// Process-wide mining statistics. Everything here is cheap and lock-light: the
// miners and the pool client only bump counters / push a log line, and a
// consumer (e.g. a dashboard) can read a JSON snapshot via `STATS.snapshot()`.

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_EVENTS: usize = 80;
const MAX_HISTORY: usize = 150; // 150 samples x 2 s = 5 minutes
const GPU_SLOTS: usize = 8;
const CPU_SLOTS: usize = 32;
const STALE_MS: u64 = 10_000; // a device silent for 10 s counts as idle

pub struct Event {
    pub t: u64,
    pub lvl: &'static str, // info | ok | warn | err | block
    pub msg: String,
}

pub struct Stats {
    started: AtomicU64,
    accepted: AtomicU64,
    rejected: AtomicU64,
    blocks: AtomicU64,
    last_share: AtomicU64,
    height: AtomicU64,
    target: AtomicU64,
    share_target: AtomicU64,
    connected: AtomicBool,
    authorized: AtomicBool,
    cpu_rates: Mutex<[(f64, u64); CPU_SLOTS]>, // (MH/s, last update ms) per CPU thread
    mode: Mutex<&'static str>,                 // idle | solo | pool
    pool: Mutex<(String, String)>,             // (endpoint, payout address)
    gpus: Mutex<[(f64, u64); GPU_SLOTS]>,      // (MH/s, last update in unix ms)
    reject_reasons: Mutex<Vec<(String, u64)>>,
    events: Mutex<VecDeque<Event>>,
    history: Mutex<VecDeque<(u64, f64)>>,
    peak_sum: Mutex<(f64, f64, u64)>,          // (peak MH/s, sum of samples, sample count)
}

pub static STATS: Stats = Stats::new();

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

impl Stats {
    pub const fn new() -> Self {
        Self {
            started: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            blocks: AtomicU64::new(0),
            last_share: AtomicU64::new(0),
            height: AtomicU64::new(0),
            target: AtomicU64::new(0),
            share_target: AtomicU64::new(0),
            connected: AtomicBool::new(false),
            authorized: AtomicBool::new(false),
            cpu_rates: Mutex::new([(0.0, 0); CPU_SLOTS]),
            mode: Mutex::new("idle"),
            pool: Mutex::new((String::new(), String::new())),
            gpus: Mutex::new([(0.0, 0); GPU_SLOTS]),
            reject_reasons: Mutex::new(Vec::new()),
            events: Mutex::new(VecDeque::new()),
            history: Mutex::new(VecDeque::new()),
            peak_sum: Mutex::new((0.0, 0.0, 0)),
        }
    }

    pub fn mark_start(&self) {
        let _ = self.started.compare_exchange(0, now_secs(), Ordering::SeqCst, Ordering::SeqCst);
    }

    pub fn set_mode(&self, mode: &'static str) {
        *lock(&self.mode) = mode;
    }
    pub fn set_pool(&self, endpoint: &str, address: &str) {
        *lock(&self.pool) = (endpoint.to_string(), address.to_string());
    }
    pub fn set_link(&self, connected: bool, authorized: bool) {
        self.connected.store(connected, Ordering::Relaxed);
        self.authorized.store(authorized, Ordering::Relaxed);
    }
    pub fn set_job(&self, height: u64, target: u64, share_target: u64) {
        self.height.store(height, Ordering::Relaxed);
        self.target.store(target, Ordering::Relaxed);
        self.share_target.store(share_target, Ordering::Relaxed);
    }

    pub fn share_accepted(&self) {
        self.accepted.fetch_add(1, Ordering::Relaxed);
        self.last_share.store(now_secs(), Ordering::Relaxed);
    }
    pub fn share_rejected(&self, reason: &str) {
        self.rejected.fetch_add(1, Ordering::Relaxed);
        let mut reasons = lock(&self.reject_reasons);
        if let Some(entry) = reasons.iter_mut().find(|(r, _)| r == reason) {
            entry.1 += 1;
        } else if reasons.len() < 12 {
            reasons.push((reason.to_string(), 1));
        }
    }
    pub fn block_found(&self) {
        self.blocks.fetch_add(1, Ordering::Relaxed);
    }

    // Called by every CPU mining thread about every 2 s with its current rate.
    pub fn cpu_rate(&self, thread_id: usize, mhs: f64) {
        if thread_id < CPU_SLOTS {
            lock(&self.cpu_rates)[thread_id] = (mhs, now_ms());
        }
    }
    // Called by the GPU loop about every 2 s with that device's current rate.
    pub fn gpu_rate(&self, device: i32, mhs: f64) {
        if device >= 0 && (device as usize) < GPU_SLOTS {
            lock(&self.gpus)[device as usize] = (mhs, now_ms());
        }
    }

    pub fn event(&self, lvl: &'static str, msg: String) {
        let mut ev = lock(&self.events);
        ev.push_back(Event { t: now_secs(), lvl, msg });
        while ev.len() > MAX_EVENTS {
            ev.pop_front();
        }
    }

    fn live_cpu(&self) -> f64 {
        let now = now_ms();
        lock(&self.cpu_rates)
            .iter()
            .filter(|(_, t)| *t > 0 && now.saturating_sub(*t) <= STALE_MS)
            .map(|(r, _)| *r)
            .sum()
    }

    fn live_gpus(&self) -> Vec<(usize, f64)> {
        let now = now_ms();
        lock(&self.gpus)
            .iter()
            .enumerate()
            .filter(|(_, g)| g.1 > 0 && now.saturating_sub(g.1) <= STALE_MS)
            .map(|(i, g)| (i, g.0))
            .collect()
    }

    // Sampled every 2 s by a consumer: feeds the chart, peak and average.
    pub fn sample(&self) {
        let gpu_total: f64 = self.live_gpus().iter().map(|(_, m)| *m).sum();
        let total = gpu_total + self.live_cpu();
        let mut h = lock(&self.history);
        h.push_back((now_secs(), total));
        while h.len() > MAX_HISTORY {
            h.pop_front();
        }
        if total > 0.0 {
            let mut ps = lock(&self.peak_sum);
            if total > ps.0 {
                ps.0 = total;
            }
            ps.1 += total;
            ps.2 += 1;
        }
    }

    pub fn snapshot(&self) -> Value {
        let now = now_secs();
        let started = self.started.load(Ordering::Relaxed);
        let last_share = self.last_share.load(Ordering::Relaxed);
        let gpus = self.live_gpus();
        let gpu_total: f64 = gpus.iter().map(|(_, m)| *m).sum();
        let cpu_mhs = self.live_cpu();
        let (peak, sum, samples) = *lock(&self.peak_sum);
        let (endpoint, address) = lock(&self.pool).clone();

        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "mode": *lock(&self.mode),
            "connected": self.connected.load(Ordering::Relaxed),
            "authorized": self.authorized.load(Ordering::Relaxed),
            "endpoint": endpoint,
            "address": address,
            "backend": std::env::var("QBTC_MINER_BACKEND").unwrap_or_else(|_| "cpu".to_string()),
            "uptime_s": if started > 0 { now.saturating_sub(started) } else { 0 },
            "accepted": self.accepted.load(Ordering::Relaxed),
            "rejected": self.rejected.load(Ordering::Relaxed),
            "blocks": self.blocks.load(Ordering::Relaxed),
            "hashrate_mhs": gpu_total + cpu_mhs,
            "cpu_mhs": cpu_mhs,
            "gpu_mhs": gpu_total,
            "last_share_ago_s": if last_share > 0 { json!(now.saturating_sub(last_share)) } else { Value::Null },
            "height": self.height.load(Ordering::Relaxed),
            "target": self.target.load(Ordering::Relaxed),
            "share_target": self.share_target.load(Ordering::Relaxed),
            "peak_mhs": peak,
            "avg_mhs": if samples > 0 { sum / samples as f64 } else { 0.0 },
            "gpus": gpus.iter().map(|(i, m)| json!({ "id": i, "mhs": m })).collect::<Vec<_>>(),
            "history": lock(&self.history).iter().map(|(t, m)| json!([t, m])).collect::<Vec<_>>(),
            "reject_reasons": lock(&self.reject_reasons).iter().map(|(r, c)| json!({ "reason": r, "count": c })).collect::<Vec<_>>(),
            "events": lock(&self.events).iter().map(|e| json!({ "t": e.t, "lvl": e.lvl, "msg": e.msg })).collect::<Vec<_>>(),
        })
    }
}
