// src/stratum.rs
//
// Lightweight Stratum pool client for Q-BTC.
//
// Started from main.rs when `--stratum <url> --address <addr[.worker]>` is given.
// Pool URL schemes: stratum+tcp:// (plain), stratum+ssl:// or stratum+tls:// (TLS, certificate
// chain AND host name are verified). No scheme = plain TCP. A port is required. In
// this mode the process does NOT open the blockchain database, does not join
// the P2P network and does not run the local consensus: it only
//   1. keeps a TCP connection to the pool,
//   2. turns every `mining.notify` into a BlockTemplate for the miner daemon
//      (CPU / CUDA / hybrid, selected by QBTC_MINER_BACKEND as usual),
//   3. forwards every share the miners find to the pool (`mining.submit`).
//
// Wire format verified against the pool source (qbtc-stratum-pool):
//   mining.submit params = [address, job_id (JSON number), nonce (JSON number, u64)]
//   The pool does req.params[1].as_u64() / req.params[2].as_u64(), so a hex STRING
//   nonce silently becomes 0 and every share is rejected as "Low difficulty share".
//
// Environment switch:
//   QBTC_POOL_DEBUG=1        log every line sent to / received from the pool
//   QBTC_POOL_SUBMIT=num     mining.submit = [user, job (number), nonce (number)]   (default,
//                            what the reference gateway reads)
//   QBTC_POOL_SUBMIT=hex     [user, "job", "nonce as 16 hex digits"]
//   QBTC_POOL_SUBMIT=full    [user, "job", "nonce hex", "block hash hex"]
//   QBTC_POOL_TLS_INSECURE=1 accept an invalid / self-signed pool certificate (unsafe: only for
//                            a pool you trust)

use crate::block::Block;
use crate::miner::{BlockTemplate, Miner, MinerCommand, PoolJob};
use crate::stats::STATS;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::mpsc::UnboundedReceiver;

const MAX_TRACKED_JOBS: usize = 16;
const ID_SUBSCRIBE: u64 = 1;
const ID_AUTHORIZE: u64 = 2;
const FIRST_SUBMIT_ID: u64 = 4;
const IO_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Default)]
struct Stats {
    accepted: u64,
    rejected: u64,
}

enum SessionOutcome {
    // Reconnecting would not help (e.g. the pool refused our address).
    Fatal(String),
    // Network hiccup: back off and reconnect.
    Retry { msg: String, was_authorized: bool },
}

// A job we sent to the miners, remembered so a found share can be mapped back
// to the pool's job id. The miners return a plain Block; its header carries
// previous_hash, merkle_root and the pool's timestamp, which identify the job.
struct JobRecord {
    job_id: u64,
    previous_hash: [u8; 32],
    merkle_root: [u8; 32],
    timestamp: u64,
    // Vardiff: the pool re-sends the same job with a new share target.
    share_target: u64,
}

fn debug_enabled() -> bool {
    std::env::var_os("QBTC_POOL_DEBUG").is_some()
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn decode_hash32(s: &str) -> Result<[u8; 32], String> {
    let s = s.trim();
    let s = s.strip_prefix("0x").unwrap_or(s);
    if s.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", s.len()));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|_| format!("invalid hex near position {}", i * 2))?;
    }
    Ok(out)
}

// The exact 120 header bytes that were hashed (timestamp | prev | merkle | commit | nonce | target,
// all integers big-endian). Lets the pool operator compare with their own computation.
fn header_hex(b: &Block) -> String {
    let h = &b.header;
    let mut v: Vec<u8> = Vec::with_capacity(120);
    v.extend_from_slice(&h.timestamp.to_be_bytes());
    v.extend_from_slice(&h.previous_hash);
    v.extend_from_slice(&h.merkle_root);
    v.extend_from_slice(&h.commit_merkle_root);
    v.extend_from_slice(&h.nonce.to_be_bytes());
    v.extend_from_slice(&h.target.to_be_bytes());
    to_hex(&v)
}

fn as_u64_loose(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
}

// ---------------------------------------------------------------------------
// Pool URL, connection (plain or TLS) and login handling
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
struct PoolEndpoint {
    host: String,
    port: u16,
    tls: bool,
}

impl PoolEndpoint {
    fn connect_addr(&self) -> String {
        if self.host.contains(':') {
            format!("[{}]:{}", self.host, self.port) // IPv6 literal
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
    fn label(&self) -> String {
        format!("{}{}", self.connect_addr(), if self.tls { " [TLS]" } else { "" })
    }
}

fn parse_pool_url(url: &str) -> Result<PoolEndpoint, String> {
    let s = url.trim();
    if s.is_empty() {
        return Err("empty pool url".to_string());
    }
    let (scheme, rest) = match s.find("://") {
        Some(i) => (s[..i].to_ascii_lowercase(), &s[i + 3..]),
        None => (String::new(), s),
    };
    let tls = match scheme.as_str() {
        "" | "stratum+tcp" | "stratum" | "tcp" => false,
        "stratum+ssl" | "stratum+tls" | "ssl" | "tls" => true,
        other => {
            return Err(format!(
                "unsupported scheme '{}://' (use stratum+tcp:// or stratum+ssl://)",
                other
            ))
        }
    };
    // Drop any path/query and any user:pass@ part.
    let rest = rest.split(|c| c == '/' || c == '?' || c == '#').next().unwrap_or("");
    let rest = rest.rsplit('@').next().unwrap_or(rest);

    let (host, port_str): (&str, Option<&str>) = if let Some(stripped) = rest.strip_prefix('[') {
        let end = stripped.find(']').ok_or_else(|| "unterminated '[' in host".to_string())?;
        (&stripped[..end], stripped[end + 1..].strip_prefix(':'))
    } else {
        match rest.rfind(':') {
            Some(i) => (&rest[..i], Some(&rest[i + 1..])),
            None => (rest, None),
        }
    };
    if host.is_empty() {
        return Err("pool host is empty".to_string());
    }
    let port_str = port_str.ok_or_else(|| {
        "pool port is missing (example: stratum+ssl://pool.example.org:3336)".to_string()
    })?;
    let port: u16 = port_str
        .parse()
        .map_err(|_| format!("invalid pool port '{}'", port_str))?;
    Ok(PoolEndpoint { host: host.to_string(), port, tls })
}

// "ADDRESS.worker" -> ("ADDRESS", Some("worker")). Bech32 addresses never contain '.'.
fn split_login(user: &str) -> (String, Option<String>) {
    match user.split_once('.') {
        Some((addr, worker)) if !worker.is_empty() => (addr.to_string(), Some(worker.to_string())),
        _ => (user.to_string(), None),
    }
}

trait PoolIo: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> PoolIo for T {}
type BoxIo = Box<dyn PoolIo>;

async fn open_connection(ep: &PoolEndpoint) -> Result<BoxIo, String> {
    let addr = ep.connect_addr();
    let tcp = match tokio::time::timeout(IO_TIMEOUT, TcpStream::connect(&addr)).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return Err(format!("connect to {} failed: {}", addr, e)),
        Err(_) => return Err(format!("connect to {} timed out", addr)),
    };
    let _ = tcp.set_nodelay(true);
    if !ep.tls {
        return Ok(Box::new(tcp));
    }
    // TLS: the system trust store verifies the certificate chain and the host name.
    let insecure = std::env::var_os("QBTC_POOL_TLS_INSECURE").is_some();
    let mut builder = native_tls::TlsConnector::builder();
    if insecure {
        tracing::warn!("[WARN] Pool: QBTC_POOL_TLS_INSECURE is set, the certificate is NOT verified!");
        builder.danger_accept_invalid_certs(true);
        builder.danger_accept_invalid_hostnames(true);
    }
    let connector = builder.build().map_err(|e| format!("TLS init failed: {}", e))?;
    let connector = tokio_native_tls::TlsConnector::from(connector);
    match tokio::time::timeout(IO_TIMEOUT, connector.connect(&ep.host, tcp)).await {
        Ok(Ok(tls)) => Ok(Box::new(tls)),
        Ok(Err(e)) => Err(format!(
            "TLS handshake with {} failed: {} (certificate problem? Only if you trust this pool you may set QBTC_POOL_TLS_INSECURE=1)",
            ep.host, e
        )),
        Err(_) => Err("TLS handshake timed out".to_string()),
    }
}

async fn send_json<W: AsyncWrite + Unpin>(wr: &mut W, v: &Value) -> Result<(), String> {
    if debug_enabled() {
        tracing::info!("[DEBUG] Pool: -> {}", v);
    }
    let mut line = v.to_string();
    line.push('\n');
    match tokio::time::timeout(IO_TIMEOUT, wr.write_all(line.as_bytes())).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("write timeout".to_string()),
    }
}

// Parses params of `mining.notify`:
// [height, prev_hash, merkle_root, commit_merkle_root, timestamp, target, share_target?]
fn parse_notify(params: &[Value]) -> Result<(JobRecord, BlockTemplate), String> {
    if params.len() < 6 {
        return Err(format!("notify has {} params, expected at least 6", params.len()));
    }
    let height = as_u64_loose(&params[0]).ok_or("height is not a number")?;
    let previous_hash = decode_hash32(params[1].as_str().ok_or("previous_hash is not a string")?)
        .map_err(|e| format!("previous_hash: {}", e))?;
    let merkle_root = decode_hash32(params[2].as_str().ok_or("merkle_root is not a string")?)
        .map_err(|e| format!("merkle_root: {}", e))?;
    let commit_merkle_root = decode_hash32(params[3].as_str().ok_or("commit_merkle_root is not a string")?)
        .map_err(|e| format!("commit_merkle_root: {}", e))?;
    let timestamp = as_u64_loose(&params[4]).ok_or("timestamp is not a number")?;
    let target = as_u64_loose(&params[5]).ok_or("target is not a number")?;
    // 7th param (if present) is the pool's share target (easier than the network target).
    let share_target = params.get(6).and_then(as_u64_loose).unwrap_or(target).max(target);

    let nonce_min = params.get(7).and_then(as_u64_loose);
    let nonce_max = params.get(8).and_then(as_u64_loose);

    let job = PoolJob {
        job_id: height,
        merkle_root,
        commit_merkle_root,
        timestamp,
        header_target: target,
        share_target,
        nonce_min,  
        nonce_max,
    };
    let template = BlockTemplate {
        previous_hash,
        transactions: Vec::new(),
        target,
        current_height: height,
        pool_job: Some(job),
    };
    let record = JobRecord { job_id: height, previous_hash, merkle_root, timestamp, share_target };
    Ok((record, template))
}

// Default dialect = the reference gateway (qbtc-stratum-pool), which reads job id and nonce with
// serde's as_u64(): both must be JSON numbers. Other pools may differ, see QBTC_POOL_SUBMIT.
fn build_submit(id: u64, user: &str, job_id: u64, nonce: u64, hash_hex: &str) -> Value {
    let mode = std::env::var("QBTC_POOL_SUBMIT").unwrap_or_default().to_ascii_lowercase();
    let params = match mode.as_str() {
        "hex" => json!([user, job_id.to_string(), format!("{:016x}", nonce)]),
        "full" => json!([user, job_id.to_string(), format!("{:016x}", nonce), hash_hex]),
        _ => json!([user, job_id, nonce]),
    };
    json!({ "id": id, "method": "mining.submit", "params": params })
}

// `user` is the login sent to the pool, usually "ADDRESS" or "ADDRESS.worker".
pub async fn run(url: String, user: String, password: String) -> Result<(), Box<dyn std::error::Error>> {
    let ep = parse_pool_url(&url)?;
    let (payout, worker) = split_login(&user);
    let label = ep.label();
    tracing::info!(
        "[INFO] Pool: Stratum mode. Pool = {}, payout address = {}, worker = {}",
        label, payout, worker.as_deref().unwrap_or("-")
    );
    STATS.mark_start();
    STATS.set_mode("pool");
    let shown = match &worker {
        Some(w) => format!("{} / {}", label, w),
        None => label.clone(),
    };
    STATS.set_pool(&shown, &payout);
    STATS.event("info", format!("Stratum mode. Pool {}", shown));
    tracing::info!("[INFO] Pool: P2P, local storage and consensus are disabled in this mode.");

    // Miner daemon (same engine as solo mode, backend chosen via QBTC_MINER_BACKEND).
    let (cmd_tx, cmd_rx) = crossbeam_channel::bounded::<MinerCommand>(10);
    let (res_tx, res_rx) = crossbeam_channel::bounded::<Block>(100);
    Miner::start_daemon(cmd_rx, res_tx);

    // Bridge: blocking crossbeam receiver -> async channel. Shares never touch
    // the local consensus queue.
    let (share_tx, mut share_rx) = tokio::sync::mpsc::unbounded_channel::<Block>();
    std::thread::spawn(move || {
        while let Ok(block) = res_rx.recv() {
            if share_tx.send(block).is_err() {
                break;
            }
        }
    });

    let mut stats = Stats::default();

    let session_loop = async {
        let mut backoff: u64 = 2;
        loop {
            let outcome = run_session(&ep, &user, &password, &cmd_tx, &mut share_rx, &mut stats).await;
            // Old work is useless without a live pool connection.
            let _ = cmd_tx.try_send(MinerCommand::Stop);
            STATS.set_link(false, false);
            match outcome {
                SessionOutcome::Fatal(msg) => return Err::<(), String>(msg),
                SessionOutcome::Retry { msg, was_authorized } => {
                    if was_authorized {
                        backoff = 2;
                    }
                    tracing::warn!("[WARN] Pool: {}. Reconnecting in {}s...", msg, backoff);
                    STATS.event("warn", format!("{}. Reconnecting in {}s", msg, backoff));
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(60);
                }
            }
        }
    };

    tokio::select! {
        res = session_loop => {
            let _ = cmd_tx.try_send(MinerCommand::Shutdown);
            match res {
                Ok(()) => Ok(()),
                Err(msg) => Err(msg.into()),
            }
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("[INFO] Pool: Ctrl+C received. Shutting down miner.");
            let _ = cmd_tx.try_send(MinerCommand::Shutdown);
            Ok(())
        }
    }
}

async fn run_session(
    ep: &PoolEndpoint,
    user: &str,
    password: &str,
    cmd_tx: &crossbeam_channel::Sender<MinerCommand>,
    share_rx: &mut UnboundedReceiver<Block>,
    stats: &mut Stats,
) -> SessionOutcome {
    // Shares found before a reconnect belong to jobs the pool no longer knows.
    while share_rx.try_recv().is_ok() {}

    tracing::info!("[INFO] Pool: Connecting to {} ...", ep.label());
    let io: BoxIo = match open_connection(ep).await {
        Ok(io) => io,
        Err(msg) => return SessionOutcome::Retry { msg, was_authorized: false },
    };
    STATS.set_link(true, false);
    STATS.event("info", format!("Connected to {}", ep.label()));
    let (rd, mut wr) = tokio::io::split(io);
    let mut lines = BufReader::new(rd).lines();

    if let Err(e) = send_json(
        &mut wr,
        &json!({ "id": ID_SUBSCRIBE, "method": "mining.subscribe", "params": ["quantum-btc/0.1"] }),
    )
    .await
    {
        return SessionOutcome::Retry { msg: format!("send subscribe failed: {}", e), was_authorized: false };
    }

    let mut authorized = false;
    let mut jobs: VecDeque<JobRecord> = VecDeque::new();
    // submit request id -> job id
    let mut pending: HashMap<u64, (u64, bool)> = HashMap::new();
    let mut next_id: u64 = FIRST_SUBMIT_ID;

    loop {
        tokio::select! {
            line = lines.next_line() => {
                let line = match line {
                    Ok(Some(l)) => l,
                    Ok(None) => return SessionOutcome::Retry { msg: "pool closed the connection".to_string(), was_authorized: authorized },
                    Err(e) => return SessionOutcome::Retry { msg: format!("read error: {}", e), was_authorized: authorized },
                };
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if debug_enabled() {
                    tracing::info!("[DEBUG] Pool: <- {}", line);
                }
                let msg: Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!("[WARN] Pool: unparsable line ({}): {}", e, line);
                        continue;
                    }
                };

                // ---- notifications ----
                if let Some(method) = msg.get("method").and_then(Value::as_str) {
                    if method == "mining.notify" {
                        let params = msg.get("params").and_then(Value::as_array).cloned().unwrap_or_default();
                        match parse_notify(&params) {
                            Ok((record, template)) => {
                                // Pools repeat the same job (e.g. after a vardiff change). Same job and
                                // same share target: nothing to do. Same job, new share target: keep the
                                // job record, but restart the miners with the new target.
                                let same_job = jobs.back().map_or(false, |j| {
                                    j.job_id == record.job_id
                                        && j.previous_hash == record.previous_hash
                                        && j.merkle_root == record.merkle_root
                                        && j.timestamp == record.timestamp
                                });
                                let same_target = jobs.back().map_or(false, |j| j.share_target == record.share_target);
                                if !(same_job && same_target) {
                                    tracing::info!(
                                        "[INFO] Pool: {} height={} target={} share_target={} prev={}...",
                                        if same_job { "Share target changed (vardiff)," } else { "New job" },
                                        record.job_id, template.target, record.share_target, to_hex(&record.previous_hash[..6])
                                    );
                                    STATS.set_job(record.job_id, template.target, record.share_target);
                                    STATS.event("info", format!(
                                        "{} height={} share_target={}",
                                        if same_job { "Vardiff update" } else { "New job" },
                                        record.job_id, record.share_target
                                    ));
                                    if same_job {
                                        if let Some(last) = jobs.back_mut() {
                                            last.share_target = record.share_target;
                                        }
                                    } else {
                                        jobs.push_back(record);
                                        while jobs.len() > MAX_TRACKED_JOBS {
                                            jobs.pop_front();
                                        }
                                    }
                                    if cmd_tx.try_send(MinerCommand::Mine(template)).is_err() {
                                        tracing::warn!("[WARN] Pool: miner command queue is full, job dropped");
                                    }
                                }
                            }
                            Err(e) => tracing::warn!("[WARN] Pool: bad mining.notify ignored: {}", e),
                        }
                    } else if method == "mining.set_difficulty" {
                        // The share target travels inside mining.notify (7th param); this message
                        // is only informational here.
                        let d = msg.get("params").map(|p| p.to_string()).unwrap_or_default();
                        tracing::info!("[INFO] Pool: set_difficulty {}", d);
                        STATS.event("info", format!("Pool set_difficulty {}", d));
                    } else if method == "client.reconnect" {
                        return SessionOutcome::Retry { msg: "pool asked to reconnect".to_string(), was_authorized: authorized };
                    } else if method == "client.get_version" || method == "mining.ping" {
                        if let Some(idv) = msg.get("id").filter(|v| !v.is_null()) {
                            let result = if method == "mining.ping" { json!(true) } else { json!("quantum-btc/0.1") };
                            let _ = send_json(&mut wr, &json!({ "id": idv, "result": result, "error": Value::Null })).await;
                        }
                    } else if debug_enabled() {
                        tracing::info!("[DEBUG] Pool: unhandled method {}", method);
                    }
                    continue;
                }

                // ---- responses ----
                let id = match msg.get("id").and_then(Value::as_u64) {
                    Some(id) => id,
                    None => continue,
                };
                let error = msg.get("error").filter(|e| !e.is_null());
                let ok = msg.get("result").and_then(Value::as_bool) == Some(true) && error.is_none();

                if id == ID_SUBSCRIBE {
                    if error.is_some() {
                        return SessionOutcome::Fatal(format!("subscribe refused: {}", error.unwrap()));
                    }
                    tracing::info!("[INFO] Pool: Subscribed ({})", msg.get("result").map(|r| r.to_string()).unwrap_or_default());
                    if let Err(e) = send_json(
                        &mut wr,
                        &json!({ "id": ID_AUTHORIZE, "method": "mining.authorize", "params": [user, password] }),
                    )
                    .await
                    {
                        return SessionOutcome::Retry { msg: format!("send authorize failed: {}", e), was_authorized: false };
                    }
                } else if id == ID_AUTHORIZE {
                    if !ok {
                        return SessionOutcome::Fatal(format!(
                            "authorization refused for {} (error: {})",
                            user,
                            error.map(|e| e.to_string()).unwrap_or_else(|| "result=false".to_string())
                        ));
                    }
                    authorized = true;
                    tracing::info!("[INFO] Pool: Authorized. Waiting for jobs...");
                    STATS.set_link(true, true);
                    STATS.event("ok", "Authorized. Waiting for jobs".to_string());
                } else if let Some((job_id, is_block)) = pending.remove(&id) {
                    if ok {
                        stats.accepted += 1;
                        STATS.share_accepted();
                        if is_block {
                            STATS.block_found();
                            STATS.event("block", format!("BLOCK-LEVEL SHARE ACCEPTED (job {})", job_id));
                        } else {
                            STATS.event("ok", format!("Share accepted (job {})", job_id));
                        }
                        tracing::info!(
                            "[INFO] Pool: Share accepted! (job {}, accepted={} rejected={})",
                            job_id, stats.accepted, stats.rejected
                        );
                    } else {
                        stats.rejected += 1;
                        let reason = error
                            .map(|e| e.to_string())
                            .or_else(|| msg.get("result").map(|r| format!("result={}", r)))
                            .unwrap_or_else(|| "unknown".to_string());
                        let reason_clean = reason.trim_matches('"').to_string();
                        STATS.share_rejected(&reason_clean);
                        STATS.event("err", format!("Share rejected: {} (job {})", reason_clean, job_id));
                        tracing::warn!(
                            "[WARN] Pool: Share rejected: {} (job {}, accepted={} rejected={})",
                            reason, job_id, stats.accepted, stats.rejected
                        );
                    }
                }
            }

            found = share_rx.recv() => {
                let block = match found {
                    Some(b) => b,
                    None => return SessionOutcome::Fatal("miner result channel closed".to_string()),
                };
                if !authorized {
                    continue;
                }
                let job_id = jobs.iter().rev().find(|j| {
                    j.previous_hash == block.header.previous_hash
                        && j.merkle_root == block.header.merkle_root
                        && j.timestamp == block.header.timestamp
                }).map(|j| j.job_id);
                let job_id = match job_id {
                    Some(j) => j,
                    None => {
                        tracing::info!("[INFO] Pool: dropping a share for a job that is no longer tracked");
                        continue;
                    }
                };
                let hash = block.calculate_hash();
                let hash_hex = to_hex(&hash);
                let id = next_id;
                next_id += 1;
                let request = build_submit(id, user, job_id, block.header.nonce, &hash_hex);
                // The pool silently ignores shares for stale jobs (no reply); do not let
                // those entries pile up.
                if pending.len() > 1000 {
                    pending.clear();
                }
                let prefix = u64::from_be_bytes(hash[0..8].try_into().unwrap());
                let is_block = prefix <= block.header.target;
                pending.insert(id, (job_id, is_block));
                // Only the socket write happens here; the miners keep hashing
                // on their own threads and are never blocked by this.
                if let Err(e) = send_json(&mut wr, &request).await {
                    return SessionOutcome::Retry { msg: format!("send submit failed: {}", e), was_authorized: authorized };
                }
                if debug_enabled() {
                    tracing::info!("[DEBUG] Pool: share header_hex={}", header_hex(&block));
                    tracing::info!("[DEBUG] Pool: share hash={}", hash_hex);
                }
                if is_block {
                    tracing::info!("[INFO] Pool: *** This share also meets the NETWORK target: a real block! ***");
                }
                tracing::info!(
                    "[INFO] Pool: Share submitted (job {}, nonce {}, hash prefix {}...)",
                    job_id, block.header.nonce, &hash_hex[..16]
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(host: &str, port: u16, tls: bool) -> PoolEndpoint {
        PoolEndpoint { host: host.to_string(), port, tls }
    }

    #[test]
    fn parses_plain_and_tls_urls() {
        assert_eq!(parse_pool_url("stratum+tcp://192.0.2.10:3334"), Ok(ep("192.0.2.10", 3334, false)));
        assert_eq!(parse_pool_url("stratum+ssl://pool.example.org:3336"), Ok(ep("pool.example.org", 3336, true)));
        assert_eq!(parse_pool_url("STRATUM+TLS://Pool.Example.org:443/"), Ok(ep("Pool.Example.org", 443, true)));
        assert_eq!(parse_pool_url("pool.example.org:3333"), Ok(ep("pool.example.org", 3333, false)));
        assert_eq!(parse_pool_url("tcp://user:pw@host.net:1234/path?x=1"), Ok(ep("host.net", 1234, false)));
        assert_eq!(parse_pool_url("stratum+tcp://[::1]:3333"), Ok(ep("::1", 3333, false)));
    }

    #[test]
    fn rejects_bad_urls() {
        assert!(parse_pool_url("").is_err());
        assert!(parse_pool_url("stratum+tcp://host").is_err()); // no port
        assert!(parse_pool_url("stratum+tcp://host:notaport").is_err());
        assert!(parse_pool_url("http://host:80").is_err());
        assert!(parse_pool_url("stratum+tcp://:3333").is_err());
    }

    #[test]
    fn splits_worker_names() {
        assert_eq!(split_login("qbtc1abc.gpu0"), ("qbtc1abc".to_string(), Some("gpu0".to_string())));
        assert_eq!(split_login("qbtc1abc"), ("qbtc1abc".to_string(), None));
        assert_eq!(split_login("qbtc1abc."), ("qbtc1abc.".to_string(), None));
    }
}
