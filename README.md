# QBTC miner fork: Stratum pool, CUDA and OpenCL

A fork of the original **Quantum Bitcoin (QBTC)** node/miner by
**[Q-Jack-core](https://github.com/Q-Jack-core/quantum-btc)**, adapted for standalone
mining with a Stratum pool client and an OpenCL backend.

This fork, maintained by **[GodConsole](https://github.com/GodConsole)**, adds:

* **Stratum pool client** (`src/stratum.rs`): plain TCP and TLS (`stratum+ssl://`), vardiff,
  automatic reconnect. `mining.notify` numbers are accepted as JSON numbers or numeric strings.
* **OpenCL backend** (`cuda/qbtc_sha256_miner.cl`, `cuda/qbtc_sha256_miner_opencl.c`) for AMD,
  Intel and NVIDIA cards, next to the original CUDA backend.
* Runtime statistics (`src/stats.rs`), `--bench-cuda`, `--max-power`, solo auto-start via `--address`.

The miner has **no developer fee**. 100 % of every coinbase reward — pool or solo — goes to the
address passed with `--address`. If this fork saves you time, a small donation is appreciated.

## Credits

| Role | Name | Link |
|---|---|---|
| **Original author** of `quantum-btc` | Q-Jack-core | https://github.com/Q-Jack-core/quantum-btc |
| **Fork maintainer** (Stratum client, OpenCL backend, dashboard, bug fixes) | GodConsole | https://github.com/GodConsole |

Issues, pull requests, or contributions related to the original node, consensus engine, or
base mining code should go to the upstream repository. This fork focuses on the standalone
miner experience.

## Donate

At start-up the miner prints this table:

```
+=================================================================================+
|                      QBTC MINER FORK  -  SUPPORT THE FORK                   |
+=================================================================================+
| Project       : QBTC miner: CUDA + OpenCL + Stratum pool client                 |
| Fork by       : GodConsole                                                      |
| GitHub        : https://github.com/GodConsole                                   |
| Donate (QBTC) : qbtc16udq46jjtmt42a46gazdnva8lavhd60lyttwmz8muujruu45fjuqpnkw9v |
+=================================================================================+
```

| | |
|---|---|
| Upstream author | [Q-Jack-core](https://github.com/Q-Jack-core/quantum-btc) |
| Fork by | [GodConsole](https://github.com/GodConsole) |
| Donate (QBTC) | `qbtc16udq46jjtmt42a46gazdnva8lavhd60lyttwmz8muujruu45fjuqpnkw9v` |

> **Fork support is voluntary and at the maintainer's discretion.** The donation address above
> belongs to **GodConsole**, the fork maintainer — not to the original author (Q-Jack-core).
> Nothing is required. The miner works the same whether you donate or not. Bug reports and pull
> requests go to https://github.com/GodConsole (this fork), not to the upstream repository.

## Quick start (Windows)

Put `quantum-btc.exe` and the `.bat` files in one folder, then edit the CONFIG block at the top
of the file you use. The `ADDRESS` placeholder must be replaced with your own `qbtc1...` address,
otherwise the script refuses to start.

| File | What it does |
|---|---|
| `pool-start.bat` | Mine in a pool (`POOL_URL`, `ADDRESS`, `WORKER`). Restarts the miner if it crashes |
| `solo-start.bat` | Run a node and mine solo. Set `MINE=0` to run the node without mining |
| `node-start.bat` | Run only the node. Nothing is mined |

### Starting only the node (no mining)

Solo mining starts automatically when `--address` is passed. Without it the node only syncs the
chain and opens the interactive console:

```
quantum-btc.exe --datadir .\qbtc_data --port 8000
```

You can start and stop mining later from that console:

```
auto_mine start <address>
auto_mine stop
```

## Command line

```
# pool (CUDA, or OpenCL automatically if no NVIDIA device is found)
quantum-btc --stratum stratum+ssl://pool.example.org:3336 --address qbtc1... --worker rig1 --backend gpu

# force OpenCL
quantum-btc --stratum stratum+ssl://pool.example.org:3336 --address qbtc1... --backend gpu --gpu-api opencl
# shorthand: --backend opencl

# solo mining
quantum-btc --datadir ./qbtc_data --port 8000 --address qbtc1...

# benchmark
quantum-btc --bench-cuda 20 --gpu-api opencl
```

## Environment variables

| Variable | Meaning |
|---|---|
| `QBTC_MINER_BACKEND` | `cpu` (default), `gpu`/`cuda`, `opencl`/`cl`, `hybrid` (CPU + GPU) |
| `QBTC_GPU_BACKEND` | `cuda` or `opencl`. Default: CUDA, OpenCL as fallback if no CUDA device |
| `QBTC_GPU_DEVICES` | Device ids, e.g. `0,1` (`QBTC_CUDA_DEVICES` still works) |
| `QBTC_CUDA_BATCH_SIZE` | Hashes per kernel launch (default 100000000). Lower it on weak or display GPUs |
| `QBTC_CL_BLOCKS`, `QBTC_CL_THREADS` | OpenCL launch size (work-groups, work-group size) |
| `QBTC_CL_INCLUDE_CPU=1` | Also list CPU OpenCL devices (PoCL, Intel CPU runtime); default: GPUs only |
| `QBTC_POOL_DEBUG=1` | Log all pool traffic |
| `QBTC_POOL_SUBMIT` | `num` (default), `hex`, `full`: `mining.submit` dialect |
| `QBTC_POOL_TLS_INSECURE=1` | Accept an invalid pool certificate (unsafe) |
| `QBTC_POOL_ANY_USER=1` | Skip the payout address check (pools with other account names) |

## Build

Needs a Rust toolchain. CUDA and OpenCL are both optional: whatever cannot be built is replaced
by a stub and the miner falls back to the CPU.

* CUDA: install the CUDA Toolkit (`CUDA_PATH` is detected, or set it).
* OpenCL: `CL/cl.h` and the OpenCL import library are taken from the CUDA Toolkit if present.
  Without it set `OPENCL_INCLUDE` (directory containing `CL/cl.h`, e.g. Khronos OpenCL-Headers)
  and `OPENCL_LIB` (directory containing `OpenCL.lib` / `libOpenCL.so`).
* To ship one binary that also starts on machines without the CUDA runtime, build with
  `QBTC_CUDART_STATIC=1`.

```
cargo build --release
```

At run time only the GPU driver is needed (it provides `OpenCL.dll` / `libOpenCL.so`).

## Added or changed compared with the upstream sources

Compared with [`Q-Jack-core/quantum-btc`](https://github.com/Q-Jack-core/quantum-btc):

`Cargo.toml`, `build.rs`, `src/lib.rs`, `src/main.rs`, `src/miner.rs`, `src/gpu_miner.rs`,
`src/stratum.rs` (new), `src/stats.rs` (new), `cuda/qbtc_sha256_miner.cl` (new),
`cuda/qbtc_sha256_miner_opencl.c` (new), `cuda/qbtc_sha256_miner.cu`, the `.bat` launchers.


## Tested configurations

This fork has been built and tested **only on Windows** with the following hardware:

| Hardware | Backend | Status |
|---|---|---|
| NVIDIA RTX 2060 SUPER (Turing, 8 GB) | CUDA, OpenCL | ✅ tested, works |
| NVIDIA RTX 2050 Mobile (Ampere, 4 GB) | CUDA, OpenCL | ✅ tested, works |
| Intel Core i5-11260H — Intel UHD Graphics (Tiger Lake) | OpenCL | ✅ tested, works (low hashrate) |
| Intel Core i5-11260H — CPU only | CPU | ✅ tested, works |

Everything else — **AMD GPUs, other Intel GPUs, older NVIDIA cards (Pascal / Maxwell),
Apple Silicon, Linux, macOS** — has **not** been tested. It may work; it may not. The
maintainer has no access to that hardware, so no claim is made either way.

If you run this fork on untested hardware, please open an issue with:
- OS and version
- GPU model and driver version
- `--bench-cuda 20` output
- whether the pool logs any `failed CPU re-verification` warnings

Real-world reports are the only way this table will grow.

## License

Distributed under the same license as the upstream `quantum-btc` sources. See `LICENSE`.
