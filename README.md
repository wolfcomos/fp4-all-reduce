# fp4-allreduce

Fused NVFP4-transport tensor-parallel AllReduce for VR200 (sm_107a) and GB200 (sm_100a), built for MiniMax-M3 prefill (hidden 6144, BF16 activations, 1K-16K rows per call).

One kernel per call encodes each rank's BF16 partial to NVFP4 (E2M1 values + E4M3 scale per 16 + FP32 row scale, 3.56x fewer bytes than BF16), moves it over NVLink, decodes and sums in FP32 in rank order, and writes BF16:

- **Push design.** Each rank TMA bulk-stores its encoded rows straight into the peers' receive buffers (256 B NVLink packets).
- **Lamport-in-data readiness.** Receive buffers are pre-filled with values no encoder can produce (E2M1 -0 nibble, scale byte `0xFF`); receivers poll the data itself. No flags and no system-scope fences on the hot path. The call index lives on the device, so the kernel is CUDA-graph capturable.
- **TP2 one-shot, TP4 two-shot** (row owner `row % 4` reduces, re-encodes and broadcasts).
- **Optional fused epilogue** (TP2): residual add + Gemma RMSNorm, BF16 out (NVFP4 / MXFP8 out also available).
- **NVFP4-input mode.** If the producing GEMM's epilogue already emits NVFP4 partials (static per-rank scale), the AllReduce skips the encode.

A pull variant family (peers TMA-load from each other's windows) is included as the measured comparison; it loses on VR200 at every tier.

## GB200 TP2: FP8 transport

This fork adds CuTe DSL AllReduce kernels for [standard MXFP8 group32](harness/cute_fp8/mxfp8_lamport_batched.py) and [custom FP8 group128](harness/cute_fp8/ue8m0_lamport_batched.py) alongside the original NVFP4 kernels. See [implementation and quickstart](docs/FP8.md), [the four-format measurements](docs/results/fp8_group32_gb200_tp2.md), and [timings plus numerical data](docs/results/fp8_group32_gb200_tp2.json).

**Measured scope:** two GB200 GPUs over NVLink, hidden 6144, BF16 inputs and outputs. Quantization, communication, decode/reduction and synchronization are all timed. NCCL uses out-of-place BF16 Sum with the recorded environment settings, no explicit algorithm/channel/protocol overrides and no tuning search. NVFP4 is upstream TP2 L2 one-shot with default tuning. The upstream source is `2878bf50e20241d55500c2df76b87f0a81ef1c0f`; the VR200 results below remain a separate study.

| Rows | BF16 NCCL (µs) | MXFP8 group32 (µs) | FP8 group128 (µs) | NVFP4 L2 (µs) | MXFP8 speedup vs BF16 |
| --: | --: | --: | --: | --: | --: |
| 1 | 11.174 | 5.805 | 5.990 | 6.243 | 1.925× |
| 4 | 11.168 | 5.987 | 6.064 | 6.883 | 1.865× |
| 16 | 11.200 | 5.997 | 6.259 | 6.477 | 1.863× |
| 64 | 12.854 | 6.269 | 6.646 | 6.883 | 2.052× |
| 1024 | 49.258 | 17.504 | 17.738 | 14.019 | 2.814× |
| 4096 | 118.934 | 53.066 | 52.070 | 33.917 | 2.242× |
| 16384 | 398.413 | 189.946 | 181.606 | 106.010 | 2.098× |

Latencies are the median of three fresh sweep medians. Each sweep uses 200 paired rank-max samples after 20 warmups, with 10 complete calls per CUDA Graph replay. Speedups are medians of the three paired within-sweep ratios, not ratios of the aggregate medians. GPU clocks were not locked. Software: PyTorch 2.13.0+cu130, CUDA 13.0, CuTe DSL 4.6.2 and NCCL 2.31.2.

Both FP8 kernels use **tile 2048, 152 CTAs and 7 teams** in this same-run comparison. This geometry came from the [earlier group128 study](docs/results/fp8_gb200_tp2.md); group32 was **not independently tuned**. Standard MXFP8 uses E4M3 values and one E8M0 scale per 32 values, with exponent floor -127. Custom group128 uses one UE8M0 scale per 128 values, with floor -120. Both use rank-ordered FP32 multiply/add and BF16 output rounding, with no second output quantization.

**Synthetic numerical error at 16384 × 6144, TP2.** All four rows now use actual AllReduce outputs on the same three seeded BF16 input pairs and the same unquantized FP32 sum reference. Inputs are Gaussian samples with 32 outliers per source. Ranges below span the three seeds across the three performance sweeps.

| Format | Relative L2 error (%) | Max absolute error | Evidence |
| --- | --: | --: | --- |
| BF16 NCCL | 0.17780–0.17782 | 0.015625 | Measured AllReduce output |
| Standard MXFP8 E4M3 + E8M0, group32 | 2.66322–2.66380 | 0.375 | Measured AllReduce output |
| FP8 E4M3 + UE8M0, group128 | 2.66322–2.66380 | 0.375 | Measured AllReduce output |
| NVFP4, block16 + row scale | 9.51402–9.51633 | 2.41797–2.43750 | Measured AllReduce output |

Group32 uses [NVIDIA TensorRT’s round-up E8M0 scale recipe](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html#mx-compliant-dynamic-quantization), validated against independent full-output GPU and sampled NumPy references. At this shape its FP32-reduced summary metrics match group128; this does not imply elementwise equivalence or equal error on other workloads. The [earlier accuracy-only group32 probe](docs/results/fp8_format_accuracy.md) remains available separately.

For reference `r` and BF16 output `y`, relative L2 is `100*sqrt(mean((float(y)-r)^2)/mean(r^2))`; max absolute error is `max(abs(float(y)-r))`. The harness computes these full-output metrics with FP32 arithmetic. NCCL passed a full-output bitwise check against BF16-rounded FP32 addition. Both FP8 kernels passed codec/wire, finite, rank, changed-input graph, stress and triple-buffer reuse checks. Group32 also passed full-output oracle and actual-kernel subnormal/signed-zero edge checks; group128 retained frozen-baseline parity. A relative L2 percentage is vector error, **not task-accuracy loss**; no model-accuracy or serving/end-to-end speedup is claimed.

Each FP8 workspace keeps three receive buffers. At the maximum shape above, symmetric allocation is **311,430,720 bytes (297.003 MiB) per rank for group32** and **304,352,832 bytes (290.253 MiB) for group128**, including device counters and excluding input/output tensors. Capacity scales with the configured maximum input size. See [FP8 implementation notes](docs/FP8.md) for the numerical domain and workspace lifecycle.

## Headline results (VR200, measured)

CUDA graph, median of max over ranks, µs per call. HBM locked at 4752 MHz. Every cell passes the bitwise correctness gate. Repeat spread <= 1.9%. Details: [docs/RESULTS.md](docs/RESULTS.md).

**AllReduce only, TP2 (one-shot) vs tuned BF16 NCCL in the same harness**

| Rows | Tuned BF16 NCCL (µs) | Push, BF16 in (µs) | Push, NVFP4 in (µs) | Speedup, NVFP4 in |
| --: | --: | --: | --: | --: |
| 1K | 30.9 | 10.8 | **8.6** | **3.59x** |
| 2K | 50.5 | 15.2 | **12.5** | **4.04x** |
| 4K | 77.1 | 21.9 | **18.4** | **4.19x** |
| 8K | 125.3 | 34.4 | **30.8** | **4.07x** |
| 16K | 216.6 | 60.7 | **55.4** | **3.91x** |

**AllReduce only, TP4 (two-shot) vs tuned BF16 NCCL**

| Rows | Tuned BF16 NCCL (µs) | Push, BF16 in (µs) | Push, NVFP4 in (µs) | Speedup, NVFP4 in |
| --: | --: | --: | --: | --: |
| 1K | 40.2 | 19.8 | **16.4** | **2.45x** |
| 2K | 60.7 | 25.6 | **21.2** | **2.86x** |
| 4K | 102.8 | 34.8 | **30.7** | **3.36x** |
| 8K | 177.0 | 56.2 | **50.5** | **3.50x** |
| 16K | 306.9 | 103.8 | **98.3** | **3.12x** |

Tuned BF16 = best of NCCL Ring 32/64 channels x LL128/Simple x registered/unregistered buffers; NCCL defaults are 1.2-1.8x slower than that.

**M3 block, TP2: AllReduce + residual + RMSNorm, BF16 out (the production path)**

| Rows | vLLM 0.30 today (µs) | Best BF16 path (µs) | Fused, BF16 in (µs) | Fused, NVFP4 in (µs) | NVFP4 in vs vLLM / vs best BF16 |
| --: | --: | --: | --: | --: | --: |
| 1K | 22.6 | 22.6 | 16.9 | **14.1** | 1.60x / 1.60x |
| 2K | 45.9 | 45.9 | 22.9 | **21.8** | 2.11x / 2.11x |
| 4K | 120.2 | 89.1 | 34.9 | **34.9** | 3.44x / 2.55x |
| 8K | 169.3 | 167.4 | 61.2 | **55.2** | 3.07x / 3.03x |
| 16K | 311.4 | 295.9 | 113.4 | **96.8** | 3.22x / 3.06x |

vLLM today: FlashInfer fused one-shot (<= 2K), FlashInfer two-shot (4K), NCCL + separate norm (>= 8K, past the 64 MiB fusion cap). Best BF16: FlashInfer one-shot (<= 4K) or registered NCCL + norm (>= 8K).

**Per prefill step (121 AR + norm blocks per step; block time x 121, from measured block times)**

| Rows per step | vLLM today (ms) | Fused, BF16 in (ms) | Fused, NVFP4 in (ms) | Saved, NVFP4 in (ms) |
| --: | --: | --: | --: | --: |
| 1K | 2.73 | 2.04 | 1.71 | 1.03 |
| 4K | 14.54 | 4.22 | 4.22 | 10.32 |
| 16K | 37.68 | 13.72 | 11.71 | 25.97 |

**Projected TTFT (PROJECTION, not measured in serving)**: measured round-1 production TTFT (TP2, concurrency 1, p50) minus the per-step saving, assuming the block stays unoverlapped on the compute stream and, for NVFP4 in, a free GEMM-epilogue quantization.

| ISL | Measured TTFT (ms) | Projected, fused BF16 in (ms) | Projected, fused NVFP4 in (ms) |
| --: | --: | --: | --: |
| 4K | 85.1 | 74.8 (-12.1%) | 74.8 (-12.1%) |
| 16K | 284.0 | 260.0 (-8.4%) | 258.0 (-9.1%) |
| 64K | 1117.6 | 1021.8 (-8.6%) | 1013.7 (-9.3%) |

Close to the hardware: at 16K the TP2 push transport runs at 1076 GB/s, 90% of the ~1197 GB/s payload ceiling for 256 B packets on these links (36 x 41 GB/s per direction; ~1300 GB/s wire ceiling). See [docs/NVLINK.md](docs/NVLINK.md).

## Status and limitations

| Item | Status |
| --- | --- |
| Accuracy | **Unvalidated.** Synthetic data only: rel-L2 vs exact FP32 sum 0.095 (one-shot) / 0.133 (two-shot), vs 0.0018 for BF16 transport. Real M3 per-layer partials and a task eval are required before adoption ([docs/INTEGRATION.md](docs/INTEGRATION.md) section 10) |
| NVFP4-input mode | Assumes a GEMM epilogue change: the row-parallel GEMM must emit NVFP4 (3456 B rows, calibrated static per-tensor scale per rank, -0 canonicalized). No such producer exists yet; the benchmark emulates it outside the timed region |
| vLLM integration | None yet. Kernels run in a standalone C++ benchmark. Needed work: [docs/INTEGRATION.md](docs/INTEGRATION.md) |
| Fused residual + RMSNorm | TP2 only. Multiplies the BF16-rounded residual sum, not FP32 (<= one extra 2^-9 rounding). NVFP4/MXFP8 outputs use linear scale layouts and were not bit-compared to FlashInfer. Built on an earlier push build (AR part 67.2 vs 60.7 µs at 16K) |
| Fixed grid | Per-CTA call state requires every kernel in a context to use the same grid (`sm_count`) |
| Pull window mode | Requires that two consecutive AR calls never use the same input slot (reuse distance >= 2); breaks for `K ≡ 1 (mod 5)` AR calls per graph with the default rotation. Push kernels have no such precondition |
| Window memory | ~1 GB per rank at TP2 16K rows (symmetric NCCL window, 3-buffer rotation) |
| Not done | TP8 across two nodes; NVLS wire counters; TP4 fused norm; one kernel per input format (small-tier instruction-fetch fix, [docs/PROFILING.md](docs/PROFILING.md)) |

## Quickstart

Requirements: CUDA >= 13.4 for `sm_107a` (VR200 builds used nvcc 13.5) or CUDA 13.0 for `sm_100a`; NCCL >= 2.31 (symmetric windows, `ncclGetLsaDevicePointer`); 2 or 4 GPUs on one NVLink domain.

The commands below come from the repository's build, test and run scripts. They have not yet been exercised end to end from this repository on hardware (the numbers above come from the frozen campaign builds; `make verify` checks that the packaged kernel sources match them).

**Build**

```bash
make verify                         # md5-check kernels/include against the frozen measured sources
make ARCH=107a NCCL_HOME=/usr       # VR200 -> build/<tag>/fp4ar_bench (+ fp4ar_norm_bench), build/latest
make ARCH=100a NCCL_HOME=/path/nccl # GB200
make check                          # compile every variant offline, no link (x86 is fine)
```

On a Slurm cluster (scripts never allocate; they `srun --overlap` into an existing job): copy `scripts/cluster.env.example` to `scripts/cluster.env`, set `JOB`, `IMG`, `WORK`, then from the login node:

```bash
CLUSTER=vr200 scripts/sync.sh                          # copy sources to $WORK/src
scripts/clk.sh lock <node>                              # HBM 4752 MHz with readback (reset: clk.sh reset <node>)
scripts/step.sh <node> 0 600 build_all.sh <tag>         # build in the container
scripts/step.sh <node> 0 300 codec_test.sh              # codec vs CPU oracle; must PASS
```

**Run**

```bash
# one cell: TP2, push kernel, NVFP4 input, gate then time fused/xport/compute/empty, eager + graph, with the BF16 baseline
scripts/step.sh <node> 0,1 840 run_cell.sh my-cell -- \
  --tp 2 --variant L2q --input nvfp4 --rows 1024,2048,4096,8192,16384 \
  --check --mode fused,xport,compute,empty --graph 0,10 --algo 1 --bf16
# TP4 two-shot
scripts/step.sh <node> 0,1,2,3 840 run_cell.sh my-tp4 -- --tp 4 --variant L4 --algo 2 --check --mode fused --graph 10
python3 scripts/summarize.py $WORK/runs/*.jsonl --graph 10
```

`fp4ar_bench --list` prints the linked variants. Variants: `L2`/`L2q` (TP2 push, BF16/NVFP4 input), `L4`/`L4q` (TP4 push), `P`, `Pq`, `P4`, `P4t` (pull), `r1` (round-1 control). `--input bf16|nvfp4|nvfp4-window`, `--algo 1` one-shot / `2` two-shot, `--tune a,b,c,d` knobs (defaults are tuned).

**Gate** (run both; a gate that cannot go red proves nothing)

```bash
scripts/step.sh <node> 0,1,2,3 3600 gate.sh             # every variant x TP x input x tier, plus the fused-norm gate
scripts/step.sh <node> 0,1 900 negative_control.sh      # stale and bit-flip injections must FAIL
```

## Repository layout

| Path | Contents |
| --- | --- |
| `include/` | `fp4_codec.cuh` (codec spec v2, sentinels), `codec_fast.cuh` (fast row encode/decode building blocks), `fp4ar_api.h` (variant interface) |
| `kernels/push/` | TP2 one-shot (`tp2_l2.cu`: `L2`, `L2q`) and TP4 two-shot (`tp4_l4.cu`: `L4`, `L4q`) push kernels |
| `kernels/pull/` | pull kernels `P`, `Pq`, `P4`, `P4t`, `P4q` (comparison) |
| `kernels/fused_norm/` | AR + residual + RMSNorm epilogue kernel (`lnorm.cu`, `lnorm_epi.cuh`) |
| `kernels/control/` | round-1 fused kernel (`r1`), the control |
| `kernels/verify_origin.sh` | md5 check against the measured builds |
| `harness/` | `fp4ar_bench.cu` (driver, NCCL BF16 baseline, gate, timing, JSON) and `ref.cuh` (FP32 reference) |
| `tests/` | `gate.sh`, `negative_control.sh`, `gate_fused_norm.sh`, `codec_test.{cu,sh}`, `gate_check.py` |
| `scripts/` | build, cluster profile, sync, step/lock wrappers, clock lock, cell driver, `summarize.py` |
| `bench/` | `micro/` (NVLink bandwidth, latency, tear, TMA validity, fences), `link/` (NVLink counters, nccl-tests), `codec/`, `fused_norm/` (driver), `block/` (vLLM/FlashInfer block timing), `baselines/` |
| `docs/` | [DESIGN](docs/DESIGN.md), [RESULTS](docs/RESULTS.md), [NVLINK](docs/NVLINK.md), [PROFILING](docs/PROFILING.md), [INTEGRATION](docs/INTEGRATION.md), [HISTORY](docs/HISTORY.md) |
