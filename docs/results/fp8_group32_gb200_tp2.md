# GB200 TP2: measured MXFP8 group32 AllReduce

This follow-up measures the actual group32 communication kernel alongside group128 FP8, BF16 NCCL and upstream NVFP4 L2 in the same allocation. All formats use BF16 inputs and outputs with hidden size 6144. The complete quantize, communicate, decode/reduce and synchronize path is timed.

Both FP8 kernels use tile 2048, 152 CTAs and 7 teams. This geometry was inherited from the earlier group128 study; group32 was not independently tuned. The transport is TMA peer transfer with Lamport payload polling and three receive buffers. This run does not use Switch reduction.

| Rows | BF16 NCCL (µs) | MXFP8 group32 (µs) | FP8 group128 (µs) | NVFP4 (µs) | MXFP8 speedup vs BF16 | Group32 latency vs group128 |
| --: | --: | --: | --: | --: | --: | --: |
| 1 | 11.174 | 5.805 | 5.990 | 6.243 | 1.925× | -3.10% |
| 4 | 11.168 | 5.987 | 6.064 | 6.883 | 1.865× | -1.27% |
| 16 | 11.200 | 5.997 | 6.259 | 6.477 | 1.863× | -3.78% |
| 64 | 12.854 | 6.269 | 6.646 | 6.883 | 2.052× | -5.68% |
| 1024 | 49.258 | 17.504 | 17.738 | 14.019 | 2.814× | -1.32% |
| 4096 | 118.934 | 53.066 | 52.070 | 33.917 | 2.242× | +1.82% |
| 16384 | 398.413 | 189.946 | 181.606 | 106.010 | 2.098× | +4.59% |

Latencies are medians of three fresh-process sweep medians. Each sweep has 200 paired max-over-rank samples after 20 warmups, with 10 complete calls per CUDA Graph replay and 100 stress calls per cell. Ratios are medians of the three within-sweep ratios. Clocks were not locked. NCCL used the recorded settings without an algorithm/channel tuning search; NVFP4 used upstream TP2 L2 default tuning.

Group32 measured 1.27–5.68% lower latency at 1–1024 rows, 1.82% higher at 4096 rows, and 4.59% higher at 16384 rows. These are observed differences with this configuration, without a causal profiling claim. At the largest shape it is 2.098× faster than the paired BF16 NCCL baseline.

Software: PyTorch 2.13.0+cu130, CUDA 13.0, CuTe DSL 4.6.2, NCCL 2.31.2. The loaded NCCL identity was verified for PyTorch and the native bridge. The [JSON](fp8_group32_gb200_tp2.json) contains each sweep median and range, paired ratios, all 84 distinct shape/format/seed error records, source hashes, workspace metadata and the original report hashes.

## Numerical error

At 16384 × 6144, all four rows below are **measured AllReduce outputs**, compared with the same unquantized FP32 sum. Ranges cover three seeded BF16 input pairs. All repeated eager/graph/post-timing metrics agree across the three sweeps.

| Format | Relative L2 error (%) | Max absolute error |
| --- | --: | --: |
| BF16 NCCL | 0.17780–0.17782 | 0.015625 |
| MXFP8 E4M3 + E8M0, group32 | 2.66322–2.66380 | 0.375 |
| FP8 E4M3 + UE8M0, group128 | 2.66322–2.66380 | 0.375 |
| NVFP4 upstream L2 | 9.51402–9.51633 | 2.41797–2.43750 |

Relative L2 is `sqrt(mean((float(y)-r)**2)/mean(r**2))`, reported above as a percentage; max absolute error is `max(abs(float(y)-r))`. Tensor differences, squares and means use FP32. Inputs are the common Gaussian-plus-32-outliers generator; see the JSON for seeds and formulas. Similar aggregate group32/group128 errors on this dataset do not imply elementwise equivalence or equal error on other workloads.

MXFP8 uses E4M3 with one E8M0 scale per 32 values, `e=max(-127,ceil(log2(amax/448)))`. Scale byte 0 decodes to `2**-127`. Source dequantization and rank-ordered reduction use separate FP32 multiply/add, then BF16 output rounding, without a second FP8 quantization. Group128 retains its custom 128-value blocks and -120 exponent floor.

## Validation and workspace

The partial-tail smoke and all three formal sweeps passed. Group32 checks include a full-output bitwise GPU numerical reference, independent NumPy sampled complete-group codec and received-wire checks, actual BF16 subnormal/code-0/signed-zero edge fixtures, eager and changed-input graph replay, rank agreement, stress/rank skew and alternating-shape triple-buffer reuse. Group128 retains its independent oracle and frozen implementation parity checks. These are bounded operator checks, not the full upstream protocol suite or model evaluation.

At the maximum capacity, symmetric workspace is 311,430,720 bytes (297.003 MiB) per rank for group32 and 304,352,832 bytes (290.253 MiB) for group128, excluding input/output tensors. Each 2048-value tile transports 2112 versus 2064 bytes (+2.33%). The byte difference alone does not establish the cause of the observed latency difference.

See [build, reproduction and API instructions](../FP8.md). Historical [group128 timing](fp8_gb200_tp2.md) and [accuracy-only group32 reference](fp8_format_accuracy.md) remain separate artifacts with their original numbers.
