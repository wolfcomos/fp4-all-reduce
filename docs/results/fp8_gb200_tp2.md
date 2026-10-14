# GB200 TP2 FP8 AllReduce results

**Historical study:** these timings cover the original group128 implementation. A later [four-format comparison](fp8_group32_gb200_tp2.md) measures group32 and group128 kernels together at the same geometry; the original values below remain unchanged.

All formats use the same BF16 source tensors, shape/seed pairing and unquantized FP32 sum reference. Inputs are synthetic Gaussian samples with 32 outliers per source. Hidden size is 6144; source magnitudes are 1.0 and 0.5. These are operator measurements, not model-quality results.

**Format naming:** the measured FP8 implementation uses E4M3 values with one UE8M0 scale per 128 values. It is a custom group128 format, not standard group32 MXFP8. No standard MXFP8 AllReduce timing is included.

FP8 geometry: tile 2048, 152 CTAs, 7 teams. NVFP4 uses upstream L2 one-shot defaults with 16-value E2M1 blocks, E4M3 block scales and an FP32 row scale; this is not a best-configuration claim. BF16 uses out-of-place NCCL Sum. Software: PyTorch 2.13.0+cu130, CUDA 13.0, CuTe DSL 4.6.2, NCCL 2.31.2. GPU clocks were not locked.

| Rows | BF16 NCCL µs | FP8 group128 µs | NVFP4 L2 µs | FP8 speedup vs BF16 |
|---:|---:|---:|---:|---:|
| 1 | 11.722 | 5.869 | 6.352 | 1.995× |
| 4 | 12.435 | 6.221 | 6.931 | 1.999× |
| 16 | 12.458 | 6.227 | 6.701 | 1.998× |
| 64 | 13.606 | 6.874 | 6.938 | 1.978× |
| 1024 | 49.120 | 18.784 | 14.035 | 2.615× |
| 4096 | 116.458 | 52.717 | 33.968 | 2.208× |
| 16384 | 397.354 | 182.653 | 106.166 | 2.176× |

Timings include the full BF16-to-BF16 call. Each sweep uses 200 paired rank-max samples, 10 collectives/sample, after 20 warmups. Each table latency is the median of three fresh sweep medians; the JSON retains each sweep and min/max. Speedups are medians of three paired within-sweep ratios, not ratios of aggregate medians.

| Rows | BF16 relative L2 | FP8 group128 relative L2 | NVFP4 relative L2 |
|---:|---:|---:|---:|
| 1 | 0.0436%–0.0460% | 0.6584%–0.6749% | 6.3314%–6.4942% |
| 4 | 0.0813%–0.0825% | 1.2084%–1.2305% | 6.7526%–7.0009% |
| 16 | 0.1276%–0.1280% | 1.9004%–1.9149% | 8.0456%–8.1539% |
| 64 | 0.1596%–0.1605% | 2.4004%–2.4072% | 8.9482%–9.0332% |
| 1024 | 0.1765%–0.1766% | 2.6424%–2.6462% | 9.4749%–9.4824% |
| 4096 | 0.1775%–0.1776% | 2.6588%–2.6597% | 9.5058%–9.5078% |
| 16384 | 0.1778%–0.1778% | 2.6632%–2.6638% | 9.5140%–9.5163% |

| Rows | BF16 max absolute error | FP8 group128 max absolute error | NVFP4 max absolute error |
|---:|---:|---:|---:|
| 1 | 0.015625 | 0.171875–0.1875 | 2.45312–2.48828 |
| 4 | 0.015625 | 0.1875–0.203125 | 2.33984–2.48047 |
| 16 | 0.015625 | 0.1875–0.203125 | 2.39844–2.46094 |
| 64 | 0.015625 | 0.234375–0.328125 | 2.42578–2.49609 |
| 1024 | 0.015625 | 0.375 | 2.32422–2.45312 |
| 4096 | 0.015625 | 0.359375–0.375 | 2.44141–2.48438 |
| 16384 | 0.015625 | 0.375 | 2.41797–2.4375 |

For original BF16 sources, the reference is their rank-ordered FP32 sum `r`; output error is `e=float(y)-r`. Relative L2 is `sqrt(mean(e²)/mean(r²))`, RMSE is `sqrt(mean(e²))`, and max absolute error is `max(abs(e))`. Squares and means use FP32 PyTorch arithmetic. BF16 NCCL therefore has nonzero error against the unrounded FP32 reference, despite passing its exact BF16-rounded FP32-addition gate.

Accuracy ranges are min/max over input sets 0, 1, 2, using `seed=1000+17*rows+7777*input_set` for both source ranks. Seed-level metrics were identical across all three sweeps; graph and post-timing checks matched their eager seed. Each seed appears once in the JSON, which includes RMSE, source hashes and sweep hashes. Relative L2 values are not averaged or pooled across shapes.

All declared finite, rank, graph, independent codec/wire, frozen-baseline parity and reuse gates passed. Full-output error metrics are separate from sampled complete-group FP8 codec checks. The supported finite-output domain requires finite decoded terms and FP32 partial sums plus BF16-representable output; these checks do not qualify arbitrary inputs, model accuracy or serving throughput.

[Machine-readable timings, ranges and seed-level accuracy](fp8_gb200_tp2.json).

Upstream source: [fans-nv/fp4-all-reduce](https://github.com/fans-nv/fp4-all-reduce), commit `2878bf50e20241d55500c2df76b87f0a81ef1c0f`.
