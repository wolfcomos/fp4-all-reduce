# Four-format synthetic accuracy on GB200 TP2

BF16 input/output, hidden size 6144, seven row counts and three identical native CUDA-generated inputs per shape. The FP32 reference sums the original BF16 sources in rank order with a +0 seed. All 84 format/input/shape records passed on both ranks.

MXFP8 group32 is a **local GPU numerical reference with no communication or timing**. BF16 NCCL, group128 FP8 and NVFP4 use actual AllReduce outputs. Their 63 seed-level metric records exactly match [the performance artifact](fp8_gb200_tp2.json).

Relative L2 is `sqrt(mean((float(output)-reference)^2) / mean(reference^2))`; RMSE is `sqrt(mean(error^2))`, and max absolute error is `max(abs(error))`. Squares and means use FP32 PyTorch arithmetic. Percentages below are relative L2 × 100, not task accuracy. Ranges are min/max over three seeds, without averaging ratios or pooling shapes.

| Rows | BF16 NCCL rel. L2 (%) | MXFP8 group32 ref. (%) | FP8 group128 (%) | NVFP4 L2 (%) |
|---:|---:|---:|---:|---:|
| 1 | 0.043591–0.045965 | 0.658438–0.674891 | 0.658438–0.674891 | 6.331401–6.494218 |
| 4 | 0.081329–0.082488 | 1.208401–1.230536 | 1.208401–1.230536 | 6.752582–7.000923 |
| 16 | 0.127618–0.127971 | 1.900384–1.914906 | 1.900384–1.914906 | 8.045613–8.153859 |
| 64 | 0.159554–0.160474 | 2.400417–2.407235 | 2.400417–2.407235 | 8.948240–9.033236 |
| 1024 | 0.176537–0.176648 | 2.642393–2.646234 | 2.642393–2.646234 | 9.474862–9.482417 |
| 4096 | 0.177527–0.177607 | 2.658820–2.659659 | 2.658820–2.659659 | 9.505774–9.507768 |
| 16384 | 0.177799–0.177819 | 2.663224–2.663796 | 2.663224–2.663796 | 9.514019–9.516327 |

At rows=16384:

| Format | Relative L2 (%) | Max absolute error |
|---|---:|---:|
| BF16 NCCL | 0.177799–0.177819 | 0.015625–0.015625 |
| MXFP8 E4M3 + E8M0, group32 reference | 2.663224–2.663796 | 0.375–0.375 |
| FP8 E4M3 + UE8M0, group128 | 2.663224–2.663796 | 0.375–0.375 |
| NVFP4 upstream L2 | 9.514019–9.516327 | 2.4179688–2.4375 |

The group32 reference follows NVIDIA TensorRT’s round-up recipe: `e=max(-127,ceil(log2(amax/448)))`, `scale=2^e`, E8M0 byte `e+127`; zero groups use byte 0 with scale `2^-127`. Values use E4M3FN RNE with finite saturation, separate rank-ordered FP32 multiply/add, and BF16 RNE output. It does not quantize the output to FP8 a second time. OCP defines the group32 format but leaves detailed scale algorithms open; this is not the max-exponent/floor recipe in Microsoft’s MX emulator. The measured group128 implementation instead floors at exponent -120 (zero code 7).

GPU qualification compared FP8 bytes, scale bytes, decoded FP32 scale bits and BF16 output bits against an independent NumPy reference for 128 complete groups per seeded tensor, plus all nine edge groups on both ranks. The CPU scale test covered 65,280 finite BF16 patterns. All full outputs were finite and identical across ranks. The packaged GPU smoke also passed graph, frozen-baseline, signed-zero, partial-tail, triple-buffer wrap, alternating-shape, rank-skew and reuse gates at tile4096 / 152 CTAs / 5 teams.

These synthetic inputs yield very similar group32 and group128 errors; that is a measured property of these inputs, not a general equivalence claim. The results do not qualify model accuracy, serving throughput or all-input numerical behavior. Native NVFP4 uses unchanged upstream L2 defaults, without a best-configuration claim.

[Full 84 records, rank evidence, recipes and measured source hashes](fp8_format_accuracy.json). Source hashes refer to measured snapshots; current support files may include subsequent metadata-only privacy cleanup.

[OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) · [TensorRT MX-compliant dynamic quantization](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html#mx-compliant-dynamic-quantization)
