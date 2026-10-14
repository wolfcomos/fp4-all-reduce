# CuTe DSL FP8 AllReduce on GB200

This fork adds TP2, BF16-in/BF16-out AllReduce kernels for **standard MXFP8 group32** (E4M3 values and E8M0 scales) and **custom FP8 group128** (E4M3 values and UE8M0 scales). Each uses a single CuTe DSL kernel for TMA input loading, quantization, peer transfer, decode, FP32 reduction and BF16 output, with payload readiness, three rotating receive buffers and device call counters.

See the [four-format GB200 results](results/fp8_group32_gb200_tp2.md) and [machine-readable measurements](results/fp8_group32_gb200_tp2.json). The [earlier group128 study](results/fp8_gb200_tp2.md) remains available. These are standalone operator measurements on two GB200 GPUs. The upstream [VR200 results](RESULTS.md) are a separate study with different hardware, tuning and clock settings. None establishes serving throughput or model accuracy for these FP8 implementations.

## Environment and build

The qualified configuration is **TP2 on SM100-family hardware**, measured on two GB200 GPUs in one NVLink domain. Linux, CUDA development tools, a CUDA-enabled PyTorch installation, CuTe DSL and matching NCCL development headers/runtime are required. The native comparison bridge builds for `sm_100a`; this public Python benchmark does not support TP4.

Recorded software versions:

| Component | Measured version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA / native nvcc | 13.0 / 13.0.88 |
| `nvidia-cutlass-dsl` | 4.6.2 |
| `nvidia-cutlass-dsl-libs-cu13` | 4.6.2 |
| `cuda-python` / `cuda-bindings` | 13.3.1 / 13.3.1 |
| NumPy | 2.2.6 |
| Loaded NCCL library and headers | 2.31.2 |

Start from an environment with the appropriate **PyTorch CUDA 13 build for your platform**. PyTorch and the native NCCL development package are installed separately from the Python requirements below. No container image or cluster launcher is required by the source package.

Run commands from the repository root:

```bash
python -m pip install -r requirements-fp8.txt

# Set this to your matching NCCL development installation.
export NCCL_HOME=/path/to/nccl
export NCCL_INC="$NCCL_HOME/include"
export NCCL_LIB="$NCCL_HOME/lib"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$NCCL_LIB/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"

ARCH=100a bash scripts/build_fp8_bridge.sh
export FP8_BRIDGE_LIB="$PWD/build/fp8_bridge/libfp8_native_bridge.so"
```

The bridge requires **NCCL >=2.31**, including `nccl_device/core.h` and `ncclGetLsaDevicePointer`. It builds the upstream L2/L4 registrations, but the FP8 comparison driver runs TP2 L2 only. Build logs and toolchain information are retained under `build/fp8_bridge/`. `NVCC`, `BUILD_DIR` and `FP8_BRIDGE_LIB` can override the compiler and output locations.

**PyTorch and the native bridge must load the same NCCL library.** `LD_PRELOAD` alone was insufficient in an earlier environment: PyTorch explicitly opened a second library from its Python package directory. Use a consistently provisioned environment, or configure a container-local read-only library alias so both paths refer to the same library. Package metadata and `torch.cuda.nccl.version()` are insufficient evidence of the library actually loaded.

This read-only check rejects mixed loaded NCCL files after importing PyTorch and opening the bridge. Run it with the same environment as the benchmark:

```bash
python - <<'PY'
import ctypes
import os
from pathlib import Path
import torch

expected = Path(os.environ["NCCL_LIB"]) / "libnccl.so.2"
bridge = ctypes.CDLL(os.environ["FP8_BRIDGE_LIB"])
version = ctypes.c_int()
bridge.fp8_bridge_nccl_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
bridge.fp8_bridge_nccl_version.restype = ctypes.c_int
assert bridge.fp8_bridge_nccl_version(ctypes.byref(version)) == 0
assert version.value >= 23100, version.value
paths = {
    line.split()[-1]
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "libnccl.so" in line
}
assert paths and all(Path(p).is_file() and os.path.samefile(p, expected)
                     for p in paths), sorted(paths)
print("Loaded NCCL:", version.value, sorted(paths))
PY
```

The benchmark also records loaded NCCL mappings and runtime versions. Preserve those when checking a new environment. Do not substitute a different library silently to obtain a passing run.

## Benchmark commands

### Group32 and group128 comparison

[`harness/fp8_group32_compare.py`](../harness/fp8_group32_compare.py) compares actual BF16 NCCL, upstream NVFP4 L2, frozen group128 FP8 and standard group32 MXFP8 collectives. Both FP8 backends use tile 2048, 152 CTAs and 7 teams. This matches the earlier selected group128 geometry; **group32 was not independently tuned**.

Run the partial-tail smoke first. Tile 4096 does not divide a 6144-element row. The driver also runs the actual group32 kernel on nine groups (288 elements), including a partial 128-element tail, BF16 subnormals, signed zeros and scale codes at or above 128.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  harness/fp8_group32_compare.py \
  --rows 1,16,1024 --hidden 6144 \
  --lamport-tile 4096 --lamport-ctas 152 --lamport-teams 7 \
  --graph-k 10 --iters 20 --warmup 3 --stress 20 \
  --output results/fp8-group32-tail-smoke.json
```

Then run three fresh processes:

```bash
for repeat in 1 2 3; do
  torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
    harness/fp8_group32_compare.py \
    --rows 1,4,16,64,1024,4096,16384 --hidden 6144 \
    --backends nccl,fp4,fp8-ue8m0-lamport-batched,fp8-mxfp8-lamport-batched \
    --lamport-tile 2048 --lamport-ctas 152 --lamport-teams 7 \
    --graph-k 10 --iters 200 --warmup 20 --stress 100 \
    --output "results/fp8-group32-sweep-$repeat.json" || exit
done
```

The timing method is the same as the group128 study below: 200 paired rank-max samples, ten complete calls per CUDA Graph, 20 warmups, then the median of three fresh sweep medians. References and validation are outside timing. Speedups use the median of paired within-sweep ratios. The recorded NCCL environment had no explicit algorithm/channel/protocol overrides; no NCCL tuning search was performed.

Group32 must match the full BF16 GPU numerical reference and independent NumPy sampled output, q bytes and scale bytes. Gates also require finite outputs, full rank agreement, three changed inputs in eager and graph modes, 100 stress calls, and 12 alternating-shape phases across the triple-buffer ring with rank skew and idle CTAs. Group128 retains its original codec and full-output frozen-baseline gates. All gates, the separate partial-tail smoke, and all three sweeps must pass before reporting a completed comparison.

The CPU-only audit command below recomputes timing from raw rank samples and checks the recorded gates, configuration and source hashes. It deliberately pins this study's software versions and loaded NCCL path (`/nccl/lib/libnccl.so.2`); it is not a validator for arbitrary environments. Supply matching arguments when auditing smoke or another supported configuration.

```bash
python harness/fp8_group32_compare.py \
  --validate results/fp8-group32-sweep-1.json \
  --rows 1,4,16,64,1024,4096,16384 --hidden 6144 \
  --lamport-tile 2048 --lamport-ctas 152 --lamport-teams 7 \
  --graph-k 10 --iters 200 --warmup 20 --stress 100
```

### Earlier group128 study

The commands in this subsection reproduce the original group128 experiment and retain its additional tuned FP8 baseline. They do not invoke the new group32 kernel.

The public entry point is [`harness/fp8_bench.py`](../harness/fp8_bench.py). It delegates to the tested comparison driver and retains its correctness gates. Its default backends are `nccl,fp4,fp8-ue8m0-tuned,fp8-ue8m0-lamport-batched`. The original `fp8-ue8m0-sm` is also allowed. Explicit arguments override defaults; unsupported or duplicate backend names are rejected.

The selected configuration is **tile 2048, 152 CTAs, 7 two-warp teams per CTA**. The public wrapper sets these values. The frozen low-level workspace constructor retains its earlier defaults of **tile 6144 and 5 teams**, so direct API users must explicitly supply the selected geometry.

First run the mandatory partial-tail smoke. Tile 4096 does not divide a row of 6144 values: this exercises input tails and alternating large→tiny→large calls, including CTAs with no current input work.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  harness/fp8_bench.py \
  --rows 1,16,1024 \
  --lamport-tile 4096 --lamport-ctas 152 --lamport-teams 5 \
  --graph-k 10 --iters 20 --warmup 3 --stress 20 \
  --output results/fp8-tail-smoke.json
```

Then run three **fresh processes** for the formal comparison:

```bash
for repeat in 1 2 3; do
  torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
    harness/fp8_bench.py \
    --rows 1,4,16,64,1024,4096,16384 \
    --backends nccl,fp4,fp8-ue8m0-tuned,fp8-ue8m0-lamport-batched \
    --fused-tile 16384 --fused-ctas 152 \
    --ue8m0-tile 32768 --ue8m0-ctas 152 --ue8m0-threads 256 \
    --lamport-tile 2048 --lamport-ctas 152 --lamport-teams 7 \
    --graph-k 10 --iters 200 --warmup 20 --stress 100 \
    --output "results/fp8-sweep-$repeat.json" || exit
done
```

Each sample times ten complete calls in a CUDA Graph and divides by ten. The driver takes the maximum across the two ranks for each paired sample, then the upper median of 200 samples. Published latency is the median of the three sweep medians; reported speedup is the median of the three paired within-sweep ratios. Input generation and validation are outside timing. Quantization, communication, decode/reduction and synchronization are inside timing. GPU clocks were not modified.

The NCCL baseline is out-of-place BF16 Sum with the recorded environment settings; no best-NCCL claim is made. The NVFP4 baseline is the upstream TP2 L2 one-shot codec with default tuning. The previous tuned FP8 baseline uses tile 32768, 152 CTAs and 256 threads; the separate untimed frozen UE8M0 correctness control uses tile 16384 and 152 CTAs.

Accept results only after a zero process exit and a completed report with `status="passed"` and `correctness_passed=true`. Reports are checkpointed on failure; partial data must not be treated as a completed sweep. Gates cover codec/wire samples, full-output finiteness and rank agreement, full-output equality with the preserved UE8M0 implementation, changed-input graph replay, stress/rank skew and repeated shape changes. The selected tile 2048 sweep does not replace the partial-tail smoke.

## Direct Python API

Save the following group128 example as `example_fp8.py` in the repository root and launch it with `torchrun --standalone --nproc-per-node=2 example_fp8.py`. The kernel API does not itself require the native comparison bridge.

```python
import os
import torch
import torch.distributed as dist
from harness.cute_fp8.ue8m0_lamport_batched import (
    BatchedLamportUE8M0FP8AllReduceWorkspace,
    BatchedLamportUE8M0FP8AllReduce,
)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
assert dist.get_world_size() == 2
device = torch.device("cuda", local_rank)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(stream):
    x = torch.randn((1024, 6144), device=device, dtype=torch.bfloat16)
    workspace = BatchedLamportUE8M0FP8AllReduceWorkspace(
        max_numel=x.numel(), device=device, group=dist.group.WORLD,
        group_size=128, tile_elems=2048, ctas=152, teams=7,
    )
    operation = BatchedLamportUE8M0FP8AllReduce(
        x, group=dist.group.WORLD, workspace=workspace,
    )
    output = torch.empty_like(x)
    operation(x, out=output)

torch.cuda.current_stream().wait_stream(stream)
torch.cuda.synchronize()
dist.barrier()
dist.destroy_process_group()
```

For standard group32 MXFP8, keep the same process/stream setup and replace the import and construction with:

```python
from harness.cute_fp8.mxfp8_lamport_batched import (
    MXFP8LamportBatchedAllReduceWorkspace,
    MXFP8LamportBatchedAllReduce,
)

# Inside the same dedicated stream context, after creating x:
workspace = MXFP8LamportBatchedAllReduceWorkspace(
    max_numel=x.numel(), device=device, group=dist.group.WORLD,
    group_size=32, tile_elems=2048, ctas=152, teams=7,
)
operation = MXFP8LamportBatchedAllReduce(
    x, group=dist.group.WORLD, group_size=32, workspace=workspace,
)
output = torch.empty_like(x)
operation(x, out=output)
```

The two formats use separate workspaces. At capacity `16384 * 6144` and the measured geometry, symmetric allocations are 311,430,720 bytes (297.003 MiB) per rank for group32 and 304,352,832 bytes (290.253 MiB) for group128. These include three receive buffers and device counters, excluding inputs and outputs.

Construct/rendezvous the workspace and compile each shape specialization on both ranks before graph capture. Use a dedicated non-default stream for capture; retain the workspace, operations and captured buffers until execution finishes. Different shapes may share one sufficiently large workspace, but each shape needs its own operation object and calls must use the same sequence on both ranks. `operation(x)` reuses its internally owned output; pass a distinct `out=` buffer when a previous result must remain available.

Lifecycle requirements:

- Exactly two ranks make matching **nonempty** calls in the same order. Each input has a positive number of elements divisible by its group size (32 or 128) and fits the workspace capacity.
- Input and output are contiguous, 16-byte-aligned CUDA BF16 tensors on the specialization's device, with matching shapes and no storage overlap.
- Every operation sharing a workspace uses the same fixed cooperative CTA grid and geometry. The grid cannot exceed the smaller peer SM count; the measured 152-CTA setting therefore requires at least 152 SMs on each GPU.
- Calls are strictly serialized through whole-kernel stream ordering. No concurrent workspace use, cross-stream overlap or programmatic dependent launch (PDL) is supported. The host checks cannot detect every violation of this ordering contract.
- All fixed CTAs launch, including those without current payload work. Every transmitted q/scale byte is polled, including zero-filled tail groups. All TMA sends and previous-buffer resets finish before kernel exit. These conditions make three-buffer reuse safe without a peer completion handshake.

The implementation follows the measured TMA/payload-polling path on this hardware. This qualification is not a general proof of mixed-size memory publication semantics on other architectures.

## Numerical contract and four-format comparison

For each source-local group of 128 BF16 values:

```text
amax = max(abs(x))
e = max(-120, ceil(log2(amax / 448)))     # zero group: e = -120
scale = 2**e                          # stored byte = e + 127
q = E4M3_RNE(x * 2**(-e))
```

The group128 kernel uses separate FP32 multiply/add operations, a +0 accumulator seed, source rank 0 then source rank 1, and final BF16 RNE. There is no second FP8 quantization of the reduced output. Zero groups use scale byte 7; generated scale codes are 7..247. Negative zero in E4M3 (`0x80`) is preserved, while `0xFF` is reserved for payload readiness.

The standard group32 kernel uses the same reduction order and output rounding, with one E8M0 scale per 32 values and `e=max(-127,ceil(log2(amax/448)))`. Byte 0 represents the nonzero scale `2**-127`; zero groups use this code. This is NVIDIA TensorRT's round-up scale recipe. The actual kernel passed full-output GPU reference checks, independent sampled NumPy checks, and the nine-group edge test including the minimum BF16 subnormal `2**-133`, signed zeros, 448/450 scale boundaries and large finite values. The `2**100` edge case uses byte/output gates without FP32 squared-error metrics, which would overflow.

The supported numerical domain requires finite BF16 inputs, finite decoded FP32 source terms and reduction intermediates, and a BF16-representable result. Finite input alone is insufficient: extreme finite BF16 values can overflow after quantization/dequantization or final output rounding. Such inputs are unsupported; the implementation does not silently saturate the final sum into a finite result.

The **earlier accuracy-only probe** below compares actual NCCL BF16, upstream NVFP4 L2 and group128 FP8 outputs with a local group32 MXFP8 numerical reference on the same CUDA-generated BF16 inputs. That probe uses the same group32 scale recipe but performs no group32 communication or timing. The [new four-format performance study](results/fp8_group32_gb200_tp2.md) measures the actual group32 collective separately; the historical probe and its results remain unchanged.

```bash
python harness/fp8_format_accuracy.py --self-test

torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  harness/fp8_format_accuracy.py \
  --bridge "$FP8_BRIDGE_LIB" \
  --rows 1,4,16,64,1024,4096,16384 \
  --lamport-tile 2048 --lamport-ctas 152 --lamport-teams 7 \
  --output results/fp8-format-accuracy.json
```

The probe separately checks GPU edge cases and sampled complete groups against a NumPy oracle, including quantized bytes, scale bytes, decoded scale bits and BF16 output bits. It records full-output maximum absolute error, RMSE and relative L2 against the original BF16 source inputs summed in FP32. For output `y` and reference `r`, relative L2 is `sqrt(mean((float(y)-r)**2)/mean(r**2))`; multiply by 100 when reporting a percentage. Metrics use FP32 arithmetic, matching the measured benchmark.

These are synthetic vector-error and implementation-correctness checks. Relative L2 is **not model/task accuracy loss**, and codec bitwise agreement does not establish model quality. See the [historical four-format numerical results](results/fp8_format_accuracy.md) and [current four-format performance results](results/fp8_group32_gb200_tp2.md) for the measured values and comparison limitations.
