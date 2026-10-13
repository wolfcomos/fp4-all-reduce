#!/usr/bin/env python3
"""Compare fused BF16-in/out FP4, CuTe FP8 and BF16 NCCL on identical inputs.

Run under torchrun. CUDA Graph timing follows fp4ar_bench.cu: K operations
inside one graph, alignment before the event, per-sample maximum over ranks.
No performance number is emitted for a cell whose correctness gate failed.
"""
from __future__ import annotations

import argparse
import ctypes as C
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HIDDEN = 6144
GROUP_SIZE = 128
KERNEL_SHA256 = "2a008f18fb171d6d91c6c7a78523f7328470d163bcb10fca2ae011f0a1e4d07f"


def timing_summary(rank_samples):
    if not rank_samples or not rank_samples[0]:
        raise ValueError("empty samples")
    if len({len(s) for s in rank_samples}) != 1:
        raise ValueError("different sample counts across ranks")
    if any(not math.isfinite(x) or x <= 0 for samples in rank_samples for x in samples):
        raise ValueError("invalid CUDA event measurement")
    maxima = [max(values) for values in zip(*rank_samples)]
    ordered = sorted(maxima)
    n = len(ordered)
    return dict(median_us=ordered[n // 2], p10_us=ordered[n // 10],
                p90_us=ordered[(9 * n) // 10], rank_samples_us=rank_samples,
                max_rank_samples_us=maxima)


def finite_json(x):
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Native:
    """Typed, allocation-free launch bindings; no Python work inside replay."""
    def __init__(self, library, rank, world, device, rows, variant, host_group):
        self.lib = C.CDLL(str(library))
        p, i = C.c_void_p, C.c_int
        signatures = {
            "uid_size": (C.c_size_t, []), "get_unique_id": (i, [p]),
            "last_error": (C.c_char_p, []), "nccl_version": (i, [C.POINTER(i)]),
            "create": (i, [C.POINTER(p), i, i, i, i, C.c_char_p, p]),
            "launch_fp4": (i, [p, p, p, i, i, C.POINTER(i), p]),
            "allreduce_bf16": (i, [p, p, p, i, p]), "align": (i, [p, p]),
            "generate": (i, [p, p, i, i, C.c_uint32, p]),
            "reference": (i, [p, p, p, p, i, i, p]),
            "register_nccl_buffers": (i, [p, p, p, C.c_size_t]),
            "unregister_nccl_buffers": (i, [p]), "destroy": (i, [p]),
        }
        for name, (restype, argtypes) in signatures.items():
            fn = getattr(self.lib, "fp8_bridge_" + name)
            fn.restype, fn.argtypes = restype, argtypes
        uid = C.create_string_buffer(self.lib.fp8_bridge_uid_size())
        if rank == 0:
            self.call("get_unique_id", uid)
        payload = [uid.raw if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0, group=host_group)
        uid = C.create_string_buffer(payload[0], len(payload[0]))
        self.handle = p()
        self.call("create", C.byref(self.handle), rank, world, device, rows,
                  variant.encode(), uid)
        version = i()
        self.call("nccl_version", C.byref(version))
        self.version = version.value

    def call(self, name, *args):
        if getattr(self.lib, "fp8_bridge_" + name)(*args):
            raise RuntimeError(self.lib.fp8_bridge_last_error().decode())

    @staticmethod
    def pointer(x):
        return None if x is None else x.data_ptr()

    @staticmethod
    def stream():
        return torch.cuda.current_stream().cuda_stream

    def generate(self, sources, rows, seed):
        for rank in range(sources.shape[0]):
            self.call("generate", self.handle, sources[rank].data_ptr(), rank,
                      rows, seed, self.stream())

    def reference(self, sources, codec, fp32, rows, algo):
        self.call("reference", self.handle, sources.data_ptr(), self.pointer(codec),
                  self.pointer(fp32), rows, algo, self.stream())

    def align(self):
        self.call("align", self.handle, self.stream())

    def launch(self, backend, x, out, rows, algo, tune):
        if backend == "nccl":
            self.call("allreduce_bf16", self.handle, x.data_ptr(), out.data_ptr(), rows, self.stream())
        else:
            self.call("launch_fp4", self.handle, x.data_ptr(), out.data_ptr(), rows,
                      algo, tune, self.stream())


def bitwise_equal(a, b):
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def gather_objects(value, host_group):
    values = [None] * dist.get_world_size(host_group)
    dist.all_gather_object(values, value, group=host_group)
    return values


def all_passed(value, host_group):
    return all(gather_objects(bool(value), host_group))


def sampled_codec(backend, op, sources, output, seed):
    import numpy as np
    from fp8_oracle import sample_group_indices, switch_stage_contract
    from cute_fp8.reference import two_shot_reference

    groups = sample_group_indices(output.numel(), input_seed=seed)
    indices = torch.tensor(groups, dtype=torch.long, device=output.device)

    def sample(tensor):
        return tensor.flatten().reshape(-1, GROUP_SIZE).index_select(0, indices).flatten()

    def gather(tensor, floating=False):
        peers = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(peers, tensor)
        stacked = torch.stack(peers)
        return (stacked.float() if floating else stacked).cpu().numpy()

    inputs = sources.reshape(sources.shape[0], -1, GROUP_SIZE).index_select(1, indices)
    inputs = inputs.flatten(1).float().cpu().numpy()
    actual = gather(sample(output), True)
    if backend == "fp8-switch":
        result = switch_stage_contract(
            inputs, gather(sample(op.views[0])), gather(sample(op.views[1])),
            gather(op.views[2].index_select(0, indices), True),
            gather(op.views[3].index_select(0, indices), True), actual, GROUP_SIZE)
    else:
        expected = two_shot_reference(inputs, GROUP_SIZE).output
        def codes(array):
            bits = array.view(np.uint32) >> 16
            magnitude = (bits & np.uint32(0x7FFF)).astype(np.int64)
            return np.where(bits & np.uint32(0x8000), -magnitude, magnitude)
        distance = int(np.abs(codes(actual) - codes(expected)).max())
        result = dict(passed=bool(np.isfinite(actual).all()) and distance <= 2,
                      reference="two_shot_reference", max_bf16_ulp_distance=distance,
                      tolerance_bf16_ulps=2)
    result.update(sampled_group_indices=groups, sampled_numel=len(groups) * GROUP_SIZE,
                  numerical_gate_scope="sampled complete groups including original generator outliers")
    return result


def validate(backend, op, sources, out, codec, fp32, seed, host_group):
    finite = bool(torch.isfinite(out).all().item())
    peer = out.view(torch.uint8).clone()
    dist.broadcast(peer, src=0)
    rank_equal = bool(torch.equal(peer, out.view(torch.uint8)))
    del peer
    error = out.float() - fp32
    mse = error.square().mean().item()
    reference_mse = fp32.square().mean().item()
    metrics = dict(max_abs=error.abs().max().item(), rmse=math.sqrt(mse),
                   relative_l2=math.sqrt(mse / reference_mse) if reference_mse else math.sqrt(mse))
    del error
    if backend == "fp4":
        numerical = dict(passed=bitwise_equal(out, codec), contract="full_tensor_bitwise_original_fp4_codec")
    elif backend == "nccl":
        # Direct TP2 addition preserves (-0)+(-0)=-0. The upstream metric
        # reference starts from +0 and erases this sign before adding rank 1.
        # Keep a strict bitwise gate against the correct direct-add oracle.
        if sources.shape[0] == 2:
            expected = sources[0].float().add_(sources[1].float()).bfloat16()
            equal = bitwise_equal(out, expected)
            numerical = dict(passed=equal, contract="full_tensor_tp2_direct_fp32_sum_to_bf16",
                             signed_zero="Direct two-input addition; no +0 accumulator seed")
            if not equal:
                mismatch = out.flatten().view(torch.int16) != expected.flatten().view(torch.int16)
                indices = torch.nonzero(mismatch).flatten()[:64]
                numerical["diagnostics"] = dict(
                    bit_mismatch_count=int(mismatch.sum().item()),
                    numeric_mismatch_count=int((out != expected).sum().item()),
                    indices=indices.cpu().tolist(),
                    inputs=sources.flatten(1).index_select(1, indices).float().cpu().tolist(),
                    fp32_sum=fp32.flatten().index_select(0, indices).cpu().tolist(),
                    expected=expected.flatten().index_select(0, indices).float().cpu().tolist(),
                    actual=out.flatten().index_select(0, indices).float().cpu().tolist(),
                    expected_bits=expected.flatten().view(torch.int16).index_select(0, indices).cpu().tolist(),
                    actual_bits=out.flatten().view(torch.int16).index_select(0, indices).cpu().tolist(),
                )
            del expected
        else:
            raise ValueError("TP4 NCCL numerical gate must be specified before enabling TP4")
    else:
        numerical = sampled_codec(backend, op, sources, out, seed)
    passed = all_passed(finite and rank_equal and numerical["passed"], host_group)
    return dict(passed=passed, finite=finite, full_tensor_rank_bitwise_equal=rank_equal,
                numerical=numerical, error_vs_original_fp32_sum=metrics)


def capture(operation, count, native, host_group):
    operation()
    native.align()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(count):
            operation()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)
    return graph


def time_graph(graph, native, args, host_group):
    for _ in range(args.warmup):
        native.align()
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    # PyTorch CUDA events are lazy; materialize their CUDA handles before any
    # start/end interval, matching the native harness's precreated events.
    for event in starts + ends:
        event.record()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)
    for begin, end in zip(starts, ends):
        native.align()
        begin.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    local = [begin.elapsed_time(end) * 1000 / args.graph_k for begin, end in zip(starts, ends)]
    return timing_summary(gather_objects(local, host_group))


def run_cell(rows, backend, native, workspace, args, host_group, checkpoint, record):
    from cute_fp8.kernels import FP8AllReduce
    world, rank = dist.get_world_size(), dist.get_rank()
    sources = torch.empty((world, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    x = torch.empty_like(sources[rank])
    out = torch.empty_like(x)
    codec = torch.empty_like(x) if backend == "fp4" else None
    fp32 = torch.empty_like(x, dtype=torch.float32)
    tune = (C.c_int * 4)(*args.tune)
    op = None
    if backend.startswith("fp8-"):
        op = FP8AllReduce(x, group_size=GROUP_SIZE, backend=backend[4:], workspace=workspace)
        operation = lambda: op(x, out=out)
        record["operation"] = op.metadata
    else:
        operation = lambda: native.launch(backend, x, out, rows, args.algo, tune)
    record["checks"] = []
    checkpoint()
    # Compile and initialize before capture; inputs remain at fixed addresses.
    native.generate(sources, rows, 1000 + 17 * rows)
    x.copy_(sources[rank])
    graph = capture(operation, args.graph_k, native, host_group)
    stress_inputs, stress_outputs = [], []
    for set_index in range(3):
        seed = 1000 + 17 * rows + 7777 * set_index
        native.generate(sources, rows, seed)
        x.copy_(sources[rank])
        native.reference(sources, codec, fp32, rows, args.algo)
        operation()
        torch.cuda.synchronize()
        eager = out.clone()
        check = validate(backend, op, sources, out, codec, fp32, seed, host_group)
        check.update(input_set=set_index, seed=seed, mode="eager")
        record["checks"].append(check)
        checkpoint()
        if not check["passed"]:
            raise RuntimeError(f"{backend}/{rows}/set{set_index}: numerical gate failed")
        graph.replay()
        torch.cuda.synchronize()
        graph_check = validate(backend, op, sources, out, codec, fp32, seed, host_group)
        graph_check.update(input_set=set_index, mode="changed_input_graph",
                           bitwise_equal_to_eager=all_passed(bitwise_equal(out, eager), host_group))
        graph_check["passed"] &= graph_check["bitwise_equal_to_eager"]
        record["checks"].append(graph_check)
        checkpoint()
        if not graph_check["passed"]:
            raise RuntimeError(f"{backend}/{rows}/set{set_index}: graph gate failed")
        stress_inputs.append(x.clone())
        stress_outputs.append(eager)
        del eager
    # Bounded per-call byte checks expose stale results, including skewed arrival.
    # Comparing on-stream prevents the check from observing the next call's out.
    marks = torch.empty((args.stress,), dtype=torch.bool, device="cuda")
    for index in range(args.stress):
        set_index = index % len(stress_inputs)
        x.copy_(stress_inputs[set_index])
        if index % 11 == 0 and rank == (index // 11) % world:
            torch.cuda._sleep(10000)
        if index % 2:
            graph.replay()
        else:
            operation()
        marks[index] = torch.all(out.view(torch.uint8) == stress_outputs[set_index].view(torch.uint8))
    stress_passed = all_passed(bool(marks.all().item()), host_group)
    record["stress"] = dict(passed=stress_passed, checked_calls=args.stress,
                            graph_k=args.graph_k, alternating_eager_graph=True,
                            rotating_input_sets=3,
                            skew_sleep_cycles=10000, bounded_smoke_not_full_upstream_protocol_suite=True)
    checkpoint()
    if not stress_passed:
        raise RuntimeError(f"{backend}/{rows}: repeat/skew gate failed")
    del marks, stress_inputs, stress_outputs
    # Benchmark the original input set 0, restored outside measured operation.
    seed = 1000 + 17 * rows
    native.generate(sources, rows, seed)
    x.copy_(sources[rank])
    native.reference(sources, codec, fp32, rows, args.algo)
    measured = time_graph(graph, native, args, host_group)
    post = validate(backend, op, sources, out, codec, fp32, seed, host_group)
    post.update(input_set=0, mode="after_timing")
    record["checks"].append(post)
    if not post["passed"]:
        # Keep diagnostics but remove invalid performance from results.
        record["invalid_timing"] = measured
        checkpoint()
        raise RuntimeError(f"{backend}/{rows}: post-timing gate failed")
    record["timing"] = measured
    record["status"] = "passed"
    record["correctness_passed"] = True
    checkpoint()
    graph.reset()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,4,16,64,1024,4096,16384")
    parser.add_argument("--backends", default="nccl,fp4,fp8-sm,fp8-switch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, default=Path(os.environ.get("FP8_BRIDGE_LIB", ROOT / "build/fp8_bridge/libfp8_native_bridge.so")))
    parser.add_argument("--variant", choices=["L2", "L4"], default="L2")
    parser.add_argument("--algo", type=int, choices=[1, 2], default=1)
    parser.add_argument("--tune", default="0,0,0,0")
    parser.add_argument("--graph-k", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--stress", type=int, default=100)
    args = parser.parse_args()
    args.rows = [int(x) for x in args.rows.split(",")]
    args.backends = args.backends.split(",")
    args.tune = [int(x) for x in args.tune.split(",")]
    if not args.rows or any(r <= 0 or r > (2**31 - 1) // HIDDEN for r in args.rows):
        parser.error("rows must be positive and fit native int indexing")
    if len(set(args.rows)) != len(args.rows) or len(set(args.backends)) != len(args.backends):
        parser.error("duplicate rows/backends")
    if not set(args.backends) <= {"nccl", "fp4", "fp8-sm", "fp8-switch"} or not args.backends:
        parser.error("unsupported backend")
    if len(args.tune) != 4 or min(args.graph_k, args.iters, args.warmup, args.stress) <= 0:
        parser.error("four tuning integers and positive timing/stress counts required")
    if args.variant != "L2" or args.algo != 1:
        parser.error("this driver currently qualifies TP2 L2 one-shot only")
    return args


def main():
    global torch, dist
    args = parse_args()
    import torch
    import torch.distributed as dist
    from cute_fp8.kernels import FP8AllReduceWorkspace
    from fp8_oracle import SWITCH_CONTRACT, SWITCH_CONTRACT_METADATA
    kernel_hash = hashlib.sha256((HERE / "cute_fp8/kernels.py").read_bytes()).hexdigest()
    if kernel_hash != KERNEL_SHA256:
        raise RuntimeError("vendored kernel differs from current candidate snapshot")
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=8))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise RuntimeError("TP2 only: switch numerical contract is not qualified for TP4")
    host_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(minutes=8))
    started = time.time()
    report = dict(status="running", correctness_passed=False, tp_size=world,
                  rows=args.rows, backends=args.backends, hidden=HIDDEN, group_size=GROUP_SIZE,
                  kernel_sha256=kernel_hash, upstream_commit="2878bf50e20241d55500c2df76b87f0a81ef1c0f",
                  graph_k=args.graph_k, iters=args.iters, warmup=args.warmup,
                  fp4_variant=args.variant, fp4_algo=args.algo, fp4_tune=args.tune,
                  switch_contract=SWITCH_CONTRACT, switch_contract_metadata=SWITCH_CONTRACT_METADATA,
                  timing_contract="BF16 input to BF16 output; quantization and synchronization included; out-of-place NCCL; no timed input copy; max rank per sample then upper median",
                  numerical_scope="Synthetic codec and bounded protocol checks; not model accuracy qualification; FP8 sampled groups plus full-tensor finite/rank/repeat checks",
                  results=[], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def checkpoint():
        if rank == 0:
            report["elapsed_s"] = time.time() - started
            write_json(args.output, report)
    checkpoint()
    native = None
    try:
        native = Native(args.bridge, rank, world, device, max(args.rows), args.variant, host_group)
        runtime = dict(rank=rank, device=device, name=torch.cuda.get_device_name(),
                       torch=torch.__version__, cuda=torch.version.cuda,
                       cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
                       nccl_loaded_version=native.version, nccl_torch_query=torch.cuda.nccl.version(),
                       loaded_nccl_maps=[s for s in Path("/proc/self/maps").read_text().splitlines() if "libnccl" in s])
        report["runtime"] = gather_objects(runtime, host_group)
        workspaces = {backend: FP8AllReduceWorkspace(max(args.rows) * HIDDEN, torch.device("cuda", device),
                      group_size=GROUP_SIZE, backend=backend[4:])
                      for backend in args.backends if backend.startswith("fp8-")}
        report["workspaces"] = {backend: w.metadata for backend, w in workspaces.items()}
        checkpoint()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for rows in args.rows:
                for backend in args.backends:
                    record = dict(rows=rows, numel=rows * HIDDEN, bf16_input_bytes=rows * HIDDEN * 2,
                                  backend=backend, status="running", correctness_passed=False)
                    report["results"].append(record)
                    if rank == 0:
                        print(f"START {backend} rows={rows}", flush=True)
                    run_cell(rows, backend, native, workspaces.get(backend), args, host_group, checkpoint, record)
                    if rank == 0:
                        print(f"PASS {backend} rows={rows} median_us={record['timing']['median_us']:.3f}", flush=True)
        native.call("destroy", native.handle)
        native = None
        report.update(status="passed", correctness_passed=True)
        checkpoint()
        dist.destroy_process_group()
    except BaseException as error:
        if report["results"] and report["results"][-1]["status"] == "running":
            report["results"][-1]["status"] = "failed"
        report.update(status="failed", correctness_passed=False, error=repr(error), traceback=traceback.format_exc())
        checkpoint()
        # Do not enter collective cleanup after asymmetric exceptions; torchrun
        # terminates the other rank. The job has an outer timeout for GPU hangs.
        raise


if __name__ == "__main__":
    main()
