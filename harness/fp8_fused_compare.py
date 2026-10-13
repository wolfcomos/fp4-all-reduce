#!/usr/bin/env python3
"""Standalone comparison of fused CuTe FP8 against the preserved FP8/FP4/NCCL baselines.

Run with torchrun. Timed calls include input quantization, transport, reduction,
decode and internal synchronization. Adversarial and workspace-reuse checks are
untimed. This driver imports existing helpers without modifying their sources.
"""
from __future__ import annotations

import argparse
import ctypes as C
import datetime
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import time
import traceback

import fp8_compare as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HIDDEN, GROUP_SIZE = 6144, 128
LEGACY = {"nccl", "fp4", "fp8-sm", "fp8-switch"}
FUSED = {"fp8-fused-sm", "fp8-fused-switch"}
PIPELINED = {"fp8-pipelined-sm"}
UE8M0_TUNED = {"fp8-ue8m0-tuned"}
LAMPORT = {"fp8-ue8m0-lamport", "fp8-ue8m0-lamport-batched"}
UE8M0_EQUIVALENT = UE8M0_TUNED | LAMPORT
UE8M0 = {"fp8-ue8m0-sm"} | UE8M0_EQUIVALENT
EXPERIMENTAL = FUSED | PIPELINED | UE8M0
ALL_BACKENDS = LEGACY | EXPERIMENTAL


def gather_tensor(tensor):
    peers = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, tensor)
    return torch.stack(peers)


def sampled_one_shot(op, sources, output, seed):
    import numpy as np
    from fp8_oracle import sample_group_indices
    from fp8_fused_oracle import one_shot_contract, one_shot_reference
    groups = sample_group_indices(output.numel(), input_seed=seed)
    indices = torch.tensor(groups, device=output.device, dtype=torch.long)
    inputs = sources.reshape(2, -1, GROUP_SIZE).index_select(1, indices).flatten(1).float().cpu().numpy()
    sampled = output.reshape(-1, GROUP_SIZE).index_select(0, indices).flatten()
    actual = gather_tensor(sampled).float().cpu().numpy()
    result = one_shot_contract(inputs, actual)
    received = op.sample_received_groups(indices)
    if received["source_rank"] != 1 - dist.get_rank():
        raise RuntimeError("Fused incoming payload does not identify the peer source")
    received_q = gather_tensor(received["q"].reshape(-1)).cpu().numpy()
    received_scales = gather_tensor(received["scales"].reshape(-1)).float().cpu().numpy()
    expected = one_shot_reference(inputs)
    expected_q = expected["quantized_inputs"][[1, 0]]
    expected_scales = expected["input_scales"][[1, 0]]
    q_mismatch = received_q != expected_q
    scale_mismatch = received_scales.view(np.uint32) != expected_scales.view(np.uint32)
    stage_passed = not bool(q_mismatch.any() or scale_mismatch.any())
    result["received_payload"] = dict(passed=stage_passed,
        source_order_per_receiving_rank=[1, 0],
        quantized_byte_mismatches_per_rank=q_mismatch.sum(axis=1).tolist(),
        scale_bit_mismatches_per_rank=scale_mismatch.sum(axis=1).tolist(),
        negative_zero_byte_count_per_rank=(received_q == 0x80).sum(axis=1).tolist(),
        expected_negative_zero_byte_count_per_rank=(expected_q == 0x80).sum(axis=1).tolist(),
        scale_floor_count_per_rank=(received_scales == np.float32(2.0 ** -120)).sum(axis=1).tolist())
    result["passed"] &= stage_passed
    result.update(sampled_group_indices=groups, sampled_numel=len(groups) * GROUP_SIZE,
        numerical_gate_scope="All groups for small tensors; otherwise sampled complete groups including original-generator outlier groups")
    return result


def sampled_ue8m0(op, sources, output, seed):
    from fp8_oracle import sample_group_indices
    from fp8_ue8m0_oracle import ue8m0_contract, ue8m0_reference, received_payload_contract
    groups = sample_group_indices(output.numel(), input_seed=seed)
    indices = torch.tensor(groups, device=output.device, dtype=torch.long)
    inputs = sources.reshape(2, -1, GROUP_SIZE).index_select(1, indices).flatten(1).float().cpu().numpy()
    sampled = output.reshape(-1, GROUP_SIZE).index_select(0, indices).flatten()
    actual = gather_tensor(sampled).float().cpu().numpy()
    result = ue8m0_contract(inputs, actual)
    received = op.sample_received_groups(indices)
    if received["source_rank"] != 1 - dist.get_rank():
        raise RuntimeError("UE8M0 incoming payload does not identify the peer source")
    received_q = gather_tensor(received["q"].reshape(-1)).cpu().numpy()
    received_codes = gather_tensor(received["scale_codes"].reshape(-1)).cpu().numpy()
    # Scale bytes are the wire representation. The CPU payload audit decodes
    # them independently; the output oracle also checks kernel scale arithmetic.
    payload = received_payload_contract(ue8m0_reference(inputs), received_q, received_codes)
    result["received_payload"] = payload
    result["passed"] &= payload["passed"]
    result.update(sampled_group_indices=groups, sampled_numel=len(groups) * GROUP_SIZE,
        numerical_gate_scope="All groups for small tensors; otherwise sampled complete groups including original-generator outlier groups")
    return result


def validate(backend, op, sources, out, codec, fp32, seed, host_group):
    if backend in LEGACY:
        return base.validate(backend, op, sources, out, codec, fp32, seed, host_group)
    if backend == "fp8-fused-switch":
        return base.validate("fp8-switch", op, sources, out, codec, fp32, seed, host_group)
    finite = bool(torch.isfinite(out).all().item())
    peer = out.view(torch.uint8).clone()
    dist.broadcast(peer, src=0)
    rank_equal = base.bitwise_equal(peer, out.view(torch.uint8))
    del peer
    error = out.float() - fp32
    mse, reference_mse = error.square().mean().item(), fp32.square().mean().item()
    metrics = dict(max_abs=error.abs().max().item(), rmse=math.sqrt(mse),
                   relative_l2=math.sqrt(mse / reference_mse) if reference_mse else math.sqrt(mse))
    del error
    numerical = (sampled_ue8m0(op, sources, out, seed) if backend in UE8M0
                 else sampled_one_shot(op, sources, out, seed))
    passed = base.all_passed(finite and rank_equal and numerical["passed"], host_group)
    return dict(passed=passed, finite=finite, full_tensor_rank_bitwise_equal=rank_equal,
                numerical=numerical, error_vs_original_fp32_sum=metrics)


def make_operation(backend, x, out, workspace, native, rows, args):
    op = None
    if backend in ("fp8-sm", "fp8-switch"):
        from cute_fp8.kernels import FP8AllReduce
        op = FP8AllReduce(x, group_size=GROUP_SIZE, backend=backend[4:], workspace=workspace)
    elif backend == "fp8-fused-sm":
        from cute_fp8.fused import FusedFP8AllReduce
        op = FusedFP8AllReduce(x, group_size=GROUP_SIZE, workspace=workspace)
    elif backend == "fp8-fused-switch":
        from cute_fp8.fused_switch import FusedSwitchFP8AllReduce
        op = FusedSwitchFP8AllReduce(x, group_size=GROUP_SIZE, workspace=workspace)
    elif backend == "fp8-pipelined-sm":
        from cute_fp8.pipelined import PipelinedFP8AllReduce
        op = PipelinedFP8AllReduce(x, group_size=GROUP_SIZE, workspace=workspace)
    elif backend == "fp8-ue8m0-sm":
        from cute_fp8.ue8m0 import UE8M0FP8AllReduce
        op = UE8M0FP8AllReduce(x, group_size=GROUP_SIZE, workspace=workspace)
    elif backend == "fp8-ue8m0-tuned":
        from cute_fp8.ue8m0_tuned import TunedUE8M0FP8AllReduce
        op = TunedUE8M0FP8AllReduce(x, group_size=GROUP_SIZE, workspace=workspace)
    elif backend == "fp8-ue8m0-lamport":
        from cute_fp8.ue8m0_lamport import LamportUE8M0FP8AllReduce
        op = LamportUE8M0FP8AllReduce(x, group_size=GROUP_SIZE, workspace=workspace)
    elif backend == "fp8-ue8m0-lamport-batched":
        from cute_fp8.ue8m0_lamport_batched import BatchedLamportUE8M0FP8AllReduce
        op = BatchedLamportUE8M0FP8AllReduce(x, group_size=GROUP_SIZE, workspace=workspace)
    if op is not None:
        return op, lambda: op(x, out=out)
    tune = (C.c_int * 4)(*args.tune)
    return None, lambda: native.launch(backend, x, out, rows, 1, tune)


def run_cell(rows, backend, native, workspace, args, host_group, checkpoint, record, baseline_workspace=None):
    rank = dist.get_rank()
    sources = torch.empty((2, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    x, out = torch.empty_like(sources[rank]), torch.empty_like(sources[rank])
    codec = torch.empty_like(x) if backend == "fp4" else None
    fp32 = torch.empty_like(x, dtype=torch.float32)
    op, operation = make_operation(backend, x, out, workspace, native, rows, args)
    if op is not None:
        record["operation"] = op.metadata
    baseline_op = baseline_operation = baseline_out = baseline_graph = None
    if backend == "fp8-pipelined-sm" or backend in UE8M0_EQUIVALENT:
        baseline_backend = "fp8-fused-sm" if backend == "fp8-pipelined-sm" else "fp8-ue8m0-sm"
        if baseline_workspace is None or baseline_workspace is workspace:
            raise RuntimeError(f"{backend} equivalence requires a separate {baseline_backend} workspace")
        baseline_out = torch.empty_like(out)
        baseline_op, baseline_operation = make_operation(baseline_backend, x, baseline_out,
                                                         baseline_workspace, native, rows, args)
        record["baseline_equivalence"] = dict(status="running", passed=False,
            baseline_backend=baseline_backend, contract="full_tensor_bf16_bits_same_inputs_v1",
            separate_workspace=True, separate_output=True, untimed=True, graph_k=args.graph_k,
            baseline_operation=baseline_op.metadata, checks=[])
    record["checks"] = []
    checkpoint()
    native.generate(sources, rows, 1000 + 17 * rows)
    x.copy_(sources[rank])
    graph = base.capture(operation, args.graph_k, native, host_group)
    if baseline_operation is not None:
        baseline_graph = base.capture(baseline_operation, args.graph_k, native, host_group)

    def compare_baseline(mode, set_index, seed):
        if baseline_operation is None:
            return
        baseline_out.fill_(float("nan"))
        baseline_operation() if mode == "eager" else baseline_graph.replay()
        torch.cuda.synchronize()
        finite = base.all_passed(bool(torch.isfinite(baseline_out).all().item()), host_group)
        equal = base.all_passed(base.bitwise_equal(out, baseline_out), host_group)
        comparison = dict(mode=mode, input_set=set_index, seed=seed, baseline_finite=finite,
                          full_tensor_bitwise_equal=equal, passed=finite and equal)
        record["baseline_equivalence"]["checks"].append(comparison)
        checkpoint()
        if not comparison["passed"]:
            raise RuntimeError(f"{backend}/{rows}/set{set_index}: {mode} differs from preserved {baseline_backend}")

    stress_inputs, stress_outputs = [], []
    for set_index in range(3):
        seed = 1000 + 17 * rows + 7777 * set_index
        native.generate(sources, rows, seed)
        x.copy_(sources[rank])
        native.reference(sources, codec, fp32, rows, 1)
        out.fill_(float("nan"))
        operation()
        torch.cuda.synchronize()
        eager = out.clone()
        check = validate(backend, op, sources, out, codec, fp32, seed, host_group)
        check.update(input_set=set_index, seed=seed, mode="eager")
        record["checks"].append(check)
        checkpoint()
        if not check["passed"]:
            raise RuntimeError(f"{backend}/{rows}/set{set_index}: numerical gate failed")
        compare_baseline("eager", set_index, seed)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        graph_check = validate(backend, op, sources, out, codec, fp32, seed, host_group)
        graph_check.update(input_set=set_index, mode="changed_input_graph",
                           bitwise_equal_to_eager=base.all_passed(base.bitwise_equal(out, eager), host_group))
        graph_check["passed"] &= graph_check["bitwise_equal_to_eager"]
        record["checks"].append(graph_check)
        checkpoint()
        if not graph_check["passed"]:
            raise RuntimeError(f"{backend}/{rows}/set{set_index}: graph gate failed")
        compare_baseline("graph", set_index, seed)
        stress_inputs.append(x.clone())
        stress_outputs.append(eager)
    if baseline_operation is not None:
        record["baseline_equivalence"].update(status="passed", passed=True)
    marks = torch.empty((args.stress,), dtype=torch.bool, device="cuda")
    for index in range(args.stress):
        set_index = index % 3
        x.copy_(stress_inputs[set_index])
        out.fill_(float("nan"))
        if index % 11 == 0 and rank == (index // 11) % 2:
            torch.cuda._sleep(10000)
        graph.replay() if index % 2 else operation()
        marks[index] = torch.all(out.view(torch.uint8) == stress_outputs[set_index].view(torch.uint8))
    stress_passed = base.all_passed(bool(marks.all().item()), host_group)
    record["stress"] = dict(passed=stress_passed, checked_calls=args.stress, graph_k=args.graph_k,
        alternating_eager_graph=True, rotating_input_sets=3, skew_sleep_cycles=10000,
        output_poisoned_between_calls=True, bounded_smoke_not_full_upstream_protocol_suite=True)
    checkpoint()
    if not stress_passed:
        raise RuntimeError(f"{backend}/{rows}: changed-input/repeat/skew gate failed")
    del marks, stress_inputs, stress_outputs, eager
    seed = 1000 + 17 * rows
    native.generate(sources, rows, seed)
    x.copy_(sources[rank])
    native.reference(sources, codec, fp32, rows, 1)
    measured = base.time_graph(graph, native, args, host_group)
    post = validate(backend, op, sources, out, codec, fp32, seed, host_group)
    post.update(input_set=0, mode="after_timing")
    record["checks"].append(post)
    if not post["passed"]:
        record["invalid_timing"] = measured
        checkpoint()
        raise RuntimeError(f"{backend}/{rows}: post-timing numerical gate failed")
    record.update(timing=measured, status="passed", correctness_passed=True)
    checkpoint()
    graph.reset()
    if baseline_graph is not None:
        baseline_graph.reset()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)


def fill_adversarial(sources, name):
    """Finite deterministic BF16 cases; output poisoning is separate."""
    n = sources.shape[1] * HIDDEN
    index = torch.arange(n, device=sources.device, dtype=torch.int64)
    if name == "positive_zero":
        sources.zero_()
    elif name == "mixed_signed_zero":
        bits = sources.flatten(1).view(torch.int16)
        bits[0].copy_(torch.where(index % 2 == 0, 0, -32768).to(torch.int16))
        bits[1].copy_(torch.where(index % 4 < 2, -32768, 0).to(torch.int16))
    elif name == "cancellation":
        values = ((index % 31).float() - 15) / 8
        sources[0].copy_(values.reshape_as(sources[0]))
        sources[1].copy_(-sources[0])
    elif name == "varying_ranges_outliers":
        # Keep squared-error/reference metrics finite in the preserved FP32
        # metric helper while exercising both the scale floor and wide ranges.
        exponents = torch.tensor([-120, -100, -30, -10, 0, 10, 20, 30, 40], device=sources.device)
        scales = torch.pow(2.0, exponents[(index // GROUP_SIZE) % len(exponents)].float())
        values = ((index % 29).float() - 14) * scales / 16
        values = torch.where(index % GROUP_SIZE == 0, scales * 40, values)
        sources[0].copy_(values.reshape_as(sources[0]))
        sources[1].copy_((torch.where(index % 3 == 0, -values, values) * 0.5).reshape_as(sources[1]))
    else:
        raise ValueError(name)


def adversarial_check(backend, workspace, native, args, host_group, checkpoint, entry):
    rows, rank = 1, dist.get_rank()
    sources = torch.empty((2, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    x, out = torch.empty_like(sources[rank]), torch.empty_like(sources[rank])
    fp32 = torch.empty_like(x, dtype=torch.float32)
    op, operation = make_operation(backend, x, out, workspace, native, rows, args)
    fill_adversarial(sources, "positive_zero")
    x.copy_(sources[rank])
    graph = base.capture(operation, args.graph_k, native, host_group)
    entry.update(cases=[], rows=rows, sampled_gate_is_exhaustive=True,
                 output_poisoned_between_calls=True, graph_k=args.graph_k)
    checkpoint()
    for name in ("positive_zero", "mixed_signed_zero", "cancellation", "varying_ranges_outliers"):
        fill_adversarial(sources, name)
        x.copy_(sources[rank])
        native.reference(sources, None, fp32, rows, 1)
        case = dict(name=name, checks=[])
        entry["cases"].append(case)
        eager = None
        for mode in ("eager", "graph"):
            out.fill_(float("nan"))
            operation() if mode == "eager" else graph.replay()
            torch.cuda.synchronize()
            check = validate(backend, op, sources, out, None, fp32, 1000 + 17 * rows, host_group)
            check["mode"] = mode
            if mode == "eager":
                eager = out.clone()
            else:
                check["bitwise_equal_to_eager"] = base.all_passed(base.bitwise_equal(out, eager), host_group)
                check["passed"] &= check["bitwise_equal_to_eager"]
            case["checks"].append(check)
            checkpoint()
            if not check["passed"]:
                raise RuntimeError(f"{backend}: adversarial {name}/{mode} failed")
        case["passed"] = True
        del eager
    graph.reset()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)
    entry.update(passed=True, status="passed")
    checkpoint()


def alternating_shape_check(backend, workspace, native, args, host_group, checkpoint, entry, baseline_workspace=None):
    rank = dist.get_rank()
    choices = sorted(set((min(args.rows), args.rows[len(args.rows) // 2], max(args.rows))))
    prepared = {}
    lamport = backend in LAMPORT
    input_sets = 5 if lamport else 3
    entry.update(shapes=choices, phases=[], preparation_checks=[], graph_k=args.graph_k,
                 same_workspace=True, bounded_smoke_not_full_upstream_protocol_suite=True)
    checkpoint()
    if lamport:
        if baseline_workspace is None or baseline_workspace is workspace:
            raise RuntimeError("Lamport ring reuse requires a separate frozen UE8M0 workspace")
        entry.update(triple_buffer_wrap=True, receive_buffers=3, rotating_input_sets=5,
                     skew_sleep_cycles=10000, output_poisoned_between_calls=True,
                     baseline_backend="fp8-ue8m0-sm", baseline_operations={},
                     counter_scope="all CTAs on both ranks; completed calls, including graph10 replays")
    for rows in choices:
        sources = torch.empty((2, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        x, out = torch.empty_like(sources[rank]), torch.empty_like(sources[rank])
        fp32 = torch.empty_like(x, dtype=torch.float32)
        op, operation = make_operation(backend, x, out, workspace, native, rows, args)
        native.generate(sources, rows, 1000 + 17 * rows)
        x.copy_(sources[rank])
        graph = base.capture(operation, args.graph_k, native, host_group)
        expected = []
        baseline_out = baseline_operation = None
        if lamport:
            baseline_out = torch.empty_like(out)
            baseline_op, baseline_operation = make_operation("fp8-ue8m0-sm", x, baseline_out,
                baseline_workspace, native, rows, args)
            entry["baseline_operations"][str(rows)] = baseline_op.metadata
        for index in range(input_sets):
            seed = 1000 + 17 * rows + 7777 * index
            native.generate(sources, rows, seed)
            x.copy_(sources[rank])
            native.reference(sources, None, fp32, rows, 1)
            out.fill_(float("nan"))
            operation()
            torch.cuda.synchronize()
            check = validate(backend, op, sources, out, None, fp32, seed, host_group)
            if lamport:
                baseline_out.fill_(float("nan"))
                baseline_operation()
                torch.cuda.synchronize()
                finite = base.all_passed(bool(torch.isfinite(baseline_out).all().item()), host_group)
                equal = base.all_passed(base.bitwise_equal(out, baseline_out), host_group)
                check.update(baseline_finite=finite, full_tensor_bitwise_equal_to_baseline=equal)
                check["passed"] &= finite and equal
            check.update(rows=rows, input_set=index)
            entry["preparation_checks"].append(check)
            checkpoint()
            if not check["passed"]:
                raise RuntimeError(f"{backend}: workspace-reuse preparation failed")
            expected.append((baseline_out if lamport else out).clone())
        prepared[rows] = dict(sources=sources, x=x, out=out, fp32=fp32, op=op,
                              operation=operation, graph=graph, expected=expected)
    small, middle, large = choices[0], choices[len(choices) // 2], choices[-1]
    sequence = [large, small, middle, small, large, middle, small, large]
    if lamport:
        sequence += [middle, large, small, large]
    for phase, rows in enumerate(sequence):
        state = prepared[rows]
        index = phase % input_sets
        seed = 1000 + 17 * rows + 7777 * index
        native.generate(state["sources"], rows, seed)
        state["x"].copy_(state["sources"][rank])
        native.reference(state["sources"], None, state["fp32"], rows, 1)
        state["out"].fill_(float("nan"))
        before = base.gather_objects(workspace.protocol_snapshot(), host_group) if lamport else None
        if rank == phase % 2:
            torch.cuda._sleep(10000)
        mode = "graph" if phase % 2 == 0 else "eager"
        state["graph"].replay() if mode == "graph" else state["operation"]()
        torch.cuda.synchronize()
        check = validate(backend, state["op"], state["sources"], state["out"], None,
                         state["fp32"], seed, host_group)
        equal = base.all_passed(base.bitwise_equal(state["out"], state["expected"][index]), host_group)
        check.update(phase=phase, rows=rows, input_set=index, seed=seed, mode=mode,
                     bitwise_equal_to_eager=equal)
        check["passed"] &= equal
        if lamport:
            after = base.gather_objects(workspace.protocol_snapshot(), host_group)
            delta = args.graph_k if mode == "graph" else 1
            expected_tiles = [max(0, ((rows * HIDDEN + args.lamport_tile - 1) // args.lamport_tile
                                      - cta + args.lamport_ctas - 1) // args.lamport_ctas)
                              for cta in range(args.lamport_ctas)]
            state_passed = all(
                len(b["call_counters"]) == len(a["call_counters"]) == args.lamport_ctas
                and len(set(b["call_counters"])) == len(set(a["call_counters"])) == 1
                and b["completed_epochs"] == b["call_counters"]
                and a["completed_epochs"] == a["call_counters"]
                and b["latest_buffers"] == [(n-1)%3 for n in b["call_counters"]]
                and a["latest_buffers"] == [(n-1)%3 for n in a["call_counters"]]
                and all(new == old + delta for old, new in zip(b["call_counters"], a["call_counters"]))
                and a["previous_tile_counts"] == expected_tiles
                for b, a in zip(before, after))
            state_passed &= before[0] == before[1] and after[0] == after[1]
            check.update(full_tensor_bitwise_equal_to_baseline=equal, protocol_state_before=before,
                         protocol_state_after=after, protocol_state_passed=state_passed,
                         completed_calls=delta, latest_receive_buffer_per_rank=[(a["call_counters"][0]-1)%3 for a in after])
            check["passed"] &= state_passed
        entry["phases"].append(check)
        checkpoint()
        if not check["passed"]:
            raise RuntimeError(f"{backend}: alternating-shape phase {phase} failed")
    for state in prepared.values():
        state["graph"].reset()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)
    entry.update(passed=True, status="passed")
    checkpoint()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,4,16,64,1024,4096,16384")
    parser.add_argument("--backends", default="nccl,fp4,fp8-sm,fp8-switch,fp8-fused-sm")
    parser.add_argument("--bridge", type=Path, default=Path(os.environ.get("FP8_BRIDGE_LIB", ROOT / "build/fp8_bridge/libfp8_native_bridge.so")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["L2"], default="L2")
    parser.add_argument("--algo", type=int, choices=[1], default=1)
    parser.add_argument("--tune", default="0,0,0,0")
    parser.add_argument("--graph-k", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--stress", type=int, default=100)
    parser.add_argument("--fused-tile", type=int, default=2048)
    parser.add_argument("--fused-ctas", type=int)
    parser.add_argument("--pipeline-tile", type=int, default=8192)
    parser.add_argument("--pipeline-ctas", type=int)
    parser.add_argument("--pipeline-stages", type=int, default=2)
    parser.add_argument("--ue8m0-tile", type=int, default=16384)
    parser.add_argument("--ue8m0-ctas", type=int, default=152)
    parser.add_argument("--ue8m0-threads", type=int, choices=(128, 256), default=128)
    parser.add_argument("--lamport-tile", type=int, default=6144)
    parser.add_argument("--lamport-ctas", type=int, default=152)
    parser.add_argument("--lamport-teams", type=int, default=5)
    args = parser.parse_args()
    try:
        args.rows = [int(value) for value in args.rows.split(",")]
        args.tune = [int(value) for value in args.tune.split(",")]
    except ValueError:
        parser.error("rows and tuning values must be integers")
    args.backends = args.backends.split(",")
    if (not args.rows or any(row <= 0 or row > (2**31 - 1) // HIDDEN for row in args.rows)
            or len(set(args.rows)) != len(args.rows)):
        parser.error("rows must be distinct positive integers within native indexing limits")
    if not args.backends or len(set(args.backends)) != len(args.backends) or not set(args.backends) <= ALL_BACKENDS:
        parser.error("backends must be distinct supported names")
    if not set(args.backends) & EXPERIMENTAL:
        parser.error("select at least one fused, pipelined, or UE8M0 backend")
    if len(args.tune) != 4 or min(args.iters, args.warmup, args.stress) <= 0 or args.graph_k != 10:
        parser.error("four tuning values, positive counts, and graph-k=10 are required")
    if args.fused_tile <= 0 or args.fused_tile > 16384 or args.fused_tile % 512:
        parser.error("fused-tile must be a positive multiple of 512 up to 16384")
    if args.fused_ctas is not None and args.fused_ctas <= 0:
        parser.error("fused-ctas must be positive")
    if args.pipeline_tile <= 0 or args.pipeline_tile > 16384 or args.pipeline_tile % 512:
        parser.error("pipeline-tile must be a positive multiple of 512 up to 16384")
    if args.pipeline_ctas is not None and args.pipeline_ctas <= 0:
        parser.error("pipeline-ctas must be positive")
    if args.pipeline_stages not in (2, 3):
        parser.error("pipeline-stages must be 2 or 3")
    if not 0 < args.ue8m0_tile <= 32768 or args.ue8m0_tile % (128 * (args.ue8m0_threads // 32)):
        parser.error("ue8m0-tile must be at most 32768 and divisible by 128 groups times the warp count")
    if not 1 <= args.ue8m0_ctas <= 152:
        parser.error("ue8m0-ctas must be between 1 and 152")
    if not 256 <= args.lamport_tile <= 16384 or args.lamport_tile % 256:
        parser.error("lamport-tile must be a multiple of 256 between 256 and 16384")
    if not 1 <= args.lamport_ctas <= 152 or not 1 <= args.lamport_teams <= 7:
        parser.error("lamport-ctas must be1..152 and lamport-teams must be1..7")
    return args


def main():
    global torch, dist
    args = parse_args()
    if os.environ.get("FP8_FUSED_DUMP_PTX") == "1":
        dump_dir = args.output.parent / f"ptx-rank{os.environ['RANK']}"
        dump_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
        os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    import torch
    import torch.distributed as dist
    base.torch, base.dist = torch, dist
    from fp8_fused_oracle import ONESHOT_CONTRACT, ONESHOT_CONTRACT_METADATA
    from fp8_oracle import SWITCH_CONTRACT, SWITCH_CONTRACT_METADATA
    paths = ["fp8_fused_compare.py", "fp8_fused_oracle.py", "fp8_compare.py", "fp8_oracle.py",
             "fp8_native_bridge.cu", "fp8_native_bridge.h", "cute_fp8/kernels.py", "cute_fp8/reference.py"]
    if set(args.backends) & EXPERIMENTAL:
        paths.append("cute_fp8/fused.py")
    if "fp8-fused-switch" in args.backends:
        paths.append("cute_fp8/fused_switch.py")
    if "fp8-pipelined-sm" in args.backends:
        paths.append("cute_fp8/pipelined.py")
    if set(args.backends) & UE8M0:
        paths.extend(("cute_fp8/ue8m0.py", "fp8_ue8m0_oracle.py"))
    if set(args.backends) & UE8M0_TUNED:
        paths.append("cute_fp8/ue8m0_tuned.py")
    if set(args.backends) & LAMPORT:
        paths.append("cute_fp8/ue8m0_lamport.py")
    if "fp8-ue8m0-lamport-batched" in args.backends:
        paths.append("cute_fp8/ue8m0_lamport_batched.py")
    hashes = {path: hashlib.sha256((HERE / path).read_bytes()).hexdigest() for path in paths}
    if hashes["cute_fp8/kernels.py"] != base.KERNEL_SHA256:
        raise RuntimeError("Preserved baseline kernel differs from its frozen source")
    if set(args.backends) & UE8M0_EQUIVALENT and hashes["cute_fp8/ue8m0.py"] != "3b92a6af3115b286b4ee3a65f5c9d8a97f6c15c305247999209deba0a4f3b032":
        raise RuntimeError("Preserved UE8M0 baseline differs from its frozen source")
    if (set(args.backends) & LAMPORT and set(args.backends) & UE8M0_TUNED
            and hashes["cute_fp8/ue8m0_tuned.py"] != "cee7951ba00ff8a6c7d2f9b743b7c7033753b891903406c71cc3c82656bb7cb2"):
        raise RuntimeError("Preserved tuned UE8M0 baseline differs from its frozen source")
    if ("fp8-ue8m0-lamport-batched" in args.backends
            and hashes["cute_fp8/ue8m0_lamport.py"] != "0e7896f7820f18bae7bcc9a76d3724ff8c0ac13bd8fa7a7eb0711a0c29790fc9"):
        raise RuntimeError("Preserved original Lamport baseline differs from its frozen source")
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=8))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise RuntimeError("This driver and its numerical contracts require TP2")
    host_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(minutes=8))
    started = time.time()
    report = dict(study="fp8_fused_compare_v1", status="running", correctness_passed=False,
        tp_size=world, rows=args.rows, backends=args.backends, hidden=HIDDEN, group_size=GROUP_SIZE,
        kernel_sha256=base.KERNEL_SHA256, source_sha256=hashes,
        upstream_commit="2878bf50e20241d55500c2df76b87f0a81ef1c0f", graph_k=args.graph_k,
        iters=args.iters, warmup=args.warmup, fp4_variant="L2", fp4_algo=1, fp4_tune=args.tune,
        fused_config=dict(tile_elems=args.fused_tile, ctas=args.fused_ctas),
        oneshot_contract=ONESHOT_CONTRACT, oneshot_contract_metadata=ONESHOT_CONTRACT_METADATA,
        switch_contract=SWITCH_CONTRACT, switch_contract_metadata=SWITCH_CONTRACT_METADATA,
        timing_contract="BF16 input/output, all quantization and synchronization included; graph10; out-of-place NCCL; excluded rank alignment; max rank per sample then upper median",
        numerical_scope="Distinct codecs: fused SM one-shot strict bitwise independent oracle; old SM has a second FP8 quantization; switches use empirical interval checks. Sampled complete-group codec checks plus full finite/rank/graph/reuse checks; no model accuracy qualification.",
        results=[], protocol_checks=[], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if set(args.backends) & PIPELINED:
        report["pipeline_config"] = dict(tile_elems=args.pipeline_tile, ctas=args.pipeline_ctas,
                                         pipeline_stages=args.pipeline_stages)
    if set(args.backends) & UE8M0_TUNED:
        report["ue8m0_tuned_config"] = dict(tile_elems=args.ue8m0_tile, ctas=args.ue8m0_ctas,
                                             threads=args.ue8m0_threads)
    if set(args.backends) & LAMPORT:
        report["lamport_config"] = dict(tile_elems=args.lamport_tile, ctas=args.lamport_ctas,
                                         teams=args.lamport_teams)
    if set(args.backends) & UE8M0:
        from fp8_ue8m0_oracle import UE8M0_CONTRACT, UE8M0_CONTRACT_METADATA
        report.update(ue8m0_contract=UE8M0_CONTRACT, ue8m0_contract_metadata=UE8M0_CONTRACT_METADATA,
            ue8m0_config=dict(tile_elems=args.fused_tile, ctas=args.fused_ctas),
            ue8m0_comparison_scope="Intentional power-of-two scale quantization; exact UE8M0 oracle required, no bitwise equivalence to FP32-scale fused SM",
            ue8m0_timing_scope="Full BF16-to-BF16 call, including amax, UE8M0 scale encoding, FP8 quantization, transport, reduction, decode and synchronization",
            ue8m0_error_scope="Full-output max absolute, RMSE and relative L2 error versus the original BF16 inputs summed in FP32")

    def checkpoint():
        if rank == 0:
            report["elapsed_s"] = time.time() - started
            base.write_json(args.output, report)

    checkpoint()
    try:
        native = base.Native(args.bridge, rank, world, device, max(args.rows), "L2", host_group)
        runtime = dict(rank=rank, device=device, name=torch.cuda.get_device_name(), torch=torch.__version__,
            cuda=torch.version.cuda, cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
            nccl_loaded_version=native.version, nccl_torch_query=torch.cuda.nccl.version(),
            loaded_nccl_maps=[line for line in Path("/proc/self/maps").read_text().splitlines() if "libnccl" in line])
        report["runtime"] = base.gather_objects(runtime, host_group)
        workspaces = {}
        for backend in args.backends:
            params = dict(max_numel=max(args.rows) * HIDDEN, device=torch.device("cuda", device), group_size=GROUP_SIZE)
            if backend in ("fp8-sm", "fp8-switch"):
                from cute_fp8.kernels import FP8AllReduceWorkspace
                workspaces[backend] = FP8AllReduceWorkspace(**params, backend=backend[4:])
            elif backend == "fp8-fused-sm":
                from cute_fp8.fused import FusedFP8AllReduceWorkspace
                workspaces[backend] = FusedFP8AllReduceWorkspace(**params, tile_elems=args.fused_tile, ctas=args.fused_ctas)
            elif backend == "fp8-fused-switch":
                from cute_fp8.fused_switch import FusedSwitchFP8AllReduceWorkspace
                # Switch tile is independently fixed; SM tile tuning must not
                # silently alter multicast shape/padding semantics.
                workspaces[backend] = FusedSwitchFP8AllReduceWorkspace(**params, tile_elems=2048, ctas=None)
            elif backend == "fp8-pipelined-sm":
                from cute_fp8.pipelined import PipelinedFP8AllReduceWorkspace
                workspaces[backend] = PipelinedFP8AllReduceWorkspace(**params,
                    tile_elems=args.pipeline_tile, ctas=args.pipeline_ctas, pipeline_stages=args.pipeline_stages)
            elif backend == "fp8-ue8m0-sm":
                from cute_fp8.ue8m0 import UE8M0FP8AllReduceWorkspace
                workspaces[backend] = UE8M0FP8AllReduceWorkspace(**params,
                    tile_elems=args.fused_tile, ctas=args.fused_ctas)
            elif backend == "fp8-ue8m0-tuned":
                from cute_fp8.ue8m0_tuned import TunedUE8M0FP8AllReduceWorkspace
                workspaces[backend] = TunedUE8M0FP8AllReduceWorkspace(**params,
                    tile_elems=args.ue8m0_tile, ctas=args.ue8m0_ctas, threads=args.ue8m0_threads)
            elif backend == "fp8-ue8m0-lamport":
                from cute_fp8.ue8m0_lamport import LamportUE8M0FP8AllReduceWorkspace
                workspaces[backend] = LamportUE8M0FP8AllReduceWorkspace(**params,
                    tile_elems=args.lamport_tile, ctas=args.lamport_ctas, teams=args.lamport_teams)
            elif backend == "fp8-ue8m0-lamport-batched":
                from cute_fp8.ue8m0_lamport_batched import BatchedLamportUE8M0FP8AllReduceWorkspace
                workspaces[backend] = BatchedLamportUE8M0FP8AllReduceWorkspace(**params,
                    tile_elems=args.lamport_tile, ctas=args.lamport_ctas, teams=args.lamport_teams)
        report["workspaces"] = {backend: workspace.metadata for backend, workspace in workspaces.items()}
        pipeline_baseline = None
        if set(args.backends) & PIPELINED:
            pipeline_baseline = workspaces.get("fp8-fused-sm")
            if pipeline_baseline is None:
                from cute_fp8.fused import FusedFP8AllReduceWorkspace
                pipeline_baseline = FusedFP8AllReduceWorkspace(
                    **params, tile_elems=args.fused_tile, ctas=args.fused_ctas)
            report["pipeline_baseline_workspace"] = pipeline_baseline.metadata
        ue8m0_baseline = None
        if set(args.backends) & UE8M0_EQUIVALENT:
            ue8m0_baseline = workspaces.get("fp8-ue8m0-sm")
            if ue8m0_baseline is None:
                from cute_fp8.ue8m0 import UE8M0FP8AllReduceWorkspace
                ue8m0_baseline = UE8M0FP8AllReduceWorkspace(
                    **params, tile_elems=args.fused_tile, ctas=args.fused_ctas)
            if set(args.backends) & UE8M0_TUNED:
                report["ue8m0_tuned_baseline_workspace"] = ue8m0_baseline.metadata
            if set(args.backends) & LAMPORT:
                report["lamport_baseline_workspace"] = ue8m0_baseline.metadata
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
                    run_cell(rows, backend, native, workspaces.get(backend), args, host_group, checkpoint, record,
                             baseline_workspace=ue8m0_baseline if backend in UE8M0_EQUIVALENT else pipeline_baseline)
                    if rank == 0:
                        print(f"PASS {backend} rows={rows} median_us={record['timing']['median_us']:.3f}", flush=True)
            for backend in args.backends:
                if backend not in EXPERIMENTAL:
                    continue
                for name, function in (("adversarial_codec", adversarial_check), ("alternating_shape_workspace", alternating_shape_check)):
                    entry = dict(name=name, backend=backend, status="running", passed=False)
                    report["protocol_checks"].append(entry)
                    checkpoint()
                    if backend in LAMPORT and name == "alternating_shape_workspace":
                        function(backend, workspaces[backend], native, args, host_group, checkpoint, entry,
                                 baseline_workspace=ue8m0_baseline)
                    else:
                        function(backend, workspaces[backend], native, args, host_group, checkpoint, entry)
        native.call("destroy", native.handle)
        report.update(status="passed", correctness_passed=True)
        checkpoint()
        dist.destroy_process_group()
    except BaseException as error:
        for record in report["results"] + report["protocol_checks"]:
            if record["status"] == "running":
                record["status"] = "failed"
        report.update(status="failed", correctness_passed=False, error=repr(error), traceback=traceback.format_exc())
        checkpoint()
        raise


if __name__ == "__main__":
    main()
