#!/usr/bin/env python3
"""Numerical-only TP2 comparison on the original CUDA-generated BF16 inputs.

Run with torchrun --nproc-per-node=2. BF16 NCCL, repository NVFP4 L2, and
group128 UE8M0 run their actual allreduce implementations. MXFP8 E4M3/group32
is a local quantize/dequantize/sum REFERENCE, not a communication kernel and
has no performance result. All outputs are BF16 and use the same FP32 target.

MXFP8 uses NVIDIA TensorRT's documented round-up recipe:
  s = 2**max(-127, ceil(log2(amax/448))), encoded as E8M0 byte exponent+127.
The E8M0 minimum scale is 2**-127; zero groups use byte0, not a zero scale.
OCP specifies the group32 format but leaves detailed scale algorithms open.
This is not the max-exponent/floor recipe in Microsoft's MX emulator.

--self-test needs NumPy only. CUDA input generation is deliberately NOT
reimplemented using NumPy log/cos, which would not establish identical bytes.
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
import traceback

import numpy as np

try:
    from .cute_fp8.reference import bf16_round, fp8_e4m3fn_encode, fp8_e4m3fn_decode
except ImportError:
    from cute_fp8.reference import bf16_round, fp8_e4m3fn_encode, fp8_e4m3fn_decode

HERE = Path(__file__).resolve().parent
HIDDEN = 6144
MXFP8_CONTRACT = "tp2_mxfp8_e4m3_group32_e8m0_round_up_reference_v1"
MXFP8_METADATA = dict(
    format="OCP MXFP8 E4M3", group_size=32, element_format="E4M3FN",
    scale_format="E8M0", recipe="NVIDIA TensorRT round-up-to-E8M0",
    scale_rule="e=max(-127,ceil(log2(amax/448))); scale=2**e; byte=e+127",
    scale_floor_exponent=-127, zero_scale_code=0,
    input_output_dtype="BF16", rounding="E4M3 RNE with finite saturation; BF16 RNE",
    accumulation="separate FP32 multiply/add; +0 seed; source0 then source1",
    second_output_quantization=False, communication_kernel=False,
    performance_measured=False, model_accuracy_qualified=False,
    sources=[
        "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf",
        "https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html#mx-compliant-dynamic-quantization",
    ],
)


def mxfp8_scales_numpy(amax):
    """Independent frexp/float64 threshold oracle, with true E8M0 code0."""
    amax = np.asarray(amax, dtype=np.float64)
    if not np.isfinite(amax).all() or (amax < 0).any() or (amax > np.finfo(np.float32).max).any():
        raise ValueError("amax must be finite, nonnegative, and within FP32 range")
    _, power = np.frexp(amax)
    exponent = power.astype(np.int32) - 9
    threshold = np.ldexp(np.full(amax.shape, 448.0, dtype=np.float64), exponent)
    exponent += (amax > threshold).astype(np.int32)
    exponent = np.where(amax == 0, -127, np.maximum(exponent, -127)).astype(np.int32)
    codes = (exponent + 127).astype(np.uint8)
    # Bit-shifting code0 into an FP32 exponent would produce ZERO, incorrectly.
    scales = np.ldexp(np.ones(amax.shape, dtype=np.float32), exponent)
    reciprocals = np.ldexp(np.ones(amax.shape, dtype=np.float32), -exponent)
    return codes, scales, reciprocals


def mxfp8_reference_numpy(inputs):
    """Complete group32 QDQ source-local scales, rank-ordered sum, BF16 out."""
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[0] != 2 or not inputs.shape[1] or inputs.shape[1] % 32:
        raise ValueError("requires nonempty [2,N] with N divisible by32")
    if not np.isfinite(inputs).all():
        raise ValueError("finite BF16 source inputs required")
    blocks = bf16_round(inputs).reshape(2, -1, 32)
    if not np.isfinite(blocks).all():
        raise ValueError("source overflow in BF16 conversion")
    codes, scales, reciprocals = mxfp8_scales_numpy(np.max(np.abs(blocks), axis=2))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = np.multiply(blocks, reciprocals[:, :, None], dtype=np.float32)
        q = fp8_e4m3fn_encode(np.clip(normalized, np.float32(-448), np.float32(448)))
        products = np.multiply(fp8_e4m3fn_decode(q), scales[:, :, None], dtype=np.float32)
        total = np.zeros(blocks.shape[1:], dtype=np.float32)
        total = np.add(total, products[0], dtype=np.float32)
        total = np.add(total, products[1], dtype=np.float32)
        output = bf16_round(total).reshape(-1)
    return dict(output=output, q=q.reshape(inputs.shape), scale_codes=codes,
                scales=scales, finite=bool(np.isfinite(products).all() and np.isfinite(output).all()))


def edge_inputs():
    """Signed zeros, group maximum threshold, E8M0 floor, and wide exponents."""
    blocks = np.zeros((2, 9, 32), dtype=np.float32)
    blocks[:, 1, 1::2] = np.float32(-0.0)
    for group, value in enumerate([448., 450., 1., 2.**-133, 2.**-127, 2.**-120, 2.**100], 2):
        blocks[0, group] = np.float32(value)
        blocks[0, group, 1::2] *= np.float32(-1)
        blocks[1, group] = blocks[0, group] * np.float32(-0.5)
    return bf16_round(blocks).reshape(2, -1)


def self_test():
    codes, scales, reciprocal = mxfp8_scales_numpy([0., 448., 450., 2.**-133, 2.**100])
    assert codes.tolist() == [0, 127, 128, 0, 219]
    assert scales[0] == np.float32(2.**-127) and reciprocal[0] == np.float32(2.**127)
    fixture = edge_inputs()
    result = mxfp8_reference_numpy(fixture)
    assert result["finite"]
    assert np.all(result["q"][:, 32:64:2] == 0)
    assert np.all(result["q"][:, 33:64:2] == 128)
    assert result["scale_codes"][0, 2] == 127 and result["scale_codes"][0, 3] == 128
    # 0xFF is never emitted for any finite BF16 input after explicit saturation.
    values = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    values = values[np.isfinite(values)]
    amax = np.abs(values).astype(np.float64)
    codes, scales, _ = mxfp8_scales_numpy(amax)
    assert np.all(codes <= 247) and np.all(scales > 0)
    target = np.maximum(amax / 448., 2.**-127)
    assert np.all(scales.astype(np.float64) >= target)
    assert np.all((scales.astype(np.float64) / 2 < target) | (codes == 0))
    return dict(passed=True, finite_bf16_patterns=int(values.size),
                edge_groups=9, zero_scale_code=0, negative_zero_preserved=True,
                scope="CPU scale/codec reference checks; GPU casts require separate qualification")


def mxfp8_reference_torch(sources):
    """Untimed local GPU reference; no distributed collective is performed."""
    blocks = sources.float().reshape(2, -1, 32)
    amax = blocks.abs().amax(dim=2)
    mantissa, power = torch.frexp(amax)
    exponent = (power - 9 + (mantissa > 0.875).to(torch.int32)).clamp(min=-127)
    exponent = torch.where(amax == 0, -127, exponent)
    scales = torch.ldexp(torch.ones_like(amax), exponent)
    reciprocal = torch.ldexp(torch.ones_like(amax), -exponent)
    q = (blocks * reciprocal.unsqueeze(2)).clamp(-448, 448).to(torch.float8_e4m3fn)
    # Separate eager operators make the FP32 multiply/add boundaries explicit.
    total = torch.zeros_like(blocks[0])
    total.add_(q[0].float() * scales[0].unsqueeze(1))
    total.add_(q[1].float() * scales[1].unsqueeze(1))
    output = total.bfloat16().reshape(sources.shape[1:])
    return dict(output=output, q=q.reshape(2, -1).view(torch.uint8),
                scale_codes=(exponent + 127).to(torch.uint8), scales=scales)


def sampled_mxfp8_check(sources, result, seed, all_groups=False):
    from fp8_oracle import sample_group_indices
    n = sources[0].numel()
    groups = list(range(n // 32)) if all_groups else sample_group_indices(n, input_seed=seed, group_size=32)
    indices = torch.tensor(groups, dtype=torch.long, device=sources.device)
    inputs = sources.reshape(2, -1, 32).index_select(1, indices).flatten(1).float().cpu().numpy()
    expected = mxfp8_reference_numpy(inputs)
    q = result["q"].reshape(2, -1, 32).index_select(1, indices).flatten(1).cpu().numpy()
    codes = result["scale_codes"].index_select(1, indices).cpu().numpy()
    scales = result["scales"].index_select(1, indices).cpu().numpy()
    output = result["output"].reshape(-1, 32).index_select(0, indices).flatten().float().cpu().numpy()
    counts = dict(q_byte_mismatches=int(np.count_nonzero(q != expected["q"])),
                  scale_byte_mismatches=int(np.count_nonzero(codes != expected["scale_codes"])),
                  scale_fp32_bit_mismatches=int(np.count_nonzero(scales.view(np.uint32) != expected["scales"].view(np.uint32))),
                  output_bf16_bit_mismatches=int(np.count_nonzero(output.view(np.uint32) != expected["output"].view(np.uint32))))
    return dict(passed=expected["finite"] and not any(counts.values()), **counts,
                contract=MXFP8_CONTRACT, sampled_group_indices=groups,
                sampled_numel=len(groups) * 32, independent_reference="NumPy frexp and finite E4M3 codebook RNE",
                sampled_input_bf16_sha256=hashlib.sha256((inputs.view(np.uint32) >> 16).astype('<u2').tobytes()).hexdigest())


def main():
    global torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--rows', default='1,4,16,64,1024,4096,16384')
    parser.add_argument('--bridge', default=os.environ.get('FP8_BRIDGE_LIB', str(HERE.parent / 'build/fp8_bridge/libfp8_native_bridge.so')))
    parser.add_argument('--output')
    parser.add_argument('--lamport-tile', type=int, default=2048)
    parser.add_argument('--lamport-ctas', type=int, default=152)
    parser.add_argument('--lamport-teams', type=int, default=7)
    args = parser.parse_args()
    cpu_check = self_test()
    if args.self_test:
        print(cpu_check)
        return
    if not args.output:
        parser.error('--output is required for a GPU probe')
    rows_list = [int(x) for x in args.rows.split(',')]
    if not rows_list or len(set(rows_list)) != len(rows_list) or any(x < 1 or x > 16384 for x in rows_list):
        parser.error('rows must be distinct integers in1..16384')
    import torch
    import torch.distributed as dist
    import fp8_compare as base
    import fp8_fused_compare as fused
    from cute_fp8.ue8m0_lamport_batched import (
        BatchedLamportUE8M0FP8AllReduceWorkspace, BatchedLamportUE8M0FP8AllReduce)
    base.torch, base.dist = torch, dist
    fused.torch, fused.dist = torch, dist
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=8))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise RuntimeError('probe requires exactly two ranks')
    host = dist.new_group(backend='gloo', timeout=datetime.timedelta(minutes=8))
    paths = ['fp8_format_accuracy.py', 'fp8_compare.py', 'fp8_fused_compare.py', 'fp8_oracle.py',
             'fp8_ue8m0_oracle.py', 'cute_fp8/reference.py', 'cute_fp8/kernels.py',
             'cute_fp8/ue8m0.py', 'cute_fp8/ue8m0_lamport_batched.py', 'ref.cuh', 'fp8_native_bridge.cu',
             '../include/fp4_codec.cuh', '../include/codec_fast.cuh', '../kernels/push/tp2_l2.cu']
    hashes = {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in paths}
    report = dict(study='tp2_four_format_accuracy_v1', status='running', correctness_passed=False,
        performance_measured=False, model_accuracy_qualified=False, job_id=os.environ.get('SLURM_JOB_ID'),
        rows=rows_list, hidden=HIDDEN, tp_size=2, input_sets=3,
        seed_rule='(1000 + 17*rows + 7777*input_set) modulo2**32; same seed for both source ranks',
        input_generator='Unmodified CUDA ref::gen_input + ref::gen_outliers via native bridge; never NumPy Gaussian',
        reference='Unquantized BF16 source inputs summed into FP32 in rank order with +0 seed',
        metric='FP32 Torch elementwise squared error and mean, matching archived benchmark metric',
        mxfp8_reference=MXFP8_METADATA, source_sha256=hashes, cpu_self_test=cpu_check,
        bridge_binary_sha256=hashlib.sha256(Path(args.bridge).read_bytes()).hexdigest(),
        results=[], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def checkpoint():
        if rank == 0:
            base.write_json(args.output, report)
    checkpoint()
    native = None
    try:
        native = base.Native(args.bridge, rank, world, device, max(rows_list), 'L2', host)
        report['runtime'] = base.gather_objects(dict(rank=rank, device=device,
            name=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
            cute_dsl=importlib.metadata.version('nvidia-cutlass-dsl'), nccl_loaded_version=native.version), host)
        workspace = BatchedLamportUE8M0FP8AllReduceWorkspace(max(rows_list) * HIDDEN,
            torch.device('cuda', device), tile_elems=args.lamport_tile,
            ctas=args.lamport_ctas, teams=args.lamport_teams)
        report['ue8m0_workspace'] = workspace.metadata
        edges = torch.from_numpy(edge_inputs()).to(device='cuda', dtype=torch.bfloat16)
        edge_result = mxfp8_reference_torch(edges)
        report['mxfp8_gpu_edge_checks'] = base.gather_objects(sampled_mxfp8_check(edges, edge_result, 0, True), host)
        if not all(x['passed'] for x in report['mxfp8_gpu_edge_checks']):
            raise RuntimeError('MXFP8 GPU/CPU edge-case mismatch')
        del edges, edge_result
        tune = (C.c_int * 4)(0, 0, 0, 0)
        for rows in rows_list:
            sources = torch.empty((2, rows, HIDDEN), dtype=torch.bfloat16, device='cuda')
            x = torch.empty_like(sources[rank])
            out, codec = torch.empty_like(x), torch.empty_like(x)
            fp32 = torch.empty_like(x, dtype=torch.float32)
            op = BatchedLamportUE8M0FP8AllReduce(x, workspace=workspace)
            for input_set in range(3):
                seed = (1000 + 17 * rows + 7777 * input_set) & 0xffffffff
                native.generate(sources, rows, seed)
                x.copy_(sources[rank])
                native.reference(sources, codec, fp32, rows, 1)
                for backend in ('nccl', 'fp4', 'fp8-ue8m0-lamport-batched'):
                    out.fill_(float('nan'))
                    if backend == 'fp8-ue8m0-lamport-batched':
                        op(x, out=out)
                    else:
                        native.launch(backend, x, out, rows, 1, tune)
                    torch.cuda.synchronize()
                    check = fused.validate(backend, op if backend.startswith('fp8-') else None,
                                           sources, out, codec, fp32, seed, host)
                    record = dict(rows=rows, numel=rows*HIDDEN, input_set=input_set, seed=seed,
                        backend=backend, kind='actual_allreduce_output', performance_measured=False,
                        rank_checks=base.gather_objects(check, host))
                    record['passed'] = all(c['passed'] for c in record['rank_checks'])
                    report['results'].append(record)
                    checkpoint()
                    if not record['passed']:
                        raise RuntimeError(f'actual backend numerical gate failed: {backend}/{rows}/{input_set}')
                mx = mxfp8_reference_torch(sources)
                torch.cuda.synchronize()
                sampled = sampled_mxfp8_check(sources, mx, seed)
                peer = mx['output'].view(torch.uint8).clone()
                dist.broadcast(peer, src=0)
                finite = bool(torch.isfinite(mx['output']).all().item())
                rank_equal = base.bitwise_equal(peer, mx['output'].view(torch.uint8))
                error = mx['output'].float() - fp32
                mse, ref_mse = error.square().mean().item(), fp32.square().mean().item()
                metrics = dict(max_abs=error.abs().max().item(), rmse=math.sqrt(mse),
                    relative_l2=math.sqrt(mse/ref_mse) if ref_mse else math.sqrt(mse))
                check = dict(passed=finite and rank_equal and sampled['passed'], finite=finite,
                    full_tensor_rank_bitwise_equal=rank_equal, numerical=sampled,
                    error_vs_original_fp32_sum=metrics)
                record = dict(rows=rows, numel=rows*HIDDEN, input_set=input_set, seed=seed,
                    backend='mxfp8-group32-reference', kind='local_numerical_reference_no_communication',
                    performance_measured=False, rank_checks=base.gather_objects(check, host))
                record['passed'] = all(c['passed'] for c in record['rank_checks'])
                report['results'].append(record)
                checkpoint()
                if not record['passed']:
                    raise RuntimeError(f'MXFP8 reference numerical gate failed: {rows}/{input_set}')
                if rank == 0:
                    print(f'PASS four formats rows={rows} set={input_set} MXFP8 relative_l2={metrics["relative_l2"]:.9g}', flush=True)
                del mx, peer, error
            del sources, x, out, codec, fp32, op
        native.call('destroy', native.handle)
        native = None
        report.update(status='passed', correctness_passed=True,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        checkpoint()
        dist.destroy_process_group()
    except BaseException as error:
        report.update(status='failed', correctness_passed=False, error=repr(error), traceback=traceback.format_exc())
        checkpoint()
        # Avoid asymmetric collective cleanup after exceptions; torchrun owns it.
        raise


if __name__ == '__main__':
    main()
