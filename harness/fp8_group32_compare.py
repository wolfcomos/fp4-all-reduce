#!/usr/bin/env python3
"""GB200 TP2 BF16/NVFP4/group128-FP8/standard-group32-MXFP8 comparison.

The original driver owns capture, event timing, changing-input checks and stress.
This separate driver installs explicit operation/validation dispatch adapters in
memory; no original source is edited. References run only outside timed graphs.
Use --validate REPORT with the same configuration for a CPU-only report audit.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HIDDEN = 6144
MX = 'fp8-mxfp8-lamport-batched'
UE = 'fp8-ue8m0-lamport-batched'
BACKENDS = ['nccl', 'fp4', UE, MX]
MX_CONTRACT = 'tp2_mxfp8_e4m3_group32_e8m0_round_up_reference_v1'
FROZEN = {
    'fp8_compare.py': '16fa660922f6de77cebe5611159cee4c2e57f945fb1bfe865d58f09848cfec98',
    'fp8_fused_compare.py': 'a2b1be2557d1f1b1fa2532b8b0acecbba44c48bef25e6770203c5e27289e212e',
    'fp8_format_accuracy.py': 'efc5ec1efa68eea6e730a3f8f1d20cdb4010db9755c0411669f14a35c51bcdd4',
    'cute_fp8/kernels.py': '2a008f18fb171d6d91c6c7a78523f7328470d163bcb10fca2ae011f0a1e4d07f',
    'cute_fp8/ue8m0.py': '3b92a6af3115b286b4ee3a65f5c9d8a97f6c15c305247999209deba0a4f3b032',
    'cute_fp8/ue8m0_lamport_batched.py': 'c02631c399c1df8911c55b2ebd15ee7c2a4cbba600b3be2decdbdf4fe36081cb',
    'cute_fp8/reference.py': 'fd079bbc21d25d101589e2a26103b56245891d94b202f9bf70defc505d7a081e',
    'fp8_ue8m0_oracle.py': '3f829dfa52af4076d59643718342b3ce97ceb6a1eae31ec1ceb906fb5e3a02c6',
}
SOURCE_NAMES = sorted(set(FROZEN) | {'fp8_group32_compare.py', 'cute_fp8/mxfp8_lamport_batched.py',
    'cute_fp8/fused.py', 'fp8_oracle.py', 'fp8_fused_oracle.py', 'fp8_native_bridge.cu',
    'fp8_native_bridge.h', 'ref.cuh', '../include/fp4_codec.cuh', '../include/codec_fast.cuh',
    '../kernels/push/tp2_l2.cu'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_operation(backend, x, out, workspace, native, rows, args):
    if backend != MX:
        return original_make_operation(backend, x, out, workspace, native, rows, args)
    from cute_fp8.mxfp8_lamport_batched import MXFP8LamportBatchedAllReduce
    op = MXFP8LamportBatchedAllReduce(x, group_size=32, workspace=workspace)
    return op, lambda: op(x, out=out)


def full_mxfp8_reference(sources):
    """Bound temporary GPU memory; every group uses the frozen reference math."""
    flat = sources.flatten(1)
    out = torch.empty_like(sources[0])
    for start in range(0, flat.shape[1], 1024 * HIDDEN):
        end = min(start + 1024 * HIDDEN, flat.shape[1])
        reference = oracle.mxfp8_reference_torch(flat[:, start:end])
        out.flatten()[start:end].copy_(reference['output'])
        del reference
    return out


def numerical_mxfp8(op, sources, out, seed, host_group):
    import numpy as np
    from fp8_oracle import sample_group_indices
    groups = (list(range(out.numel() // 32)) if out.numel() <= HIDDEN else
              sample_group_indices(out.numel(), input_seed=seed, group_size=32))
    indices = torch.tensor(groups, device=out.device, dtype=torch.long)
    sampled_inputs = sources.reshape(2, -1, 32).index_select(1, indices).flatten(1).float().cpu().numpy()
    expected = oracle.mxfp8_reference_numpy(sampled_inputs)
    sample = out.reshape(-1, 32).index_select(0, indices).flatten()
    actual = legacy.gather_tensor(sample).float().cpu().numpy()
    output_bad = np.count_nonzero(actual.view(np.uint32) != expected['output'][None].view(np.uint32), axis=1).tolist()
    received = op.sample_received_groups(indices)
    if received['source_rank'] != 1 - dist.get_rank():
        raise RuntimeError('Group32 wire source-rank attribution differs')
    q = legacy.gather_tensor(received['q'].reshape(-1)).cpu().numpy()
    codes = legacy.gather_tensor(received['scale_codes'].reshape(-1)).cpu().numpy()
    expected_q, expected_codes = expected['q'][[1, 0]], expected['scale_codes'][[1, 0]]
    q_bad = np.count_nonzero(q != expected_q, axis=1).tolist()
    code_bad = np.count_nonzero(codes != expected_codes, axis=1).tolist()
    decoded = np.ldexp(np.ones(codes.shape, dtype=np.float32), codes.astype(np.int32) - 127)
    scale_bad = np.count_nonzero(decoded.view(np.uint32) != expected['scales'][[1, 0]].view(np.uint32), axis=1).tolist()
    reference = full_mxfp8_reference(sources)
    reference_finite = bool(torch.isfinite(reference).all().item())
    mismatch = int((out.view(torch.int16) != reference.view(torch.int16)).sum().item())
    full_counts = base.gather_objects(mismatch, host_group)
    reference_finite = base.all_passed(reference_finite, host_group)
    del reference
    payload = dict(passed=not any(q_bad + code_bad + scale_bad), source_order_per_receiving_rank=[1, 0],
        quantized_byte_mismatches_per_rank=q_bad, scale_code_byte_mismatches_per_rank=code_bad,
        decoded_scale_bit_mismatches_per_rank=scale_bad,
        negative_zero_byte_count_per_rank=np.count_nonzero(q == 128, axis=1).tolist(),
        expected_negative_zero_byte_count_per_rank=np.count_nonzero(expected_q == 128, axis=1).tolist(),
        scale_floor_count_per_rank=np.count_nonzero(codes == 0, axis=1).tolist(),
        expected_scale_floor_count_per_rank=np.count_nonzero(expected_codes == 0, axis=1).tolist(),
        decoded_scale_scope='Independent CPU E8M0 decode; code0 is 2^-127, not zero')
    return dict(passed=expected['finite'] and reference_finite and not any(output_bad + full_counts)
                and payload['passed'], contract=MX_CONTRACT, comparison='exact_bf16_bits',
        sampled_group_indices=groups, sampled_numel=len(groups)*32,
        sampled_output_bit_mismatches_per_rank=output_bad, received_payload=payload,
        independent_reference='NumPy frexp and finite E4M3 codebook RNE',
        sampled_input_bf16_sha256=hashlib.sha256((sampled_inputs.view(np.uint32) >> 16).astype('<u2').tobytes()).hexdigest(),
        reference_finite=reference_finite, full_output_gpu_reference=dict(passed=reference_finite and not any(full_counts),
            mismatch_count_per_rank=full_counts, checked_numel=out.numel(), untimed=True,
            reference='Frozen mxfp8_reference_torch; separate FP32 multiply/add, +0 seed, rank0 then rank1; BF16 RNE'),
        second_output_quantization=False)


def validate(backend, op, sources, out, codec, fp32, seed, host_group):
    if backend != MX:
        return original_validate(backend, op, sources, out, codec, fp32, seed, host_group)
    finite = bool(torch.isfinite(out).all().item())
    peer = out.view(torch.uint8).clone()
    dist.broadcast(peer, src=0)
    equal = base.bitwise_equal(peer, out.view(torch.uint8))
    del peer
    numerical = numerical_mxfp8(op, sources, out, seed, host_group)
    result = dict(passed=base.all_passed(finite and equal and numerical['passed'], host_group),
        finite=finite, full_tensor_rank_bitwise_equal=equal, numerical=numerical)
    if fp32 is not None:
        error = out.float() - fp32
        mse, reference_mse = error.square().mean().item(), fp32.square().mean().item()
        result['error_vs_original_fp32_sum'] = dict(max_abs=error.abs().max().item(), rmse=math.sqrt(mse),
            relative_l2=math.sqrt(mse/reference_mse) if reference_mse else math.sqrt(mse))
    return result


def scale_edges(workspace, native, args, host_group, checkpoint, entry):
    """True minimum BF16 subnormal and E8M0 code0; no overflowing square metrics."""
    import numpy as np
    edge_inputs = oracle.edge_inputs()
    sources = torch.from_numpy(edge_inputs).to(device='cuda', dtype=torch.bfloat16)
    bits_preserved = bool(np.array_equal(sources.float().cpu().numpy().view(np.uint32), edge_inputs.view(np.uint32)))
    if not base.all_passed(bits_preserved, host_group):
        raise RuntimeError('GPU edge input transfer changed BF16 subnormal or signed-zero bits')
    x, out = sources[dist.get_rank()].clone(), torch.empty_like(sources[0])
    op, operation = make_operation(MX, x, out, workspace, native, 0, args)
    graph = base.capture(operation, args.graph_k, native, host_group)
    entry.update(numel=288, group_size=32, cases=9, checked_all_groups=True, graph_k=10,
        input_bf16_bits_preserved=True,
        output_poisoned_between_calls=True, metrics_omitted='2^100 edge values overflow FP32 squared metrics; byte/output gates still mandatory', checks=[])
    for mode in ('eager', 'graph'):
        out.fill_(float('nan'))
        operation() if mode == 'eager' else graph.replay()
        torch.cuda.synchronize()
        check = validate(MX, op, sources, out, None, None, 0, host_group)
        check['mode'] = mode
        if mode == 'eager':
            eager = out.clone()
        else:
            check['bitwise_equal_to_eager'] = base.all_passed(base.bitwise_equal(out, eager), host_group)
            check['passed'] &= check['bitwise_equal_to_eager']
        entry['checks'].append(check)
        checkpoint()
        if not check['passed']:
            raise RuntimeError('MXFP8 scale/subnormal/signed-zero edge gate failed')
    graph.reset()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)
    entry.update(status='passed', passed=True)
    checkpoint()


def alternating_mxfp8(workspace, native, args, host_group, checkpoint, entry):
    choices = sorted(set((min(args.rows), args.rows[len(args.rows)//2], max(args.rows))))
    entry.update(shapes=choices, phases=[], preparation_checks=[], graph_k=10, same_workspace=True,
        triple_buffer_wrap=True, receive_buffers=3, rotating_input_sets=5, skew_sleep_cycles=10000,
        output_poisoned_between_calls=True, reference='full_tensor_group32_gpu_reference',
        bounded_smoke_not_full_upstream_protocol_suite=True)
    prepared = {}
    rank = dist.get_rank()
    for rows in choices:
        sources = torch.empty((2, rows, HIDDEN), dtype=torch.bfloat16, device='cuda')
        x, out = torch.empty_like(sources[rank]), torch.empty_like(sources[rank])
        fp32 = torch.empty_like(x, dtype=torch.float32)
        op, operation = make_operation(MX, x, out, workspace, native, rows, args)
        native.generate(sources, rows, 1000+17*rows)
        x.copy_(sources[rank])
        graph = base.capture(operation, 10, native, host_group)
        expected = []
        for index in range(5):
            seed = 1000+17*rows+7777*index
            native.generate(sources, rows, seed)
            x.copy_(sources[rank])
            native.reference(sources, None, fp32, rows, 1)
            out.fill_(float('nan'))
            operation()
            torch.cuda.synchronize()
            check = validate(MX, op, sources, out, None, fp32, seed, host_group)
            check.update(rows=rows, input_set=index, seed=seed)
            entry['preparation_checks'].append(check)
            checkpoint()
            if not check['passed']:
                raise RuntimeError('Group32 alternating-shape preparation failed')
            expected.append(full_mxfp8_reference(sources))
        prepared[rows] = dict(sources=sources, x=x, out=out, fp32=fp32, op=op,
                              operation=operation, graph=graph, expected=expected)
    small, middle, large = choices[0], choices[len(choices)//2], choices[-1]
    sequence = [large, small, middle, small, large, middle, small, large, middle, large, small, large]
    for phase, rows in enumerate(sequence):
        state = prepared[rows]
        index, seed = phase % 5, 1000+17*rows+7777*(phase % 5)
        native.generate(state['sources'], rows, seed)
        state['x'].copy_(state['sources'][rank])
        native.reference(state['sources'], None, state['fp32'], rows, 1)
        state['out'].fill_(float('nan'))
        before = base.gather_objects(workspace.protocol_snapshot(), host_group)
        if rank == phase % 2:
            torch.cuda._sleep(10000)
        mode = 'graph' if phase % 2 == 0 else 'eager'
        state['graph'].replay() if mode == 'graph' else state['operation']()
        torch.cuda.synchronize()
        check = validate(MX, state['op'], state['sources'], state['out'], None, state['fp32'], seed, host_group)
        equal = base.all_passed(base.bitwise_equal(state['out'], state['expected'][index]), host_group)
        after = base.gather_objects(workspace.protocol_snapshot(), host_group)
        delta = 10 if mode == 'graph' else 1
        state_ok = protocol_state_valid(before, after, rows*HIDDEN, args, delta)
        check.update(phase=phase, rows=rows, input_set=index, seed=seed, mode=mode,
            bitwise_equal_to_eager=equal, full_tensor_bitwise_equal_to_reference=equal,
            protocol_state_before=before, protocol_state_after=after,
            protocol_state_passed=state_ok, completed_calls=delta)
        check['passed'] &= equal and state_ok
        entry['phases'].append(check)
        checkpoint()
        if not check['passed']:
            raise RuntimeError(f'Group32 ring/shape/skew gate failed at phase{phase}')
    for state in prepared.values():
        state['graph'].reset()
    torch.cuda.synchronize()
    dist.barrier(group=host_group)
    entry.update(status='passed', passed=True)
    checkpoint()


def protocol_state_valid(before, after, numel, args, delta):
    ctas, tile = args.lamport_ctas, args.lamport_tile
    tiles = (numel+tile-1)//tile
    expected = [max(0, (tiles-cta+ctas-1)//ctas) for cta in range(ctas)]
    if len(before) != 2 or len(after) != 2 or before[0] != before[1] or after[0] != after[1]:
        return False
    for b, a in zip(before, after):
        bc, ac = b['call_counters'], a['call_counters']
        if not (len(bc) == len(ac) == ctas and len(set(bc)) == len(set(ac)) == 1
                and b['completed_epochs'] == bc and a['completed_epochs'] == ac
                and b['latest_buffers'] == [(n-1)%3 for n in bc]
                and a['latest_buffers'] == [(n-1)%3 for n in ac]
                and all(new == old+delta for old, new in zip(bc, ac))
                and a['previous_tile_counts'] == expected):
            return False
    return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--bridge', type=Path, default=Path(os.environ.get('FP8_BRIDGE_LIB', ROOT/'build/fp8_bridge/libfp8_native_bridge.so')))
    p.add_argument('--rows', default='1,4,16,64,1024,4096,16384')
    p.add_argument('--hidden', type=int, choices=[6144], default=6144)
    p.add_argument('--backends', default=','.join(BACKENDS))
    for name, default in [('iters',200),('warmup',20),('stress',100),('graph-k',10),
                          ('lamport-tile',2048),('lamport-ctas',152),('lamport-teams',7)]:
        p.add_argument('--'+name, type=int, default=default)
    a = p.parse_args()
    try:
        a.rows = [int(r) for r in a.rows.split(',')]
    except ValueError:
        p.error('rows must be integers')
    a.backends = a.backends.split(',')
    if not a.rows or len(set(a.rows)) != len(a.rows) or any(r < 1 or r > 16384 for r in a.rows):
        p.error('distinct rows in1..16384 required')
    if a.backends != BACKENDS:
        p.error('all four backends in the default order are required')
    if min(a.iters, a.warmup, a.stress) < 1 or a.graph_k != 10:
        p.error('positive counts and graph-k10 required')
    if not 256 <= a.lamport_tile <= 16384 or a.lamport_tile % 256 or not 1 <= a.lamport_ctas <= 152 or not 1 <= a.lamport_teams <= 7:
        p.error('invalid tile/CTA/team geometry')
    if not a.validate and not a.output:
        p.error('--output is required for GPU execution')
    a.tune = [0,0,0,0]
    return a


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, where):
    require(type(actual) is type(expected), where + ': type differs')
    if isinstance(expected, dict):
        require(set(actual)==set(expected), where + ': keys differ')
        for key in expected: same(actual[key],expected[key],where+'.'+str(key))
    elif isinstance(expected, list):
        require(len(actual)==len(expected),where+': length differs')
        for i,(a,b) in enumerate(zip(actual,expected)): same(a,b,where+f'[{i}]')
    else:
        require(actual==expected,where+': value differs')


def flags(item, names, where):
    for name in names: require(item.get(name) is True,where+'.'+name+' must pass')


def finite_number(n, where, positive=False):
    require(type(n) in (int,float) and math.isfinite(n) and (n>0 if positive else n>=0),where+': invalid number')


def expected_groups(numel, seed, group_size):
    if (group_size==32 and numel<=HIDDEN) or numel//group_size<=128:
        return list(range(numel//group_size))
    def hash32(v):
        v&=0xffffffff; v^=v>>16; v=v*0x7feb352d&0xffffffff
        v^=v>>15; v=v*0x846ca68b&0xffffffff
        return v^(v>>16)
    groups=numel//group_size
    chosen={0,1,groups//2,groups-2,groups-1}
    chosen.update(((hash32(seed*7919+104729*(t+1))*2654435761)%numel)//group_size for t in range(32))
    counter=0
    while len(chosen)<128:
        chosen.add(hash32(0xc0dec0de+counter)%groups);counter+=1
    return sorted(chosen)


def validate_check(c, backend, numel, seed, where, parity=False, metrics=True):
    flags(c,['passed','finite','full_tensor_rank_bitwise_equal'],where)
    if parity: flags(c,['bitwise_equal_to_eager'],where)
    if metrics:
        same(set(c['error_vs_original_fp32_sum']),{'max_abs','rmse','relative_l2'},where+'.metric keys')
        for n in c['error_vs_original_fp32_sum'].values(): finite_number(n,where+'.metric')
    n=c['numerical'];flags(n,['passed'],where+'.numerical')
    if backend in ('nccl','fp4'):
        same(n['contract'],'full_tensor_tp2_direct_fp32_sum_to_bf16' if backend=='nccl' else 'full_tensor_bitwise_original_fp4_codec',where+'.contract')
        return
    gs=32 if backend==MX else 128
    groups=expected_groups(numel,seed,gs)
    same(n['sampled_group_indices'],groups,where+'.sampled groups')
    same(n['sampled_numel'],len(groups)*gs,where+'.sampled numel')
    same(n['contract'],MX_CONTRACT if backend==MX else 'tp2_fp8_ue8m0_group128_oneshot_v1',where+'.contract')
    same(n['second_output_quantization'],False,where+'.second quant')
    flags(n,['reference_finite'],where+'.reference')
    payload=n['received_payload'];flags(payload,['passed'],where+'.wire')
    same(payload['source_order_per_receiving_rank'],[1,0],where+'.wire order')
    for field in ('quantized_byte_mismatches_per_rank','scale_code_byte_mismatches_per_rank',
                  'decoded_scale_bit_mismatches_per_rank' if backend==MX else 'scale_bit_mismatches_per_rank'):
        same(payload[field],[0,0],where+'.'+field)
    for field,maximum in [('negative_zero_byte_count_per_rank',len(groups)*gs),('scale_floor_count_per_rank',len(groups))]:
        same(payload[field],payload['expected_'+field],where+'.'+field)
        require(type(payload[field]) is list and len(payload[field])==2 and
                all(type(v) is int and 0<=v<=maximum for v in payload[field]),where+': invalid diagnostic counts')
    if backend==MX:
        same(n['comparison'],'exact_bf16_bits',where+'.comparison')
        same(n['sampled_output_bit_mismatches_per_rank'],[0,0],where+'.CPU output')
        same(n['independent_reference'],'NumPy frexp and finite E4M3 codebook RNE',where+'.oracle')
        digest=n['sampled_input_bf16_sha256'];require(type(digest) is str and len(digest)==64 and all(ch in '0123456789abcdef' for ch in digest),where+'.input digest')
        full=n['full_output_gpu_reference'];flags(full,['passed','untimed'],where+'.full output')
        same(full['mismatch_count_per_rank'],[0,0],where+'.full output mismatch')
        same(full['checked_numel'],numel,where+'.full output size')
    else:
        flags(n,['finite'],where+'.UE reference')
        same(n['comparison'],'exact_bf16_bits_in_float32_storage',where+'.comparison')
        same(n['mismatch_count_per_tp_rank'],[0,0],where+'.CPU output')
        finite_number(n['max_abs_codec_error'],where+'.codec error');require(n['max_abs_codec_error']==0,where+'.codec error')
        same(n['failure_examples'],[],where+'.failures')
        same(n['overflow_diagnostics'],dict(normalized_nonfinite_count=0,dequant_nonfinite_count_per_source=[0,0],
            fp32_reduction_nonfinite_count=0,fp32_reduction_nan_count=0,bf16_output_nonfinite_count=0,bf16_conversion_overflow_count=0),where+'.overflow')
        flags(payload,['decoded_scale_finite_positive'],where+'.scale finite')


def validate_report(path, args):
    def pairs(items):
        value={}
        for k,v in items:
            require(k not in value,'duplicate JSON key: '+k);value[k]=v
        return value
    def bad_constant(value): raise ValueError('nonfinite JSON constant: '+value)
    report=json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=bad_constant)
    for key,expected in dict(study='fp8_group32_compare_v1',status='passed',correctness_passed=True,
        rows=args.rows,backends=args.backends,hidden=6144,tp_size=2,graph_k=10,iters=args.iters,
        warmup=args.warmup,stress=args.stress,fp4_variant='L2',fp4_algo=1,fp4_tune=[0,0,0,0],
        lamport_config=dict(tile_elems=args.lamport_tile,ctas=args.lamport_ctas,teams=args.lamport_teams),
        group_sizes={UE:128,MX:32},mxfp8_contract=MX_CONTRACT,model_accuracy_qualified=False,
        upstream_commit='2878bf50e20241d55500c2df76b87f0a81ef1c0f').items():
        same(report[key],expected,'report.'+key)
    require(set(report['source_sha256'])==set(SOURCE_NAMES),'source map incomplete')
    for name in SOURCE_NAMES:
        same(report['source_sha256'][name],sha(HERE/name),'measured/local source '+name)
    for name,digest in FROZEN.items(): same(report['source_sha256'][name],digest,'frozen source '+name)
    flags(report['mxfp8_recipe'],['communication_kernel','performance_measured'],'measured MXFP8 recipe')
    same(report['mxfp8_recipe']['group_size'],32,'MXFP8 recipe group size')
    self_test=report['cpu_mxfp8_self_test']
    flags(self_test,['passed','negative_zero_preserved'],'CPU reference self-test')
    for k,v in dict(finite_bf16_patterns=65280,edge_groups=9,zero_scale_code=0).items():
        same(self_test[k],v,'CPU reference self-test.'+k)
    runtime=report['runtime'];same(len(runtime),2,'runtime ranks')
    for rank,r in enumerate(runtime):
        for k,v in dict(rank=rank,device=rank,name='NVIDIA GB200',torch='2.13.0+cu130',cuda='13.0',cutlass_dsl='4.6.2').items():same(r[k],v,'runtime.'+k)
        require(type(r['nccl_loaded_version']) is int and r['nccl_loaded_version']>=23100,'NCCL runtime version')
        maps=r['loaded_nccl_maps'];require(type(maps) is list and maps,'missing loaded NCCL maps')
        require(all(len(line.split())==6 and line.split()[-1]=='/nccl/lib/libnccl.so.2' for line in maps),'mixed/deleted NCCL mappings')
    same(runtime[0]['nccl_loaded_version'],runtime[1]['nccl_loaded_version'],'rank NCCL version')
    for backend,gs in [(UE,128),(MX,32)]:
        w=report['workspaces'][backend];wire=((args.lamport_tile+args.lamport_tile//gs+15)//16)*16
        for k,v in dict(backend=backend,group_size=gs,tile_elems=args.lamport_tile,ctas=args.lamport_ctas,
            teams_per_cta=args.lamport_teams,threads_per_cta=64*args.lamport_teams+32,receive_buffers=3,
            max_numel=max(args.rows)*HIDDEN,wire_bytes_per_tile=wire,poll_batch_groups=4,reset_batch_tiles=4,
            payload_sentinel=255,end_of_call_handshake=False,polls_padded_groups=True,
            protocol='payload_ff_lamport_triple_buffer',requires_serial_use=True,cooperative_launch=True,
            scale_floor_exponent=-127 if gs==32 else -120,zero_scale_code=0 if gs==32 else 7,
            numeric_contract=MX_CONTRACT if gs==32 else 'tp2_fp8_ue8m0_group128_oneshot_v1').items():same(w[k],v,backend+'.workspace.'+k)
        module='cute_fp8/mxfp8_lamport_batched.py' if gs==32 else 'cute_fp8/ue8m0_lamport_batched.py'
        same(w['source_sha256'],report['source_sha256'][module],backend+'.kernel hash')
    same(len(report['results']),len(args.rows)*4,'cell count');seen=set()
    for c in report['results']:
        key=(c['rows'],c['backend']);require(key not in seen and key[0] in args.rows and key[1] in args.backends,'missing/duplicate cell');seen.add(key)
        rows,backend=key;where=f'{backend}/{rows}'
        flags(c,['correctness_passed'],where);same(c['status'],'passed',where+'.status');require('invalid_timing' not in c,where+': invalid timing present')
        same(c['numel'],rows*HIDDEN,where+'.numel');same(c['bf16_input_bytes'],rows*HIDDEN*2,where+'.bytes')
        checks=c['checks'];same(len(checks),7,where+'.checks');observed=set()
        for check in checks:
            pair=(check['mode'],check['input_set']);require(pair not in observed,'duplicate check');observed.add(pair)
            index=check['input_set'];require(type(index) is int and index in (0,1,2),'invalid input set')
            seed=1000+17*rows+7777*index
            if check['mode']=='eager':same(check['seed'],seed,where+'.seed')
            validate_check(check,backend,rows*HIDDEN,seed,where,parity=check['mode']=='changed_input_graph')
        same(observed,{('eager',i) for i in range(3)}|{('changed_input_graph',i) for i in range(3)}|{('after_timing',0)},where+'.check coverage')
        s=c['stress'];flags(s,['passed','alternating_eager_graph','output_poisoned_between_calls','bounded_smoke_not_full_upstream_protocol_suite'],where+'.stress')
        for k,v in dict(checked_calls=args.stress,graph_k=10,rotating_input_sets=3,skew_sleep_cycles=10000).items():same(s[k],v,where+'.stress.'+k)
        if backend==UE:
            e=c['baseline_equivalence'];flags(e,['passed','separate_workspace','separate_output','untimed'],where+'.baseline');same(e['status'],'passed',where+'.baseline status')
            same(e['baseline_backend'],'fp8-ue8m0-sm',where+'.baseline backend');same(e['graph_k'],10,where+'.baseline graph')
            same(len(e['checks']),6,where+'.baseline check count');seen_eq=set()
            for x in e['checks']:
                k=(x['mode'],x['input_set']);require(k not in seen_eq,'duplicate baseline comparison');seen_eq.add(k)
                flags(x,['passed','baseline_finite','full_tensor_bitwise_equal'],where+'.baseline check')
                same(x['seed'],1000+17*rows+7777*x['input_set'],where+'.baseline seed')
            same(seen_eq,{(m,i) for m in ('eager','graph') for i in range(3)},where+'.baseline coverage')
        t=c['timing'];samples=t['rank_samples_us'];same(len(samples),2,where+'.timing ranks')
        for rank_samples in samples:
            same(len(rank_samples),args.iters,where+'.sample count')
            for value in rank_samples:finite_number(value,where+'.sample',True)
        maxima=[max(x) for x in zip(*samples)];ordered=sorted(maxima)
        same(t['max_rank_samples_us'],maxima,where+'.paired maxima')
        for k,i in [('median_us',args.iters//2),('p10_us',args.iters//10),('p90_us',9*args.iters//10)]:same(t[k],ordered[i],where+'.'+k)
    expected_protocol={(b,n) for b in (UE,MX) for n in ('adversarial_codec','alternating_shape_workspace')}|{(MX,'mxfp8_scale_edges')}
    same(len(report['protocol_checks']),5,'protocol count');seen_protocol=set()
    for p in report['protocol_checks']:
        key=(p['backend'],p['name']);require(key in expected_protocol and key not in seen_protocol,'duplicate/unknown protocol');seen_protocol.add(key)
        backend,name=key;flags(p,['passed'],str(key));same(p['status'],'passed',str(key))
        if name=='mxfp8_scale_edges':
            for k,v in dict(numel=288,group_size=32,cases=9,checked_all_groups=True,graph_k=10,input_bf16_bits_preserved=True,output_poisoned_between_calls=True).items():same(p[k],v,'edges.'+k)
            same([c['mode'] for c in p['checks']],['eager','graph'],'edge modes')
            for c in p['checks']:
                validate_check(c,MX,288,0,'edges',parity=c['mode']=='graph',metrics=False)
                payload=c['numerical']['received_payload'];require(all(n>0 for n in payload['scale_floor_count_per_rank']) and all(n>0 for n in payload['negative_zero_byte_count_per_rank']),'edge floor/signed-zero coverage')
        elif name=='adversarial_codec':
            same([c['name'] for c in p['cases']],['positive_zero','mixed_signed_zero','cancellation','varying_ranges_outliers'],'adversarial cases')
            flags(p,['sampled_gate_is_exhaustive','output_poisoned_between_calls'],'adversarial')
            for case in p['cases']:
                flags(case,['passed'],'adversarial case');same([c['mode'] for c in case['checks']],['eager','graph'],'adversarial modes')
                for c in case['checks']:validate_check(c,backend,HIDDEN,1017,'adversarial',parity=c['mode']=='graph')
        else:
            choices=sorted(set((min(args.rows),args.rows[len(args.rows)//2],max(args.rows))))
            same(p['shapes'],choices,'reuse shapes');flags(p,['same_workspace','triple_buffer_wrap','output_poisoned_between_calls','bounded_smoke_not_full_upstream_protocol_suite'],'reuse')
            for k,v in dict(graph_k=10,receive_buffers=3,rotating_input_sets=5,skew_sleep_cycles=10000).items():same(p[k],v,'reuse.'+k)
            same(len(p['preparation_checks']),len(choices)*5,'reuse preparations');seen_p=set()
            for c in p['preparation_checks']:
                pair=(c['rows'],c['input_set']);require(pair not in seen_p and pair[0] in choices and type(pair[1]) is int and pair[1] in range(5),'reuse preparation coverage');seen_p.add(pair)
                validate_check(c,backend,c['rows']*HIDDEN,1000+17*c['rows']+7777*c['input_set'],'reuse preparation')
                if backend==UE:flags(c,['baseline_finite','full_tensor_bitwise_equal_to_baseline'],'reuse baseline')
            small,middle,large=choices[0],choices[len(choices)//2],choices[-1]
            sequence=[large,small,middle,small,large,middle,small,large,middle,large,small,large]
            same(len(p['phases']),12,'reuse phase count')
            for phase,(c,rows) in enumerate(zip(p['phases'],sequence)):
                mode='graph' if phase%2==0 else 'eager';delta=10 if mode=='graph' else 1
                for k,v in dict(phase=phase,rows=rows,input_set=phase%5,seed=1000+17*rows+7777*(phase%5),mode=mode,completed_calls=delta).items():same(c[k],v,'reuse phase.'+k)
                validate_check(c,backend,rows*HIDDEN,c['seed'],'reuse phase',parity=True)
                flags(c,['protocol_state_passed','full_tensor_bitwise_equal_to_reference' if backend==MX else 'full_tensor_bitwise_equal_to_baseline'],'reuse phase')
                require(protocol_state_valid(c['protocol_state_before'],c['protocol_state_after'],rows*HIDDEN,args,delta),'invalid protocol state')
                if phase:same(c['protocol_state_before'],p['phases'][phase-1]['protocol_state_after'],'protocol continuity')
    print(json.dumps(dict(status='passed',cells=len(seen),protocol_checks=len(seen_protocol),
        all_recorded_gates_passed=True,raw_timing_recomputed=True,report_sha256=sha(path))))


def main():
    global torch, dist, base, legacy, oracle, original_validate, original_make_operation
    args = parse_args()
    if args.validate:
        validate_report(args.validate, args)
        return
    if os.environ.get('FP8_FUSED_DUMP_PTX') == '1':
        folder = args.output.parent/f"ptx-rank{os.environ['RANK']}"
        folder.mkdir(parents=True, exist_ok=True)
        os.environ.update(CUTE_DSL_KEEP='ptx,cubin', CUTE_DSL_DUMP_DIR=str(folder))
    import torch
    import torch.distributed as dist
    import fp8_compare as base
    import fp8_fused_compare as legacy
    import fp8_format_accuracy as oracle
    base.torch, base.dist = torch, dist
    legacy.torch, legacy.dist = torch, dist
    oracle.torch = torch
    original_validate, original_make_operation = legacy.validate, legacy.make_operation
    legacy.validate, legacy.make_operation = validate, make_operation
    hashes = {p: sha(HERE/p) for p in SOURCE_NAMES}
    for name, expected in FROZEN.items():
        if hashes[name] != expected:
            raise RuntimeError('Frozen source differs: '+name)
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=8))
    if dist.get_world_size() != 2:
        raise RuntimeError('TP2 required')
    rank = dist.get_rank()
    host = dist.new_group(backend='gloo', timeout=datetime.timedelta(minutes=8))
    report = dict(study='fp8_group32_compare_v1', status='running', correctness_passed=False,
        rows=args.rows, backends=args.backends, hidden=HIDDEN, tp_size=2, graph_k=10,
        iters=args.iters, warmup=args.warmup, stress=args.stress, fp4_variant='L2', fp4_algo=1, fp4_tune=args.tune,
        lamport_config=dict(tile_elems=args.lamport_tile, ctas=args.lamport_ctas, teams=args.lamport_teams),
        group_sizes={UE:128,MX:32}, mxfp8_contract=MX_CONTRACT,
        mxfp8_recipe=dict(oracle.MXFP8_METADATA, communication_kernel=True, performance_measured=True),
        source_sha256=hashes, bridge_binary_sha256=sha(args.bridge),
        upstream_commit='2878bf50e20241d55500c2df76b87f0a81ef1c0f',
        timing_contract='BF16 input/output, all quantization and synchronization included; graph10; out-of-place NCCL; excluded rank alignment; max rank per sample then upper median',
        numerical_scope='Actual group32 collective: full GPU numerical reference plus independent sampled NumPy output/received-wire checks. All reference work is untimed. Group128 retains its original independent oracle and frozen-baseline gates.',
        model_accuracy_qualified=False, results=[], protocol_checks=[],
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    started = time.time()
    def checkpoint():
        if rank == 0:
            report['elapsed_s'] = time.time()-started
            base.write_json(args.output, report)
    checkpoint()
    try:
        native = base.Native(args.bridge, rank, 2, device, max(args.rows), 'L2', host)
        report['runtime'] = base.gather_objects(dict(rank=rank,device=device,name=torch.cuda.get_device_name(),
            torch=torch.__version__,cuda=torch.version.cuda,cutlass_dsl=importlib.metadata.version('nvidia-cutlass-dsl'),
            nccl_loaded_version=native.version,nccl_torch_query=torch.cuda.nccl.version(),
            loaded_nccl_maps=[l for l in Path('/proc/self/maps').read_text().splitlines() if 'libnccl' in l]),host)
        from cute_fp8.ue8m0_lamport_batched import BatchedLamportUE8M0FP8AllReduceWorkspace
        from cute_fp8.mxfp8_lamport_batched import MXFP8LamportBatchedAllReduceWorkspace
        from cute_fp8.ue8m0 import UE8M0FP8AllReduceWorkspace
        params=dict(max_numel=max(args.rows)*HIDDEN,device=torch.device('cuda',device))
        geometry=dict(tile_elems=args.lamport_tile,ctas=args.lamport_ctas,teams=args.lamport_teams)
        workspaces={UE:BatchedLamportUE8M0FP8AllReduceWorkspace(**params,group_size=128,**geometry),
                    MX:MXFP8LamportBatchedAllReduceWorkspace(**params,group_size=32,**geometry)}
        baseline=UE8M0FP8AllReduceWorkspace(**params,group_size=128,tile_elems=16384,ctas=152)
        report['workspaces']={k:w.metadata for k,w in workspaces.items()}
        report['group128_baseline_workspace']=baseline.metadata
        report['cpu_mxfp8_self_test']=oracle.self_test()
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Qualify the true E8M0 floor and finite tail before any timing.
            edge=dict(name='mxfp8_scale_edges',backend=MX,status='running',passed=False)
            report['protocol_checks'].append(edge)
            scale_edges(workspaces[MX],native,args,host,checkpoint,edge)
            for rows in args.rows:
                for backend in args.backends:
                    cell=dict(rows=rows,numel=rows*HIDDEN,bf16_input_bytes=rows*HIDDEN*2,
                              backend=backend,status='running',correctness_passed=False)
                    report['results'].append(cell)
                    if rank==0: print(f'START {backend} rows={rows}',flush=True)
                    legacy.run_cell(rows,backend,native,workspaces.get(backend),args,host,checkpoint,cell,
                                    baseline_workspace=baseline if backend==UE else None)
                    if rank==0: print(f"PASS {backend} rows={rows} median_us={cell['timing']['median_us']:.3f}",flush=True)
            for backend in (UE,MX):
                for name in ('adversarial_codec','alternating_shape_workspace'):
                    entry=dict(name=name,backend=backend,status='running',passed=False)
                    report['protocol_checks'].append(entry)
                    if name=='adversarial_codec':
                        legacy.adversarial_check(backend,workspaces[backend],native,args,host,checkpoint,entry)
                    elif backend==UE:
                        legacy.alternating_shape_check(backend,workspaces[backend],native,args,host,checkpoint,entry,baseline_workspace=baseline)
                    else:
                        alternating_mxfp8(workspaces[backend],native,args,host,checkpoint,entry)
        native.call('destroy',native.handle)
        report.update(status='passed',correctness_passed=True)
        checkpoint()
        dist.destroy_process_group()
    except BaseException as error:
        for item in report['results']+report['protocol_checks']:
            if item['status']=='running': item['status']='failed'
        report.update(status='failed',correctness_passed=False,error=repr(error),traceback=traceback.format_exc())
        checkpoint()
        raise


if __name__ == '__main__':
    main()
