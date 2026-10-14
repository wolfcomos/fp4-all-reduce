"""Experimental TP2 MXFP8 E4M3/group32 Lamport allreduce with batched polling.

Each lane preloads four groups of peer payload/scale words, jointly re-polls
only incomplete entries, then decodes with the frozen codec. Reset issuers
batch up to four copies from their immutable sentinel pattern before each
full wait. This variant polls every transmitted q/scale group, including
zero-filled input-tail groups, and omits the conservative peer completion gate.

Each warp partitions into four8-lane groups; each lane encodes four adjacent
values, giving one independent scale per32 values. The scale recipe is
2**max(-127,ceil(log2(amax/448))), including byte0 for zero groups. This matches
the existing MXFP8 round-up numerical reference, not the group128 scale floor.
Each team TMA-loads BF16, pushes tile k,
and consumes tile k-1 directly from peer-written global memory. Every received
payload byte is checked against FF: finite E4M3 excludes FF, while 80 remains
legal negative zero. E8M0 scale codes are 0..247; code0 means 2**-127. Two own-codec slots overlap
egress with decode; source reuse waits for TMA reads, and exit waits for writes.

Calls are strictly stream-serialized and use a fixed cooperative CTA grid.
Three receive buffers rotate by a device call counter. A reset warp re-arms
the previous call's local tile extent while workers use the current buffer.
All sends and resets finish before the kernel exits; a local counter advances
once per CTA. Full payload polling protects reset from late old writes. A peer
cannot send call c+3 into call c's buffer until it has consumed some payload
from local call c+2. Strict stream serialization means the local c+1 reset has
already finished by then. This proof requires matching NONEMPTY calls, fixed
workspace geometry, complete q/scale polling (including tails), and no PDL.
Fixed alignment padding is never interpreted as payload and need not be polled.
No per-tile READY/ACK flags, extra reset kernels, or host work occur in a call.
"""

import hashlib
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm

from .kernels import _cute
from .ue8m0 import _tma_load, _tma_store

NUMERIC_CONTRACT = "tp2_mxfp8_e4m3_group32_e8m0_round_up_reference_v1"

BASELINE_SOURCE_SHA256 = "c02631c399c1df8911c55b2ebd15ee7c2a4cbba600b3be2decdbdf4fe36081cb"
MXFP8_LAMPORT_BATCHED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dsl_user_op
def _scale_code(amax: Float32, *, loc=None, ip=None):
    """TensorRT round-up E8M0 recipe, with minimum positive scale 2**-127.

    Normal amax=2**E*m uses exponent E-8 plus one iff m>1.75.
    Zero and FP32-subnormal amax need only the -127 floor because their
    amax/448 is below it. The caller supplies finite nonnegative BF16 amax.
    """
    code = llvm.inline_asm(
        T.i32(), [amax.ir_value()],
        "{ .reg .b32 bits, field, mant, bump; .reg .s32 exponent; .reg .pred p; "
        "mov.b32 bits, $1; and.b32 bits, bits, 0x7fffffff; "
        "shr.u32 field, bits, 23; and.b32 mant, bits, 0x007fffff; "
        "add.s32 exponent, field, -135; setp.gt.u32 p, mant, 0x00600000; "
        "selp.b32 bump, 1, 0, p; add.s32 exponent, exponent, bump; "
        "max.s32 exponent, exponent, -127; add.s32 $0, exponent, 127; }",
        "=r,f", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip,
    )
    return Int32(code)


@dsl_user_op
def _scale_float(code: Int32, *, loc=None, ip=None):
    """Decode generated E8M0 codes0..247, including subnormal FP32 scale0.

    Shared Uint8 loads can sign-extend, so mask before testing or shifting.
    E8M0 code0 is 2**-127 (FP32 bits00400000), never floating-point zero.
    """
    value = llvm.inline_asm(
        T.f32(), [code.ir_value()],
        "{ .reg .b32 byte, bits; .reg .pred zero; and.b32 byte, $1, 0xff; "
        "shl.b32 bits, byte, 23; setp.eq.u32 zero, byte, 0; "
        "selp.b32 bits, 0x00400000, bits, zero; mov.b32 $0, bits; }",
        "=f,r", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip,
    )
    return Float32(value)


@dsl_user_op
def _inverse_scale_float(code: Int32, *, loc=None, ip=None):
    """Exact inverse for generated codes0..247; code0 produces 2**127."""
    value = llvm.inline_asm(
        T.f32(), [code.ir_value()],
        "{ .reg .b32 byte, bits; and.b32 byte, $1, 0xff; "
        "sub.u32 bits, 254, byte; shl.b32 bits, bits, 23; mov.b32 $0, bits; }",
        "=f,r", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip,
    )
    return Float32(value)


@cute.jit
def _subgroup_amax(value: Float32):
    # XOR1/2/4 never crosses an aligned8-lane subgroup. Every lane holds4
    # adjacent values, so each independent maximum covers exactly32 values.
    for k in cutlass.range_constexpr(3):
        value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1 << k))
    return value


@dsl_user_op
def _team_barrier(bar: Int32, *, loc=None, ip=None):
    llvm.inline_asm(T.i32(), [bar.ir_value()],
                    "{ bar.sync $1, 64; mov.u32 $0, 0; }", "=r,r,~{memory}",
                    has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _bulk_commit(*, loc=None, ip=None):
    llvm.inline_asm(T.i32(), [], "{ cp.async.bulk.commit_group; mov.u32 $0, 0; }",
                    "=r,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _bulk_wait_read1(*, loc=None, ip=None):
    llvm.inline_asm(T.i32(), [], "{ cp.async.bulk.wait_group.read 1; mov.u32 $0, 0; }",
                    "=r,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _bulk_wait_all(*, loc=None, ip=None):
    llvm.inline_asm(T.i32(), [], "{ cp.async.bulk.wait_group 0; mov.u32 $0, 0; }",
                    "=r,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _wait_at_least(ptr, epoch: Int64, *, loc=None, ip=None):
    llvm.inline_asm(T.i64(), [Int64(ptr.toint()).ir_value(), epoch.ir_value()],
                    "{ .reg .pred p; .reg .u64 observed; WAIT: ld.acquire.sys.global.u64 observed, [$1]; "
                    "setp.lt.u64 p, observed, $2; @p bra WAIT; mov.u64 $0, observed; }", "=l,l,l,~{memory}",
                    has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _load_shared64(ptr, *, loc=None, ip=None):
    value = llvm.inline_asm(T.i64(), [Int32(ptr.toint()).ir_value()],
                            "ld.shared.b64 $0, [$1];", "=l,r,~{memory}",
                            has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)
    return Int64(value)


@dsl_user_op
def _load_shared32(ptr, *, loc=None, ip=None):
    value = llvm.inline_asm(T.i32(), [Int32(ptr.toint()).ir_value()],
                            "ld.shared.b32 $0, [$1];", "=r,r,~{memory}",
                            has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)
    return Int32(value)


@dsl_user_op
def _store_shared32(ptr, value: Int32, *, loc=None, ip=None):
    llvm.inline_asm(T.i32(), [Int32(ptr.toint()).ir_value(), value.ir_value()],
                    "{ st.shared.b32 [$1], $2; mov.u32 $0, 0; }", "=r,r,r,~{memory}",
                    has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _store_global64(ptr, value: Int64, *, loc=None, ip=None):
    llvm.inline_asm(T.i32(), [Int64(ptr.toint()).ir_value(), value.ir_value()],
                    "{ st.global.b64 [$1], $2; mov.u32 $0, 0; }", "=r,l,l,~{memory}",
                    has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _poll_payload4(ptr, *, loc=None, ip=None):
    # has_zero_byte(~word) detects ANY FF byte, not merely word == FFFFFFFF.
    value = llvm.inline_asm(T.i32(), [Int64(ptr.toint()).ir_value()],
        "{ .reg .b32 inv, minus, bad; .reg .pred p; WAIT: "
        "ld.volatile.global.u32 $0, [$1]; not.b32 inv, $0; "
        "sub.u32 minus, inv, 0x01010101; and.b32 bad, minus, $0; "
        "and.b32 bad, bad, 0x80808080; setp.ne.u32 p, bad, 0; @p bra WAIT; }",
        "=r,l,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)
    return Int32(value)


@dsl_user_op
def _poll_scale(ptr, *, loc=None, ip=None):
    value = llvm.inline_asm(T.i32(), [Int64(ptr.toint()).ir_value()],
        "{ .reg .pred p; WAIT: ld.volatile.global.u8 $0, [$1]; "
        "setp.eq.u32 p, $0, 255; @p bra WAIT; }", "=r,l,~{memory}",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)
    return Int32(value)


@dsl_user_op
def _load_payload4(ptr, *, loc=None, ip=None):
    """Issue a packed volatile load without immediately waiting on its value."""
    value = llvm.inline_asm(T.i32(), [Int64(ptr.toint()).ir_value()],
        "ld.volatile.global.u32 $0, [$1];", "=r,l,~{memory}",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)
    return Int32(value)


@dsl_user_op
def _load_scale(ptr, *, loc=None, ip=None):
    value = llvm.inline_asm(T.i32(), [Int64(ptr.toint()).ir_value()],
        "ld.volatile.global.u8 $0, [$1];", "=r,l,~{memory}",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)
    return Int32(value)


@dsl_user_op
def _payload_bad(word: Int32, *, loc=None, ip=None):
    value = llvm.inline_asm(T.i32(), [word.ir_value()],
        "{ .reg .b32 inv, minus; not.b32 inv, $1; "
        "sub.u32 minus, inv, 0x01010101; and.b32 minus, minus, $1; "
        "and.b32 $0, minus, 0x80808080; }", "=r,r",
        has_side_effects=False, asm_dialect=0, loc=loc, ip=ip)
    return Int32(value)


@dsl_user_op
def _pair_bad(word: Int32, scale: Int32, *, loc=None, ip=None):
    value = llvm.inline_asm(T.i32(), [word.ir_value(), scale.ir_value()],
        "{ .reg .b32 inv, minus, bad, flag; .reg .pred p; not.b32 inv, $1; "
        "sub.u32 minus, inv, 0x01010101; and.b32 bad, minus, $1; "
        "and.b32 bad, bad, 0x80808080; setp.eq.u32 p, $2, 255; "
        "selp.b32 flag, 1, 0, p; or.b32 $0, bad, flag; }", "=r,r,r",
        has_side_effects=False, asm_dialect=0, loc=loc, ip=ip)
    return Int32(value)


@dsl_user_op
def _bf16x4_amax(bits: Int64, *, loc=None, ip=None):
    value = llvm.inline_asm(T.f32(), [bits.ir_value()],
        "{ .reg .b32 a, b, v0, v1, v2, v3; mov.b64 {a,b}, $1; "
        "and.b32 v0, a, 0x7fff; shr.u32 v1, a, 16; and.b32 v1, v1, 0x7fff; "
        "and.b32 v2, b, 0x7fff; shr.u32 v3, b, 16; and.b32 v3, v3, 0x7fff; "
        "max.u32 v0, v0, v1; max.u32 v2, v2, v3; max.u32 v0, v0, v2; "
        "shl.b32 v0, v0, 16; mov.b32 $0, v0; }", "=f,l",
        has_side_effects=False, asm_dialect=0, loc=loc, ip=ip)
    return Float32(value)


@dsl_user_op
def _encode4(bits: Int64, inverse: Float32, *, loc=None, ip=None):
    body = "{ .reg .b32 a,b; .reg .b16 h<4>,q<2>; .reg .f32 f<4>; "
    body += "mov.b64 {a,b}, $1; mov.b32 {h0,h1}, a; mov.b32 {h2,h3}, b; "
    for j in range(4):
        body += (f"cvt.f32.bf16 f{j}, h{j}; mul.rn.f32 f{j}, f{j}, $2; "
                 f"min.f32 f{j}, f{j}, 0f43e00000; max.f32 f{j}, f{j}, 0fc3e00000; ")
    body += ("cvt.rn.satfinite.e4m3x2.f32 q0, f1, f0; "
             "cvt.rn.satfinite.e4m3x2.f32 q1, f3, f2; mov.b32 $0, {q0,q1}; }")
    value = llvm.inline_asm(T.i32(), [bits.ir_value(), inverse.ir_value()], body,
                           "=r,l,f", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip)
    return Int32(value)


@dsl_user_op
def _decode_sum4(q0: Int32, q1: Int32, s0: Float32, s1: Float32, *, loc=None, ip=None):
    body = ("{ .reg .b16 a<2>,b<2>,h<8>; .reg .b32 w<4>,o<2>; "
            ".reg .f32 f<8>,p,sum<4>; mov.b32 {a0,a1}, $1; mov.b32 {b0,b1}, $2; "
            "cvt.rn.f16x2.e4m3x2 w0, a0; cvt.rn.f16x2.e4m3x2 w1, a1; "
            "cvt.rn.f16x2.e4m3x2 w2, b0; cvt.rn.f16x2.e4m3x2 w3, b1; ")
    for j in range(4):
        body += f"mov.b32 {{h{2*j},h{2*j+1}}}, w{j}; "
    for j in range(8):
        body += f"cvt.f32.f16 f{j}, h{j}; "
    for j in range(4):
        body += (f"mul.rn.f32 p, f{j}, $3; add.rn.f32 sum{j}, p, 0f00000000; "
                 f"mul.rn.f32 p, f{j+4}, $4; add.rn.f32 sum{j}, p, sum{j}; ")
    body += ("cvt.rn.bf16x2.f32 o0, sum1, sum0; cvt.rn.bf16x2.f32 o1, sum3, sum2; "
             "mov.b64 $0, {o0,o1}; }")
    value = llvm.inline_asm(T.i64(), [q0.ir_value(), q1.ir_value(), s0.ir_value(), s1.ir_value()],
                           body, "=l,r,r,f,f", has_side_effects=False,
                           asm_dialect=0, loc=loc, ip=ip)
    return Int64(value)


@cute.kernel
def _mxfp8_lamport_batched_kernel(x: cute.Tensor, out: cute.Tensor, wire: tuple, state: tuple,
                          n: cutlass.Constexpr, tile_elems: cutlass.Constexpr,
                          ctas: cutlass.Constexpr, teams: cutlass.Constexpr,
                          buffer_bytes: cutlass.Constexpr, rank: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane, warp = tid % 32, tid // 32
    groups = tile_elems // 32
    wire_bytes = ((tile_elems + groups + 15) // 16) * 16
    team_bytes = ((2 * tile_elems + 2 * wire_bytes + 127) // 128) * 128
    tiles = (n + tile_elems - 1) // tile_elems
    if cutlass.const_expr(rank == 0):
        own_wire, peer_wire = wire[0], wire[1]
        own_state, peer_state = state[0], state[1]
    else:
        own_wire, peer_wire = wire[1], wire[0]
        own_state, peer_state = state[1], state[0]
    call = own_state[block]
    previous_count = own_state[ctas + block]
    local_count = (tiles + ctas - 1 - block) // ctas
    buffer_offset = (call % Int64(3)) * buffer_bytes
    previous_offset = ((call + Int64(2)) % Int64(3)) * buffer_bytes
    smem = utils.SmemAllocator()
    storage = smem.allocate_array(Uint8, teams * team_bytes, byte_alignment=128)
    sentinel = smem.allocate_array(Uint8, wire_bytes, byte_alignment=128)
    barriers = smem.allocate_array(Int64, teams, byte_alignment=8)
    if warp < 2 * teams:
        if warp % 2 == 0 and lane == 0:
            cute.arch.mbarrier_init(barriers + warp // 2, 1)
    else:
        for i in cutlass.range_constexpr((wire_bytes + 127) // 128):
            offset = i * 128 + lane * 4
            if offset < wire_bytes:
                _store_shared32(sentinel + offset, Int32(-1))
        cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    if warp == 2 * teams:
        # Batch up to four copies per issuer from the immutable shared pattern.
        # Ownership remains block + ctas*i. A partial final batch is committed
        # and fully drained before the final CTA join and local counter update.
        i = Int32(lane)
        while Int64(i) < previous_count:
            for reset_batch in cutlass.range_constexpr(4):
                if Int64(i) < previous_count:
                    tile = block + ctas * i
                    _tma_store(own_wire.iterator + previous_offset + Int64(tile) * wire_bytes,
                               sentinel, Int32(wire_bytes))
                i += 32
            _bulk_commit()
            _bulk_wait_all()
        _bulk_wait_all()
    else:
        team, half = warp // 2, warp % 2
        bar = Int32(team + 1)
        team_ptr = storage + team * team_bytes
        input_ptr = team_ptr
        own_ptr = team_ptr + 2 * tile_elems
        team_count = (local_count + teams - 1 - team) // teams
        if half == 0 and lane == 0:
            if team_count > 0:
                tile = block + ctas * team
                nbytes = Int32(tile_elems * 2)
                if n - tile * tile_elems < tile_elems:
                    nbytes = Int32((n - tile * tile_elems) * 2)
                cute.arch.mbarrier_arrive_and_expect_tx(barriers + team, nbytes)
                _tma_load(input_ptr, x.iterator + tile * tile_elems, nbytes, barriers + team)
        k = Int32(0)
        while k < team_count + 1:
            if k < team_count:
                # Protect source reuse before EITHER warp can overwrite it.
                if k >= 2:
                    if half == 0 and lane == 0:
                        _bulk_wait_read1()
                _team_barrier(bar)
                cute.arch.mbarrier_wait(barriers + team, k % 2)
                slot = own_ptr + (k % 2) * wire_bytes
                scales = cute.make_tensor(slot + tile_elems, cute.make_layout((groups,)))
                tile = block + ctas * (team + teams * k)
                for j in cutlass.range_constexpr(groups // 8):
                    group = half * 4 + lane // 8 + 8 * j
                    offset = group * 32 + (lane % 8) * 4
                    bits = Int64(0)
                    if tile * tile_elems + group * 32 < n:
                        bits = _load_shared64(input_ptr + offset * 2)
                    code = _scale_code(_subgroup_amax(_bf16x4_amax(bits)))
                    if lane % 8 == 0:
                        scales[group] = code.to(Uint8)
                    _store_shared32(slot + offset, _encode4(bits, _inverse_scale_float(code)))
                # Padding is not a sentinel and is never used for readiness.
                if cutlass.const_expr(wire_bytes > tile_elems + groups):
                    slot_bytes = cute.make_tensor(slot, cute.make_layout((wire_bytes,)))
                    if half == 0 and lane < wire_bytes - tile_elems - groups:
                        slot_bytes[tile_elems + groups + lane] = Uint8(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                _team_barrier(bar)
                if half == 0 and lane == 0:
                    if k + 1 < team_count:
                        next_tile = block + ctas * (team + teams * (k + 1))
                        nbytes = Int32(tile_elems * 2)
                        if n - next_tile * tile_elems < tile_elems:
                            nbytes = Int32((n - next_tile * tile_elems) * 2)
                        cute.arch.mbarrier_arrive_and_expect_tx(barriers + team, nbytes)
                        _tma_load(input_ptr, x.iterator + next_tile * tile_elems, nbytes, barriers + team)
                    _tma_store(peer_wire.iterator + buffer_offset + Int64(tile) * wire_bytes,
                               slot, Int32(wire_bytes))
                    _bulk_commit()
            if k > 0:
                kd = k - 1
                tile = block + ctas * (team + teams * kd)
                slot = own_ptr + (kd % 2) * wire_bytes
                scales = cute.make_tensor(slot + tile_elems, cute.make_layout((groups,)))
                remote = own_wire.iterator + buffer_offset + Int64(tile) * wire_bytes
                # Each batch exposes four independent payload/scale requests
                # before consuming any returned value. Register tensors retain
                # completed words while the joint loop re-polls missing bytes.
                # Only nonexistent entries of a partial static batch start ready.
                # Real tail groups are polled even when they have no output.
                for batch in cutlass.range_constexpr(((groups // 8) + 3) // 4):
                    words = cute.make_rmem_tensor((4,), Int32)
                    codes = cute.make_rmem_tensor((4,), Int32)
                    for b in cutlass.range_constexpr(4):
                        words[b] = Int32(0)
                        codes[b] = Int32(0)
                        if cutlass.const_expr(batch * 4 + b < groups // 8):
                            group = half * 4 + lane // 8 + 8 * (batch * 4 + b)
                            offset = group * 32 + (lane % 8) * 4
                            words[b] = _load_payload4(remote + offset)
                            codes[b] = _load_scale(remote + tile_elems + group)
                    pending = Int32(0)
                    for b in cutlass.range_constexpr(4):
                        pending = pending | _pair_bad(words[b], codes[b])
                    while pending != 0:
                        for b in cutlass.range_constexpr(4):
                            if cutlass.const_expr(batch * 4 + b < groups // 8):
                                group = half * 4 + lane // 8 + 8 * (batch * 4 + b)
                                offset = group * 32 + (lane % 8) * 4
                                if _payload_bad(words[b]) != 0:
                                    words[b] = _load_payload4(remote + offset)
                                if codes[b] == 255:
                                    codes[b] = _load_scale(remote + tile_elems + group)
                        pending = Int32(0)
                        for b in cutlass.range_constexpr(4):
                            pending = pending | _pair_bad(words[b], codes[b])
                    for b in cutlass.range_constexpr(4):
                        if cutlass.const_expr(batch * 4 + b < groups // 8):
                            group = half * 4 + lane // 8 + 8 * (batch * 4 + b)
                            if tile * tile_elems + group * 32 < n:
                                offset = group * 32 + (lane % 8) * 4
                                peer_q = words[b]
                                peer_scale = _scale_float(codes[b])
                                own_q = _load_shared32(slot + offset)
                                own_scale = _scale_float(scales[group].to(Int32))
                                if cutlass.const_expr(rank == 0):
                                    result = _decode_sum4(own_q, peer_q, own_scale, peer_scale)
                                else:
                                    result = _decode_sum4(peer_q, own_q, peer_scale, own_scale)
                                _store_global64(out.iterator + tile * tile_elems + offset, result)
                # Both warps finish reading this own slot before its reuse.
                _team_barrier(bar)
            k += 1
        if half == 0 and lane == 0:
            _bulk_wait_all()
    # Fixed all-CTA launch advances state even for no-data CTAs. Reset and all
    # team stores are fully complete before local state is advanced. Every
    # transmitted q/scale byte was observed ready, including zero-filled tails.
    cute.arch.sync_threads()
    if tid == 0:
        own_state[2 * ctas + block] = call + Int64(1)
        own_state[ctas + block] = Int64(local_count)
        own_state[block] = call + Int64(1)


@cute.jit
def _mxfp8_lamport_batched(x: cute.Tensor, out: cute.Tensor, wire: tuple, state: tuple,
                   n: cutlass.Constexpr, tile_elems: cutlass.Constexpr,
                   ctas: cutlass.Constexpr, teams: cutlass.Constexpr,
                   buffer_bytes: cutlass.Constexpr, rank: cutlass.Constexpr,
                   stream: cuda.CUstream):
    _mxfp8_lamport_batched_kernel(x, out, wire, state, n, tile_elems, ctas, teams, buffer_bytes, rank).launch(
        grid=(ctas, 1, 1), block=(64 * teams + 32, 1, 1), stream=stream,
        cooperative=True, use_pdl=False,
    )


class MXFP8LamportBatchedAllReduceWorkspace:
    """Persistent TP2 triple buffers and per-CTA call/reset state.

    All ranks call matching shapes in the same order. Calls sharing this
    workspace must be stream-serialized. Every launch uses the same fixed
    cooperative grid, bounded by the smaller peer SM count.
    """

    def __init__(self, max_numel, device, group_size=32, group=None,
                 tile_elems=2048, ctas=152, teams=7):
        if type(max_numel) is not int or not 0 < max_numel < 2 ** 31 or max_numel % 32:
            raise ValueError("max_numel must be a positive multiple of 32 below 2**31")
        if group_size != 32:
            raise ValueError("fused one-shot supports group_size=32")
        if type(tile_elems) is not int or tile_elems < 256 or tile_elems > 16384 or tile_elems % 256:
            raise ValueError("tile_elems must be a multiple of 256 in [256, 16384]")
        if ((max_numel + tile_elems - 1) // tile_elems) * tile_elems >= 2 ** 31:
            raise ValueError("padded tile capacity must remain below 2**31 elements")
        if type(teams) is not int or not 1 <= teams <= 7:
            raise ValueError("teams must be an integer in [1, 7]")
        self.teams = teams
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("workspace requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("workspace construction must precede graph capture")
        self.group = group if group is not None else dist.group.WORLD
        self.world, self.rank = dist.get_world_size(self.group), dist.get_rank(self.group)
        if self.world != 2:
            raise ValueError("fused one-shot currently supports TP2 only")
        if torch.cuda.get_device_capability(self.device)[0] != 10:
            raise RuntimeError("this fused implementation is qualified for SM100-family devices only")
        valid_wire_bytes = tile_elems + tile_elems // 32
        wire_bytes = ((valid_wire_bytes + 15) // 16) * 16
        team_stride_bytes = ((2 * tile_elems + 2 * wire_bytes + 127) // 128) * 128
        shared_memory_bytes = teams * team_stride_bytes + wire_bytes + 8 * teams
        result, shared_memory_limit = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            self.device.index,
        )
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGetAttribute shared-memory limit failed: {result}")
        if shared_memory_bytes > shared_memory_limit:
            raise ValueError("team geometry exceeds the device per-block opt-in shared-memory limit")
        sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        peer_sms = [None] * self.world
        dist.all_gather_object(peer_sms, sm_count, group=self.group)
        max_ctas = min(peer_sms)
        self.max_numel, self.gs, self.tile_elems = max_numel, group_size, tile_elems
        self.max_tiles = (max_numel + tile_elems - 1) // tile_elems
        self.ctas = max_ctas if ctas is None else ctas
        if type(self.ctas) is not int or not 1 <= self.ctas <= max_ctas:
            raise ValueError("ctas must be between 1 and the minimum peer SM count")
        # Rendezvous must not silently accept inconsistent capacity/tiling.
        config = (max_numel, group_size, tile_elems, self.ctas, teams)
        peer_configs = [None] * self.world
        dist.all_gather_object(peer_configs, config, group=self.group)
        if any(other != config for other in peer_configs):
            raise ValueError("all ranks must use identical workspace geometry")
        self.allocator_backend = symm_mem.get_backend(self.device)
        if self.allocator_backend not in ("CUDA", "NCCL"):
            raise RuntimeError("CUDA or NCCL symmetric-memory allocator is required")
        if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 11):
            symm_mem.enable_symm_mem_for_group(self.group.group_name)
        self.valid_wire_bytes = tile_elems + tile_elems // 32
        self.wire_bytes = ((self.valid_wire_bytes + 15) // 16) * 16
        self.tensors, self.handles, self.peers = [], [], []

        def allocate(count, dtype):
            tensor = symm_mem.empty((count,), dtype=dtype, device=self.device)
            handle = symm_mem.rendezvous(tensor, group=self.group)
            peers = []
            for peer in range(self.world):
                if hasattr(handle, "get_remote_tensor"):
                    view = handle.get_remote_tensor(peer, tensor.shape, dtype)
                else:
                    view = handle.get_buffer(peer, tensor.shape, dtype, storage_offset=0)
                peers.append(view)
            self.tensors.append(tensor)
            self.handles.append(handle)
            self.peers.append(peers)
            return tensor

        self.buffer_bytes = self.max_tiles * self.wire_bytes
        self.receive = allocate(3 * self.buffer_bytes, torch.uint8)
        self.receive.fill_(255)
        self.state = allocate(3 * self.ctas, torch.int64)
        self.state.zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.metadata = dict(
            backend="fp8-mxfp8-lamport-batched", poll_batch_groups=4, reset_batch_tiles=4, world_size=2, group_size=32, max_numel=max_numel,
            tile_elems=tile_elems, ctas=self.ctas, teams_per_cta=teams,
            threads_per_cta=64 * teams + 32, worker_warps_per_cta=2 * teams,
            reset_warps_per_cta=1, team_stride_bytes=team_stride_bytes,
            shared_memory_bytes=shared_memory_bytes,
            device_shared_memory_limit_bytes=int(shared_memory_limit),
            wire_bytes_per_tile=self.wire_bytes, valid_wire_bytes=self.valid_wire_bytes,
            receive_buffers=3, receive_buffer_stride_bytes=self.buffer_bytes,
            wire_layout="tile_major_fp8_bytes_then_e8m0_group32_scales_then_zero_padding",
            state_layout="cta_call_counters_then_previous_tile_counts_then_completed_epochs",
            scale_format="E8M0", scale_floor_exponent=-127, zero_scale_code=0,
            input_pipeline_stages=1, outgoing_pipeline_stages=2,
            supported_numerics="finite_source_terms_and_fp32_partials_and_bf16_representable_output",
            overflow_policy="unsupported_inputs_can_produce_nonfinite_output; no_saturation",
            source_sha256=MXFP8_LAMPORT_BATCHED_SOURCE_SHA256,
            baseline_source_sha256=BASELINE_SOURCE_SHA256,
            group128_source_sha256=BASELINE_SOURCE_SHA256,
            scale_recipe="NVIDIA TensorRT round-up-to-E8M0", standard_mxfp8=True,
            lanes_per_group=8, values_per_lane=4,
            symmetric_allocation_count=2,
            symmetric_bytes_per_rank=self.receive.numel() + self.state.numel() * 8,
            allocator_backend=self.allocator_backend, requires_serial_use=True,
            protocol="payload_ff_lamport_triple_buffer",
            payload_sentinel=255, end_of_call_handshake=False, polls_padded_groups=True,
            cooperative_launch=True, numeric_contract=NUMERIC_CONTRACT,
        )

    def protocol_snapshot(self):
        """Untimed host diagnostic; never called inside CUDA graph capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("protocol_snapshot is an untimed host diagnostic")
        values = self.state.detach().cpu().tolist()
        calls = values[:self.ctas]
        return dict(call_counters=calls,
                    previous_tile_counts=values[self.ctas:2 * self.ctas],
                    completed_epochs=values[2 * self.ctas:],
                    latest_buffers=[(call - 1) % 3 for call in calls])



class MXFP8LamportBatchedAllReduce:
    """Fixed-shape callable; construction/compilation is outside graph capture."""

    def __init__(self, example, group_size=32, group=None, workspace=None):
        if not example.is_cuda or example.dtype != torch.bfloat16 or not example.is_contiguous():
            raise ValueError("expected contiguous CUDA BF16 input")
        if not 0 < example.numel() < 2 ** 31 or example.numel() % 32:
            raise ValueError("input numel must be a positive multiple of 32 below 2**31")
        if group_size != 32:
            raise ValueError("group_size must be 32")
        if example.data_ptr() % 16:
            raise ValueError("TMA input requires 16-byte alignment")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile shape specialization before graph capture")
        self.group = group if group is not None else dist.group.WORLD
        if workspace is None:
            workspace = MXFP8LamportBatchedAllReduceWorkspace(example.numel(), example.device, group_size, self.group)
        if not isinstance(workspace, MXFP8LamportBatchedAllReduceWorkspace):
            raise TypeError("workspace must be MXFP8LamportBatchedAllReduceWorkspace")
        if workspace.group is not self.group or workspace.device != example.device:
            raise ValueError("workspace group/device mismatch")
        if (dist.get_world_size(self.group), dist.get_rank(self.group)) != (workspace.world, workspace.rank):
            raise ValueError("workspace process-group rank/size changed")
        if example.numel() > workspace.max_numel:
            raise ValueError("input exceeds workspace capacity")
        self.workspace = workspace
        self.n, self.shape, self.device = example.numel(), example.shape, example.device
        self.rank, self.world = workspace.rank, workspace.world
        self.out = torch.empty_like(example)
        self.wire = tuple(_cute(t, align=16) for t in workspace.peers[0])
        self.flags = tuple(_cute(t, align=8) for t in workspace.peers[1])
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.kernel = cute.compile(
            _mxfp8_lamport_batched, _cute(example.view(-1), align=16), _cute(self.out.view(-1), align=16),
            self.wire, self.flags, self.n, workspace.tile_elems, workspace.ctas,
            workspace.teams, workspace.buffer_bytes, self.rank, stream,
        )
        self.metadata = dict(
            **workspace.metadata, input_numel=self.n,
            active_ctas=workspace.ctas,
            data_active_ctas=min(workspace.ctas, (self.n + workspace.tile_elems - 1) // workspace.tile_elems),
            cuda_kernels_per_call=1, global_barrier_kernels_per_call=0,
            quantization="source-local E4M3 RNE, group32, scale=2**max(-127,ceil(log2(amax/448))), E8M0 code=e+127; zero code0 represents 2**-127",
            accumulation="mul.rn.f32 then add.rn.f32, +0 seed, rank0 then rank1",
            output="BF16 RNE; no second FP8 quantization", status="experimental",
            transport="TMA input/peer push with direct packed volatile payload polling",
            codec_io="4 contiguous BF16 values per lane; 8 lanes per group32; FP8x2 and BF16x2 conversion",
        )

    def __call__(self, x, out=None):
        out = self.out if out is None else out
        for tensor in (x, out):
            if (tensor.shape != self.shape or tensor.device != self.device
                    or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous()
                    or tensor.data_ptr() % 16):
                raise ValueError("input/output must match specialization and be 16-byte aligned")
        nbytes = self.n * 2
        if x.data_ptr() < out.data_ptr() + nbytes and out.data_ptr() < x.data_ptr() + nbytes:
            raise ValueError("fused one-shot requires nonoverlapping input/output buffers")
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.kernel(_cute(x.view(-1), align=16), _cute(out.view(-1), align=16),
                    self.wire, self.flags, stream)
        return out

    def sample_received_groups(self, indices):
        """Untimed diagnostic views of the OTHER rank's transmitted source codec."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if indices.ndim != 1 or bool(((indices < 0) | (indices >= self.n // 32)).any().item()):
            raise ValueError("group indices must be a one-dimensional in-range sequence")
        workspace = self.workspace
        groups_per_tile = workspace.tile_elems // 32
        tile, within = indices // groups_per_tile, indices % groups_per_tile
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("sample_received_groups is an untimed host diagnostic")
        call = int(workspace.state[0].item())
        if call < 1:
            raise RuntimeError("sample_received_groups requires a completed call")
        buffer_offset = ((call - 1) % 3) * workspace.buffer_bytes
        q_offsets = buffer_offset + tile[:, None] * workspace.wire_bytes + within[:, None] * 32
        q_offsets = q_offsets + torch.arange(32, device=self.device)[None, :]
        q = workspace.receive[q_offsets]
        scale_offsets = buffer_offset + tile * workspace.wire_bytes + workspace.tile_elems + within
        scale_codes = workspace.receive[scale_offsets]
        return dict(q=q, scale_codes=scale_codes, source_rank=1 - self.rank)
