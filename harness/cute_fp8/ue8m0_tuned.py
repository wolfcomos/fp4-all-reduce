"""Geometry-tuned TP2 FP8 one-shot AllReduce with group-128 UE8M0 scales.

BF16 input is quantized to E4M3 RNE with scale
2**max(-120, ceil(log2(amax/448))). The wire stores one scale byte e+127
per group, including byte7 for zero/subnormal groups at the scale floor.
Exponent/mantissa tests select the scale; exact power-of-two multiplication
normalizes the input. No FP32 division, logarithm, or reciprocal approximation
is used. TMA, sideband epoch/ACK synchronization, input prefetch and ordered
FP32 decode/sum are preserved from the serial fused.py baseline.
This module changes only tile, CTA and thread geometry from ue8m0.py.

This codec deliberately changes the scale contract from fused.py. Supported
inputs require finite BF16 source values, finite decoded source terms and FP32
partial sums, and BF16-representable output. The codec does not silently clip
an overflowing dequantized term: max finite BF16 can round to q=256 at e=120,
whose dequantization overflows FP32 even when exact source values cancel.
All ranks must invoke matching shapes in order; workspace calls are serialized.
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
from cutlass import BFloat16, Float32, Float8E4M3FN, Int32, Int64, Uint8
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm

from .kernels import _cute, _mul_add_fp32, _warp_max


NUMERIC_CONTRACT = "tp2_fp8_ue8m0_group128_oneshot_v1"
UE8M0_TUNED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BASELINE_SOURCE_SHA256 = "3b92a6af3115b286b4ee3a65f5c9d8a97f6c15c305247999209deba0a4f3b032"


@dsl_user_op
def _scale_code(amax: Float32, *, loc=None, ip=None):
    """ceil(log2(amax/448)), floored at -120, encoded as e+127.

    For a normal amax=2**E*m, E-8 is sufficient exactly when m<=1.75.
    Zero and subnormal exponent fields give candidates below -120, so the
    floor handles them without log2, division, or a leading-zero operation.
    The caller supplies a finite nonnegative maximum of BF16 values.
    """
    code = llvm.inline_asm(
        T.i32(), [amax.ir_value()],
        "{ .reg .b32 bits, field, mant, bump; .reg .s32 exponent; .reg .pred p; "
        "mov.b32 bits, $1; and.b32 bits, bits, 0x7fffffff; "
        "shr.u32 field, bits, 23; and.b32 mant, bits, 0x007fffff; "
        "add.s32 exponent, field, -135; setp.gt.u32 p, mant, 0x00600000; "
        "selp.b32 bump, 1, 0, p; add.s32 exponent, exponent, bump; "
        "max.s32 exponent, exponent, -120; add.s32 $0, exponent, 127; }",
        "=r,f", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip,
    )
    return Int32(code)


@dsl_user_op
def _scale_float(code: Int32, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.f32(), [code.ir_value()],
        "{ .reg .b32 bits; and.b32 bits, $1, 0xff; "
        "shl.b32 bits, bits, 23; mov.b32 $0, bits; }",
        "=f,r", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip,
    )
    return Float32(value)


@dsl_user_op
def _inverse_scale_float(code: Int32, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.f32(), [code.ir_value()],
        "{ .reg .b32 bits; sub.u32 bits, 254, $1; "
        "shl.b32 bits, bits, 23; mov.b32 $0, bits; }",
        "=f,r", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip,
    )
    return Float32(value)


@dsl_user_op
def _mul_rn(value: Float32, multiplier: Float32, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.f32(), [value.ir_value(), multiplier.ir_value()],
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
        asm_dialect=0, loc=loc, ip=ip,
    )
    return Float32(result)


@dsl_user_op
def _tma_load(dst, src, nbytes: Int32, barrier, *, loc=None, ip=None):
    """One-dimensional TMA bulk G2S; caller arms the transaction barrier."""
    llvm.inline_asm(
        T.i32(), [Int32(dst.toint()).ir_value(), Int64(src.toint()).ir_value(),
                  nbytes.ir_value(), Int32(barrier.toint()).ir_value()],
        "{ cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$1], [$2], $3, [$4]; mov.u32 $0, 0; }",
        "=r,r,l,r,r,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def _tma_store(dst, src, nbytes: Int32, *, loc=None, ip=None):
    """One-dimensional TMA bulk shared-to-global push, including peer memory."""
    llvm.inline_asm(
        T.i32(), [Int64(dst.toint()).ir_value(), Int32(src.toint()).ir_value(), nbytes.ir_value()],
        "{ cp.async.bulk.global.shared::cta.bulk_group [$1], [$2], $3; mov.u32 $0, 0; }",
        "=r,l,r,r,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def _tma_store_commit_wait(*, loc=None, ip=None):
    # Full completion is necessary before publishing readiness: .read alone
    # would only make the shared source reusable, not publish the remote write.
    llvm.inline_asm(
        T.i32(), [],
        "{ cp.async.bulk.commit_group; cp.async.bulk.wait_group 0; mov.u32 $0, 0; }",
        "=r,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def _release_epoch(ptr, epoch: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(), [Int64(ptr.toint()).ir_value(), epoch.ir_value()],
        "{ st.release.sys.global.u64 [$1], $2; mov.u32 $0, 0; }",
        "=r,l,l,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def _wait_epoch(ptr, epoch: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i64(), [Int64(ptr.toint()).ir_value(), epoch.ir_value()],
        "{ .reg .pred p; WAIT: ld.acquire.sys.global.u64 $0, [$1]; "
        "setp.ne.u64 p, $0, $2; @p bra WAIT; }",
        "=l,l,l,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@cute.kernel
def _ue8m0_tuned_kernel(x: cute.Tensor, out: cute.Tensor, wire: tuple, flags: tuple,
                        n: cutlass.Constexpr, tile_elems: cutlass.Constexpr,
                        ctas: cutlass.Constexpr, threads: cutlass.Constexpr,
                        rank: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane, warp = tid % 32, tid // 32
    groups_per_tile = tile_elems // 128
    warps = threads // 32
    valid_wire_bytes = tile_elems + groups_per_tile
    wire_bytes = ((valid_wire_bytes + 15) // 16) * 16
    tiles = (n + tile_elems - 1) // tile_elems
    # Bind concrete tensors before indexing/storing: CuTe's assignment lowering
    # does not support a tensor store through a nested Python tuple subscript.
    if cutlass.const_expr(rank == 0):
        own_flags, peer_flags = flags[0], flags[1]
        own_wire, peer_wire = wire[0], wire[1]
    else:
        own_flags, peer_flags = flags[1], flags[0]
        own_wire, peer_wire = wire[1], wire[0]

    smem = utils.SmemAllocator()
    input_ptr = smem.allocate_array(BFloat16, 2 * tile_elems, byte_alignment=128)
    send_ptr = smem.allocate_array(Uint8, wire_bytes, byte_alignment=128)
    recv_ptr = smem.allocate_array(Uint8, wire_bytes, byte_alignment=128)
    barriers = smem.allocate_array(Int64, 3, byte_alignment=8)
    inputs = cute.make_tensor(input_ptr, cute.make_layout((2 * tile_elems,)))
    own_q = cute.make_tensor(cute.recast_ptr(send_ptr, dtype=Float8E4M3FN), cute.make_layout((tile_elems,)))
    own_scales = cute.make_tensor(cute.recast_ptr(send_ptr + tile_elems, dtype=Uint8), cute.make_layout((groups_per_tile,)))
    peer_q = cute.make_tensor(cute.recast_ptr(recv_ptr, dtype=Float8E4M3FN), cute.make_layout((tile_elems,)))
    peer_scales = cute.make_tensor(cute.recast_ptr(recv_ptr + tile_elems, dtype=Uint8), cute.make_layout((groups_per_tile,)))
    if cutlass.const_expr(wire_bytes > valid_wire_bytes):
        send_bytes = cute.make_tensor(send_ptr, cute.make_layout((wire_bytes,)))
        if tid < wire_bytes - valid_wire_bytes:
            send_bytes[valid_wire_bytes + tid] = Uint8(0)
    # Each CTA exclusively owns its counter; the first CTA barrier ensures all
    # lanes read it before thread zero can update it at the end of this launch.
    epoch = own_flags[2 * ctas + block]
    if tid == 0:
        cute.arch.mbarrier_init(barriers, 1)
        cute.arch.mbarrier_init(barriers + 1, 1)
        cute.arch.mbarrier_init(barriers + 2, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    if tid == 0:
        first_bytes = Int32(tile_elems * 2)
        if n - block * tile_elems < tile_elems:
            first_bytes = Int32((n - block * tile_elems) * 2)
        cute.arch.mbarrier_arrive_and_expect_tx(barriers, first_bytes)
        _tma_load(input_ptr, x.iterator + block * tile_elems, first_bytes, barriers)

    iteration = Int32(0)
    tile = Int32(block)
    while tile < tiles:
        stage = iteration % 2
        phase = (iteration // 2) % 2
        cute.arch.mbarrier_wait(barriers + stage, phase)
        cute.arch.sync_threads()
        # The other input stage is no longer read by any lane. Prefetch the next
        # persistent tile while this tile is quantized and sent over NVLink.
        next_tile = tile + ctas
        if tid == 0:
            if next_tile < tiles:
                next_stage = 1 - stage
                next_bytes = Int32(tile_elems * 2)
                if n - next_tile * tile_elems < tile_elems:
                    next_bytes = Int32((n - next_tile * tile_elems) * 2)
                cute.arch.mbarrier_arrive_and_expect_tx(barriers + next_stage, next_bytes)
                _tma_load(input_ptr + next_stage * tile_elems,
                          x.iterator + next_tile * tile_elems, next_bytes, barriers + next_stage)

        for k in cutlass.range_constexpr(groups_per_tile // warps):
            group = warp + warps * k
            values = cute.make_rmem_tensor((4,), Float32)
            amax = Float32(0.0)
            for j in cutlass.range_constexpr(4):
                offset = group * 128 + j * 32 + lane
                value = Float32(0.0)
                if tile * tile_elems + offset < n:
                    value = inputs[stage * tile_elems + offset].to(Float32)
                values[j] = value
                amax = cute.arch.fmax(amax, cute.arch.fmax(value, -value))
            amax = _warp_max(amax)
            code = _scale_code(amax)
            inverse = _inverse_scale_float(code)
            if lane == 0:
                own_scales[group] = code.to(Uint8)
            for j in cutlass.range_constexpr(4):
                normalized = _mul_rn(values[j], inverse)
                normalized = cute.arch.fmax(Float32(-448.0), cute.arch.fmin(Float32(448.0), normalized))
                own_q[group * 128 + j * 32 + lane] = normalized.to(Float8E4M3FN)

        # Every thread publishes its shared writes to the async proxy. The CTA
        # barrier joins them before the issuing thread starts the bulk transfer.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        if tid == 0:
            _wait_epoch(own_flags.iterator + ctas + block, epoch)
            _tma_store(peer_wire.iterator + Int64(tile) * wire_bytes, send_ptr, Int32(wire_bytes))
            _tma_store_commit_wait()
            _release_epoch(peer_flags.iterator + block, epoch + Int64(1))
            _wait_epoch(own_flags.iterator + block, epoch + Int64(1))
            # Read the received payload with TMA as well; no stale L1-cached
            # ordinary global loads can survive between repeated invocations.
            cute.arch.mbarrier_arrive_and_expect_tx(barriers + 2, wire_bytes)
            _tma_load(recv_ptr, own_wire.iterator + Int64(tile) * wire_bytes, Int32(wire_bytes), barriers + 2)
        cute.arch.sync_threads()
        cute.arch.mbarrier_wait(barriers + 2, iteration % 2)

        for k in cutlass.range_constexpr(groups_per_tile // warps):
            group = warp + warps * k
            for j in cutlass.range_constexpr(4):
                offset = group * 128 + j * 32 + lane
                acc = Float32(0.0)
                if cutlass.const_expr(rank == 0):
                    acc = _mul_add_fp32(own_q[offset].to(Float32), _scale_float(own_scales[group].to(Int32)), acc)
                    acc = _mul_add_fp32(peer_q[offset].to(Float32), _scale_float(peer_scales[group].to(Int32)), acc)
                else:
                    acc = _mul_add_fp32(peer_q[offset].to(Float32), _scale_float(peer_scales[group].to(Int32)), acc)
                    acc = _mul_add_fp32(own_q[offset].to(Float32), _scale_float(own_scales[group].to(Int32)), acc)
                if tile * tile_elems + offset < n:
                    out[tile * tile_elems + offset] = acc.to(BFloat16)
        cute.arch.sync_threads()
        epoch += Int64(1)
        if tid == 0:
            _release_epoch(peer_flags.iterator + ctas + block, epoch)
        cute.arch.sync_threads()
        iteration += 1
        tile += ctas
    if tid == 0:
        own_flags[2 * ctas + block] = epoch


@cute.jit
def _ue8m0_tuned(x: cute.Tensor, out: cute.Tensor, wire: tuple, flags: tuple,
                 n: cutlass.Constexpr, tile_elems: cutlass.Constexpr,
                 ctas: cutlass.Constexpr, threads: cutlass.Constexpr,
                 rank: cutlass.Constexpr, stream: cuda.CUstream):
    active_ctas = min(ctas, (n + tile_elems - 1) // tile_elems)
    _ue8m0_tuned_kernel(x, out, wire, flags, n, tile_elems, ctas, threads, rank).launch(
        grid=(active_ctas, 1, 1), block=(threads, 1, 1), stream=stream,
        cooperative=True, use_pdl=False,
    )


class TunedUE8M0FP8AllReduceWorkspace:
    """Collectively constructed, persistent TP2 mappings and tile epochs.

    Launches sharing this workspace must be serial on each GPU. Same-shaped
    calls and shape changes must have the same order on the two ranks. The
    cooperative grid is bounded by the smaller peer SM count and fixed here.
    """

    def __init__(self, max_numel, device, group_size=128, group=None,
                 tile_elems=16384, ctas=None, threads=128):
        if type(max_numel) is not int or not 0 < max_numel < 2 ** 31 or max_numel % 128:
            raise ValueError("max_numel must be a positive multiple of 128 below 2**31")
        if group_size != 128:
            raise ValueError("fused one-shot supports group_size=128")
        if type(threads) is not int or threads not in (128, 256):
            raise ValueError("threads must be 128 or 256")
        if (type(tile_elems) is not int or tile_elems < 512 or tile_elems > 32768
                or tile_elems % (threads * 4)):
            raise ValueError("tile_elems must be in [512, 32768] and divisible by threads * 4")
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
        valid_wire_bytes = tile_elems + tile_elems // 128
        wire_bytes = ((valid_wire_bytes + 15) // 16) * 16
        # recv_ptr starts at the next 128-byte boundary after the send payload.
        shared_memory_bytes = 4 * tile_elems + ((wire_bytes + 127) // 128) * 128 + wire_bytes + 24
        result, shared_memory_limit = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            self.device.index,
        )
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGetAttribute shared-memory limit failed: {result}")
        if shared_memory_bytes > shared_memory_limit:
            raise ValueError("tile requires more shared memory than this device's per-block opt-in limit")
        sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        peer_sms = [None] * self.world
        dist.all_gather_object(peer_sms, sm_count, group=self.group)
        max_ctas = min(peer_sms)
        self.max_numel, self.gs, self.tile_elems = max_numel, group_size, tile_elems
        self.threads = threads
        self.max_tiles = (max_numel + tile_elems - 1) // tile_elems
        self.ctas = min(max_ctas, self.max_tiles) if ctas is None else ctas
        if type(self.ctas) is not int or not 1 <= self.ctas <= max_ctas:
            raise ValueError("ctas must be between 1 and the minimum peer SM count")
        # Rendezvous must not silently accept inconsistent capacity/tiling.
        config = (max_numel, group_size, tile_elems, self.ctas, threads)
        peer_configs = [None] * self.world
        dist.all_gather_object(peer_configs, config, group=self.group)
        if any(other != config for other in peer_configs):
            raise ValueError("all ranks must use identical workspace geometry")
        self.allocator_backend = symm_mem.get_backend(self.device)
        if self.allocator_backend not in ("CUDA", "NCCL"):
            raise RuntimeError("CUDA or NCCL symmetric-memory allocator is required")
        if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 11):
            symm_mem.enable_symm_mem_for_group(self.group.group_name)
        self.valid_wire_bytes, self.wire_bytes = valid_wire_bytes, wire_bytes
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

        self.receive = allocate(self.max_tiles * self.wire_bytes, torch.uint8)
        self.state = allocate(3 * self.ctas, torch.int64)
        self.state.zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.metadata = dict(
            backend="fp8-ue8m0-tuned", world_size=2, group_size=128, max_numel=max_numel,
            tile_elems=tile_elems, ctas=self.ctas, threads_per_cta=threads,
            warps_per_cta=threads // 32,
            shared_memory_bytes=shared_memory_bytes,
            device_shared_memory_limit_bytes=int(shared_memory_limit),
            baseline_source_sha256=BASELINE_SOURCE_SHA256,
            wire_bytes_per_tile=self.wire_bytes, valid_wire_bytes=self.valid_wire_bytes,
            wire_layout="tile_major_fp8_bytes_then_ue8m0_group_scales_then_zero_padding",
            scale_format="UE8M0", scale_floor_exponent=-120, zero_scale_code=7,
            input_pipeline_stages=2, outgoing_pipeline_stages=1,
            supported_numerics="finite_source_terms_and_fp32_partials_and_bf16_representable_output",
            overflow_policy="unsupported_inputs_can_produce_nonfinite_output; no_saturation",
            source_sha256=UE8M0_TUNED_SOURCE_SHA256,
            symmetric_allocation_count=2,
            symmetric_bytes_per_rank=self.receive.numel() + self.state.numel() * 8,
            allocator_backend=self.allocator_backend, requires_serial_use=True,
            protocol="sideband_int64_tile_epoch_and_ack", payload_sentinel=False,
            cooperative_launch=True, numeric_contract=NUMERIC_CONTRACT,
        )


class TunedUE8M0FP8AllReduce:
    """Fixed-shape callable; construction/compilation is outside graph capture."""

    def __init__(self, example, group_size=128, group=None, workspace=None):
        if not example.is_cuda or example.dtype != torch.bfloat16 or not example.is_contiguous():
            raise ValueError("expected contiguous CUDA BF16 input")
        if not 0 < example.numel() < 2 ** 31 or example.numel() % 128:
            raise ValueError("input numel must be a positive multiple of 128 below 2**31")
        if group_size != 128:
            raise ValueError("group_size must be 128")
        if example.data_ptr() % 16:
            raise ValueError("TMA input requires 16-byte alignment")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile shape specialization before graph capture")
        self.group = group if group is not None else dist.group.WORLD
        if workspace is None:
            workspace = TunedUE8M0FP8AllReduceWorkspace(example.numel(), example.device, group_size, self.group)
        if not isinstance(workspace, TunedUE8M0FP8AllReduceWorkspace):
            raise TypeError("workspace must be TunedUE8M0FP8AllReduceWorkspace")
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
            _ue8m0_tuned, _cute(example.view(-1), align=16), _cute(self.out.view(-1), align=16),
            self.wire, self.flags, self.n, workspace.tile_elems, workspace.ctas, workspace.threads, self.rank, stream,
        )
        self.metadata = dict(
            **workspace.metadata, input_numel=self.n,
            active_ctas=min(workspace.ctas, (self.n + workspace.tile_elems - 1) // workspace.tile_elems),
            cuda_kernels_per_call=1, global_barrier_kernels_per_call=0,
            quantization="source-local E4M3 RNE, group128, scale=2**max(-120,ceil(log2(amax/448))), UE8M0 code=e+127",
            accumulation="mul.rn.f32 then add.rn.f32, +0 seed, rank0 then rank1",
            output="BF16 RNE; no second FP8 quantization", status="experimental",
            transport="1D TMA bulk G2S input and shared-to-peer-global push",
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
        if indices.ndim != 1 or bool(((indices < 0) | (indices >= self.n // 128)).any().item()):
            raise ValueError("group indices must be a one-dimensional in-range sequence")
        workspace = self.workspace
        groups_per_tile = workspace.tile_elems // 128
        tile, within = indices // groups_per_tile, indices % groups_per_tile
        q_offsets = tile[:, None] * workspace.wire_bytes + within[:, None] * 128
        q_offsets = q_offsets + torch.arange(128, device=self.device)[None, :]
        q = workspace.receive[q_offsets]
        scale_offsets = tile * workspace.wire_bytes + workspace.tile_elems + within
        scale_codes = workspace.receive[scale_offsets]
        return dict(q=q, scale_codes=scale_codes, source_rank=1 - self.rank)
