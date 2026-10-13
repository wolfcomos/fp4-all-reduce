"""Fused TP2 FP8 one-shot AllReduce using TMA and tile epoch publication.

One cooperative, persistent CUDA kernel performs BF16 TMA input loads, local
group-128 E4M3 quantization, TMA pushes into the peer's receive buffer, ordered
FP32 decode/sum, and BF16 output. Input tile k+1 is prefetched while tile k is
processed. Readiness is a separate 64-bit tile epoch, NOT a payload sentinel;
all finite FP8 codes, including negative zero (0x80), remain legal.

An acknowledgement protects each producer's previous transfer from reuse.
Epochs are per persistent CTA and survive launches/graph replay; the workspace's
CTA-to-tile mapping is fixed across shape specializations. All ranks must invoke
matching shapes in the same order, and workspace use must be stream-serialized.
This is an experimental implementation, not the existing two-shot SM codec.
"""

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


NUMERIC_CONTRACT = "tp2_fp8_oneshot_fp32_rank_order_v1"


@dsl_user_op
def _div_rn(a: Float32, b: Float32, *, loc=None, ip=None):
    value = llvm.inline_asm(
        T.f32(), [a.ir_value(), b.ir_value()], "div.rn.f32 $0, $1, $2;",
        "=f,f,f", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip,
    )
    return Float32(value)


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
def _fused_kernel(x: cute.Tensor, out: cute.Tensor, wire: tuple, flags: tuple,
                  n: cutlass.Constexpr, tile_elems: cutlass.Constexpr,
                  ctas: cutlass.Constexpr, rank: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane, warp = tid % 32, tid // 32
    groups_per_tile = tile_elems // 128
    wire_bytes = tile_elems + groups_per_tile * 4
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
    own_scales = cute.make_tensor(cute.recast_ptr(send_ptr + tile_elems, dtype=Float32), cute.make_layout((groups_per_tile,)))
    peer_q = cute.make_tensor(cute.recast_ptr(recv_ptr, dtype=Float8E4M3FN), cute.make_layout((tile_elems,)))
    peer_scales = cute.make_tensor(cute.recast_ptr(recv_ptr + tile_elems, dtype=Float32), cute.make_layout((groups_per_tile,)))
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

        for k in cutlass.range_constexpr(groups_per_tile // 4):
            group = warp + 4 * k
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
            scale = cute.arch.fmax(_div_rn(amax, Float32(448.0)), Float32(2.0 ** -120))
            if lane == 0:
                own_scales[group] = scale
            for j in cutlass.range_constexpr(4):
                normalized = _div_rn(values[j], scale)
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

        for k in cutlass.range_constexpr(groups_per_tile // 4):
            group = warp + 4 * k
            for j in cutlass.range_constexpr(4):
                offset = group * 128 + j * 32 + lane
                acc = Float32(0.0)
                if cutlass.const_expr(rank == 0):
                    acc = _mul_add_fp32(own_q[offset].to(Float32), own_scales[group], acc)
                    acc = _mul_add_fp32(peer_q[offset].to(Float32), peer_scales[group], acc)
                else:
                    acc = _mul_add_fp32(peer_q[offset].to(Float32), peer_scales[group], acc)
                    acc = _mul_add_fp32(own_q[offset].to(Float32), own_scales[group], acc)
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
def _fused(x: cute.Tensor, out: cute.Tensor, wire: tuple, flags: tuple,
           n: cutlass.Constexpr, tile_elems: cutlass.Constexpr,
           ctas: cutlass.Constexpr, rank: cutlass.Constexpr, stream: cuda.CUstream):
    active_ctas = min(ctas, (n + tile_elems - 1) // tile_elems)
    _fused_kernel(x, out, wire, flags, n, tile_elems, ctas, rank).launch(
        grid=(active_ctas, 1, 1), block=(128, 1, 1), stream=stream,
        cooperative=True, use_pdl=False,
    )


class FusedFP8AllReduceWorkspace:
    """Collectively constructed, persistent TP2 mappings and tile epochs.

    Launches sharing this workspace must be serial on each GPU. Same-shaped
    calls and shape changes must have the same order on the two ranks. The
    cooperative grid is bounded by the smaller peer SM count and fixed here.
    """

    def __init__(self, max_numel, device, group_size=128, group=None,
                 tile_elems=2048, ctas=None):
        if type(max_numel) is not int or not 0 < max_numel < 2 ** 31 or max_numel % 128:
            raise ValueError("max_numel must be a positive multiple of 128 below 2**31")
        if group_size != 128:
            raise ValueError("fused one-shot supports group_size=128")
        if type(tile_elems) is not int or tile_elems < 512 or tile_elems > 16384 or tile_elems % 512:
            raise ValueError("tile_elems must be a multiple of 512 in [512, 16384]")
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
        sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        peer_sms = [None] * self.world
        dist.all_gather_object(peer_sms, sm_count, group=self.group)
        max_ctas = min(peer_sms)
        self.max_numel, self.gs, self.tile_elems = max_numel, group_size, tile_elems
        self.max_tiles = (max_numel + tile_elems - 1) // tile_elems
        self.ctas = min(max_ctas, self.max_tiles) if ctas is None else ctas
        if type(self.ctas) is not int or not 1 <= self.ctas <= max_ctas:
            raise ValueError("ctas must be between 1 and the minimum peer SM count")
        # Rendezvous must not silently accept inconsistent capacity/tiling.
        config = (max_numel, group_size, tile_elems, self.ctas)
        peer_configs = [None] * self.world
        dist.all_gather_object(peer_configs, config, group=self.group)
        if any(other != config for other in peer_configs):
            raise ValueError("all ranks must use identical workspace geometry")
        self.allocator_backend = symm_mem.get_backend(self.device)
        if self.allocator_backend not in ("CUDA", "NCCL"):
            raise RuntimeError("CUDA or NCCL symmetric-memory allocator is required")
        if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 11):
            symm_mem.enable_symm_mem_for_group(self.group.group_name)
        self.wire_bytes = tile_elems + (tile_elems // 128) * 4
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
            backend="fused-sm", world_size=2, group_size=128, max_numel=max_numel,
            tile_elems=tile_elems, ctas=self.ctas, threads_per_cta=128,
            wire_bytes_per_tile=self.wire_bytes, input_pipeline_stages=2,
            symmetric_allocation_count=2,
            symmetric_bytes_per_rank=self.receive.numel() + self.state.numel() * 8,
            allocator_backend=self.allocator_backend, requires_serial_use=True,
            protocol="sideband_int64_tile_epoch_and_ack", payload_sentinel=False,
            cooperative_launch=True, numeric_contract=NUMERIC_CONTRACT,
        )


class FusedFP8AllReduce:
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
            workspace = FusedFP8AllReduceWorkspace(example.numel(), example.device, group_size, self.group)
        if not isinstance(workspace, FusedFP8AllReduceWorkspace):
            raise TypeError("workspace must be FusedFP8AllReduceWorkspace")
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
            _fused, _cute(example.view(-1), align=16), _cute(self.out.view(-1), align=16),
            self.wire, self.flags, self.n, workspace.tile_elems, workspace.ctas, self.rank, stream,
        )
        self.metadata = dict(
            **workspace.metadata, input_numel=self.n,
            active_ctas=min(workspace.ctas, (self.n + workspace.tile_elems - 1) // workspace.tile_elems),
            cuda_kernels_per_call=1, global_barrier_kernels_per_call=0,
            quantization="source-local E4M3 RNE, group128, scale=max(amax/448,2**-120)",
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
        scale_offsets = tile * (workspace.wire_bytes // 4) + workspace.tile_elems // 4 + within
        scales = workspace.receive.view(torch.float32)[scale_offsets]
        return dict(q=q, scales=scales, source_rank=1 - self.rank)
