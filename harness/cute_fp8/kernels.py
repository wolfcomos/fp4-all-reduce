"""Experimental CuTe DSL FP8 transport AllReduce, BF16 input/output.

This deliberately uses separate kernels and full inter-rank barriers. It is a
correctness/architecture prototype, not a low-latency implementation. Finite
inputs, finite quantized/dequantized intermediates, and BF16-representable
outputs are required; checking the original FP32 sum alone is insufficient.
Instances can share explicit scratch memory. All operations using a workspace
must execute serially, including CUDA graph replays on different streams.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.distributed as distributed
from cutlass import Float32, Float8E4M3FN, Int32
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm


class _DLPack:
    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def _cute(tensor, align=1):
    return from_dlpack(_DLPack(tensor.detach()), assumed_align=align)


@dsl_user_op
def _signal(ptr, *, send: bool, loc=None, ip=None):
    # Separate signal memory: every possible FP8 payload byte remains legal.
    sem = "release" if send else "acquire"
    old, new = (0, 1) if send else (1, 0)
    llvm.inline_asm(
        T.i32(), [ptr.toint().ir_value()],
        f"{{ .reg .pred p; L: atom.global.{sem}.sys.cas.b32 "
        f"$0, [$1], {old}, {new}; setp.ne.u32 p, $0, {old}; @p bra L; }}",
        "=r,l,~{memory}", has_side_effects=True, asm_dialect=0,
        loc=loc, ip=ip,
    )


@cute.kernel
def _barrier_kernel(flags: tuple, rank: cutlass.Constexpr, world: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    for peer in cutlass.range_constexpr(world):
        if cutlass.const_expr(peer != rank):
            if lane == peer:
                _signal(flags[peer].iterator + rank, send=True)
                _signal(flags[rank].iterator + peer, send=False)
    cute.arch.sync_threads()


@cute.jit
def _barrier(flags: tuple, rank: cutlass.Constexpr, world: cutlass.Constexpr,
             stream: cuda.CUstream):
    _barrier_kernel(flags, rank, world).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@cute.jit
def _warp_max(value: Float32):
    for k in cutlass.range_constexpr(5):
        value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1 << k))
    return value


@dsl_user_op
def _mul_add_fp32(q: Float32, scale: Float32, total: Float32, *, loc=None, ip=None):
    # Keep the precision contract explicit, independent of compiler contraction.
    value = llvm.inline_asm(
        T.f32(), [q.ir_value(), scale.ir_value(), total.ir_value()],
        "{ .reg .f32 product; mul.rn.f32 product, $1, $2; add.rn.f32 $0, product, $3; }",
        "=f,f,f,f", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip,
    )
    return Float32(value)


@cute.kernel
def _local_kernel(x: cute.Tensor, q: cute.Tensor, stats: cute.Tensor,
                  n: cutlass.Constexpr, gs: cutlass.Constexpr,
                  only_amax: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    group, _, _ = cute.arch.block_idx()
    vals = cute.make_rmem_tensor((gs // 32,), Float32)
    amax = Float32(0.0)
    for j in cutlass.range_constexpr(gs // 32):
        index = group * gs + j * 32 + lane
        value = Float32(0.0)
        if index < n:
            value = x[index].to(Float32)
        vals[j] = value
        amax = cute.arch.fmax(amax, cute.arch.fmax(value, -value))
    amax = _warp_max(amax)
    if cutlass.const_expr(only_amax):
        if lane == 0:
            stats[group] = amax
    else:
        scale = cute.arch.fmax(amax / Float32(448.0), Float32(2.0 ** -120))
        if lane == 0:
            stats[group] = scale
        for j in cutlass.range_constexpr(gs // 32):
            normalized = vals[j] / scale
            normalized = cute.arch.fmax(Float32(-448.0), cute.arch.fmin(Float32(448.0), normalized))
            q[group * gs + j * 32 + lane] = normalized.to(Float8E4M3FN)


@cute.jit
def _local(x: cute.Tensor, q: cute.Tensor, stats: cute.Tensor,
           n: cutlass.Constexpr, gs: cutlass.Constexpr, groups: cutlass.Constexpr,
           only_amax: cutlass.Constexpr, stream: cuda.CUstream):
    _local_kernel(x, q, stats, n, gs, only_amax).launch(
        grid=(groups, 1, 1), block=(32, 1, 1), stream=stream)


@cute.kernel
def _common_quant_kernel(x: cute.Tensor, q: cute.Tensor, maxima: tuple,
                         scales: cute.Tensor, n: cutlass.Constexpr,
                         gs: cutlass.Constexpr, world: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    group, _, _ = cute.arch.block_idx()
    amax = Float32(0.0)
    for peer in cutlass.range_constexpr(world):
        amax = cute.arch.fmax(amax, maxima[peer][group])
    # world is 2/4/8, so 448/world is representable in E4M3. Clamp before
    # conversion to preserve the bound even at FP32 division boundaries.
    scale = cute.arch.fmax((amax / Float32(448.0)) * Float32(world), Float32(2.0 ** -120))
    if lane == 0:
        scales[group] = scale
    for j in cutlass.range_constexpr(gs // 32):
        index = group * gs + j * 32 + lane
        value = Float32(0.0)
        if index < n:
            value = x[index].to(Float32)
        value = value / scale
        value = cute.arch.fmax(Float32(-448.0 / world), cute.arch.fmin(Float32(448.0 / world), value))
        q[index] = value.to(Float8E4M3FN)
    cute.arch.fence_proxy("alias")


@cute.jit
def _common_quant(x: cute.Tensor, q: cute.Tensor, maxima: tuple, scales: cute.Tensor,
                  n: cutlass.Constexpr, gs: cutlass.Constexpr,
                  groups: cutlass.Constexpr, world: cutlass.Constexpr,
                  stream: cuda.CUstream):
    _common_quant_kernel(x, q, maxima, scales, n, gs, world).launch(
        grid=(groups, 1, 1), block=(32, 1, 1), stream=stream)


@cute.kernel
def _sm_reduce_kernel(inputs: tuple, scales: tuple, reduced: cute.Tensor,
                      out_scales: cute.Tensor, gs: cutlass.Constexpr,
                      rank: cutlass.Constexpr, world: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    group = block * world + rank
    vals = cute.make_rmem_tensor((gs // 32,), Float32)
    amax = Float32(0.0)
    for j in cutlass.range_constexpr(gs // 32):
        index = group * gs + j * 32 + lane
        value = Float32(0.0)
        for peer in cutlass.range_constexpr(world):
            # A fixed rank order and FP32 accumulation; not AITER's FP16 sum.
            value = _mul_add_fp32(inputs[peer][index].to(Float32), scales[peer][group], value)
        vals[j] = value
        amax = cute.arch.fmax(amax, cute.arch.fmax(value, -value))
    amax = _warp_max(amax)
    scale = cute.arch.fmax(amax / Float32(448.0), Float32(2.0 ** -120))
    if lane == 0:
        out_scales[group] = scale
    for j in cutlass.range_constexpr(gs // 32):
        value = vals[j] / scale
        value = cute.arch.fmax(Float32(-448.0), cute.arch.fmin(Float32(448.0), value))
        reduced[group * gs + j * 32 + lane] = value.to(Float8E4M3FN)


@cute.jit
def _sm_reduce(inputs: tuple, scales: tuple, reduced: cute.Tensor, out_scales: cute.Tensor,
               gs: cutlass.Constexpr, groups: cutlass.Constexpr,
               rank: cutlass.Constexpr, world: cutlass.Constexpr, stream: cuda.CUstream):
    _sm_reduce_kernel(inputs, scales, reduced, out_scales, gs, rank, world).launch(
        grid=(groups // world, 1, 1), block=(32, 1, 1), stream=stream)


@cute.kernel
def _switch_reduce_kernel(input_mc: cute.Pointer, output_mc: cute.Pointer,
                          gs: cutlass.Constexpr, rank: cutlass.Constexpr,
                          world: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    group = block * world + rank
    # Aliases are unicast stores -> multicast reads, then multicast stores ->
    # unicast reads. Inter-kernel system barriers provide execution ordering.
    cute.arch.fence_proxy("alias")
    for j in cutlass.range_constexpr((gs + 127) // 128):
        offset = j * 128 + lane * 4
        if offset < gs:
            (word,) = distributed.multimem_ld_reduce_4xe4m3(input_mc + group * gs + offset)
            distributed.multimem_st_1xb32(output_mc + group * gs + offset, word)
    cute.arch.fence_proxy("alias")


@cute.jit
def _switch_reduce(input_mc: cute.Pointer, output_mc: cute.Pointer,
                   gs: cutlass.Constexpr, groups: cutlass.Constexpr,
                   rank: cutlass.Constexpr, world: cutlass.Constexpr, stream: cuda.CUstream):
    _switch_reduce_kernel(input_mc, output_mc, gs, rank, world).launch(
        grid=(groups // world, 1, 1), block=(32, 1, 1), stream=stream)


@cute.kernel
def _decode_kernel(reduced: tuple, scales: tuple, out: cute.Tensor,
                   n: cutlass.Constexpr, gs: cutlass.Constexpr,
                   world: cutlass.Constexpr, rank: cutlass.Constexpr,
                   switch: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    group, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(switch):
        cute.arch.fence_proxy("alias")
    for j in cutlass.range_constexpr(gs // 32):
        index = group * gs + j * 32 + lane
        if index < n:
            if cutlass.const_expr(switch):
                value = reduced[rank][index].to(Float32) * scales[rank][group]
            else:
                value = Float32(0.0)
                for peer in cutlass.range_constexpr(world):
                    if group % world == peer:
                        value = reduced[peer][index].to(Float32) * scales[peer][group]
            out[index] = value.to(out.element_type)


@cute.jit
def _decode(reduced: tuple, scales: tuple, out: cute.Tensor,
            n: cutlass.Constexpr, gs: cutlass.Constexpr, groups: cutlass.Constexpr,
            world: cutlass.Constexpr, rank: cutlass.Constexpr,
            switch: cutlass.Constexpr, stream: cuda.CUstream):
    _decode_kernel(reduced, scales, out, n, gs, world, rank, switch).launch(
        grid=(groups, 1, 1), block=(32, 1, 1), stream=stream)


class FP8AllReduceWorkspace:
    """Five persistent symmetric buffers shared by serial shape specializations.

    Construct collectively before capture, with identical capacity, backend and
    group size on all ranks. The workspace retains its mappings for the lifetime
    of every operation and graph that references it. It is not stream-safe.
    """

    def __init__(self, max_numel, device, group_size=128, backend="sm", group=None):
        if type(max_numel) is not int or not 0 < max_numel < 2 ** 31:
            raise ValueError("max_numel must be an integer in [1, 2**31)")
        if group_size not in (32, 128, 256) or backend not in ("sm", "switch"):
            raise ValueError("group_size must be 32/128/256; backend must be sm/switch")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("workspace requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot construct an FP8 workspace during CUDA graph capture")
        self.group = group if group is not None else dist.group.WORLD
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        if self.world not in (2, 4, 8):
            raise ValueError("prototype supports 2, 4, or 8 ranks")
        self.backend, self.gs = backend, group_size
        self.max_numel = max_numel
        if backend == "switch" and torch.cuda.get_device_capability(self.device)[0] != 10:
            raise RuntimeError("FP8 switch prototype requires datacenter Blackwell SM100 family")
        # Use the allocator registered by this PyTorch build. Some versions
        # register the default without listing it as switchable, so even
        # set_backend("CUDA") can fail when CUDA is already the default.
        # Builds can use NCCL for mapping/allocation without using NCCL in
        # our data or barrier kernels. Never replace a process-wide allocator
        # that other model components may already have used.
        self.allocator_backend = symm_mem.get_backend(self.device)
        if self.allocator_backend not in ("CUDA", "NCCL"):
            raise RuntimeError(
                f"Expected a registered CUDA or NCCL symmetric allocator, got {self.allocator_backend!r}"
            )
        if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 11):
            symm_mem.enable_symm_mem_for_group(self.group.group_name)
        self.groups = ((max_numel + group_size * self.world - 1) // (group_size * self.world)) * self.world
        self.handles = []
        self.tensors = []
        self.peers = []

        def alloc(count, dtype):
            tensor = symm_mem.empty((count,), dtype=dtype, device=self.device)
            handle = symm_mem.rendezvous(tensor, group=self.group)
            self.tensors.append(tensor)
            self.handles.append(handle)
            peers = []
            for peer in range(self.world):
                if hasattr(handle, "get_remote_tensor"):
                    view = handle.get_remote_tensor(peer, tensor.shape, dtype)
                else:
                    view = handle.get_buffer(peer, tensor.shape, dtype, storage_offset=0)
                peers.append(view)
            self.peers.append(peers)
            return tensor, handle, peers

        q, qh, qp = alloc(self.groups * group_size, torch.uint8)
        r, rh, rp = alloc(self.groups * group_size, torch.uint8)
        stats, _, sp = alloc(self.groups, torch.float32)
        scales, _, op = alloc(self.groups, torch.float32)
        flags, _, fp = alloc(self.world, torch.int32)
        flags.zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.mc_in = self.mc_out = None
        if backend == "switch":
            mc_in, mc_out = int(qh.multicast_ptr or 0), int(rh.multicast_ptr or 0)
            present = torch.tensor([bool(mc_in and mc_out)], device=self.device, dtype=torch.int32)
            dist.all_reduce(present, op=dist.ReduceOp.MIN, group=self.group)
            if not present.item():
                raise RuntimeError("FP8 switch requested but NVLink multicast mapping unavailable")
            self.mc_in = make_ptr(Float8E4M3FN, mc_in, cute.AddressSpace.gmem, assumed_align=16)
            self.mc_out = make_ptr(Float8E4M3FN, mc_out, cute.AddressSpace.gmem, assumed_align=16)
        self.metadata = dict(
            backend=backend, allocator_backend=self.allocator_backend,
            world_size=self.world, group_size=self.gs, max_numel=max_numel,
            padded_numel=self.groups * self.gs,
            symmetric_allocation_count=len(self.tensors),
            symmetric_bytes_per_rank=sum(t.numel() * t.element_size() for t in self.tensors),
            multicast_available=[bool(h.multicast_ptr) for h in self.handles[:2]],
            requires_serial_use=True,
        )

    def validate(self, *, numel, device, group_size, backend, group):
        """Reject a mismatched view before allocation, compilation or execution."""
        if type(numel) is not int or not 0 < numel <= self.max_numel:
            raise ValueError("input numel exceeds workspace capacity or is invalid")
        if torch.device(device) != self.device:
            raise ValueError("workspace device does not match input device")
        if group is not self.group:
            raise ValueError("workspace must use the same TP process group object")
        if (dist.get_world_size(group), dist.get_rank(group)) != (self.world, self.rank):
            raise ValueError("workspace TP group size/rank changed")
        if (group_size, backend) != (self.gs, self.backend):
            raise ValueError("workspace group_size/backend does not match operation")


class FP8AllReduce:
    """Fixed-shape, serial-stream experimental collective. Construct collectively.

    All ranks must call in the same order with matching shape/backend/group size.
    Switch selection is explicit and fails on missing multicast mappings. No
    NCCL call is hidden in either candidate's data or synchronization path.
    Every operation sharing a workspace must execute serially; each operation's
    default output remains separate and caller-provided outputs remain owned by
    the caller. Construction compiles kernels and must precede graph capture.
    """

    def __init__(self, example, group_size=128, backend="sm", group=None, workspace=None):
        if not example.is_cuda or example.dtype != torch.bfloat16 or not example.is_contiguous():
            raise ValueError("expected contiguous CUDA BF16 input")
        if example.numel() == 0 or example.numel() >= 2 ** 31:
            raise ValueError("numel must be in [1, 2**31)")
        if group_size not in (32, 128, 256) or backend not in ("sm", "switch"):
            raise ValueError("group_size must be 32/128/256; backend must be sm/switch")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot compile FP8 shape specializations during CUDA graph capture")
        self.group = group if group is not None else dist.group.WORLD
        self.backend, self.gs = backend, group_size
        self.shape, self.device, self.n = example.shape, example.device, example.numel()
        workspace_shared = workspace is not None
        if workspace is None:
            workspace = FP8AllReduceWorkspace(self.n, self.device, group_size, backend, self.group)
        if not isinstance(workspace, FP8AllReduceWorkspace):
            raise TypeError("workspace must be FP8AllReduceWorkspace")
        workspace.validate(numel=self.n, device=self.device, group_size=group_size,
                           backend=backend, group=self.group)
        self.workspace = workspace
        self.world, self.rank = workspace.world, workspace.rank
        self.allocator_backend = workspace.allocator_backend
        self.groups = ((self.n + group_size * self.world - 1) // (group_size * self.world)) * self.world
        # Prefixes begin at offset zero, so all shapes use the same unicast and
        # multicast base pointers. Retain owning tensors/handles on the workspace.
        self.tensors, self.handles = workspace.tensors, workspace.handles
        counts = (self.groups * group_size, self.groups * group_size,
                  self.groups, self.groups, self.world)
        self.views = [tensor[:count] for tensor, count in zip(self.tensors, counts)]
        qp, rp, sp, op, fp = [
            [tensor[:count] for tensor in peers]
            for peers, count in zip(workspace.peers, counts)
        ]
        q, r, stats, scales, flags = self.views
        self.q = _cute(q.view(torch.float8_e4m3fn))
        self.r = _cute(r.view(torch.float8_e4m3fn))
        self.qp = tuple(_cute(t.view(torch.float8_e4m3fn)) for t in qp)
        self.rp = tuple(_cute(t.view(torch.float8_e4m3fn)) for t in rp)
        self.sp, self.op = tuple(_cute(t) for t in sp), tuple(_cute(t) for t in op)
        self.stats, self.scales = _cute(stats), _cute(scales)
        self.flags = tuple(_cute(t) for t in fp)
        self.out = torch.empty_like(example)
        self.mc_in, self.mc_out = workspace.mc_in, workspace.mc_out
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        xc, yc = _cute(example.view(-1)), _cute(self.out.view(-1))
        self.sync = cute.compile(_barrier, self.flags, self.rank, self.world, stream)
        self.local = cute.compile(_local, xc, self.q, self.stats, self.n, self.gs,
                                  self.groups, backend == "switch", stream)
        if backend == "switch":
            self.quant = cute.compile(_common_quant, xc, self.q, self.sp, self.scales,
                                      self.n, self.gs, self.groups, self.world, stream)
            self.reduce = cute.compile(_switch_reduce, self.mc_in, self.mc_out, self.gs,
                                       self.groups, self.rank, self.world, stream)
        else:
            self.reduce = cute.compile(_sm_reduce, self.qp, self.sp, self.r, self.scales,
                                       self.gs, self.groups, self.rank, self.world, stream)
        self.decode = cute.compile(_decode, self.rp, self.op, yc, self.n, self.gs,
                                   self.groups, self.world, self.rank, backend == "switch", stream)
        self.metadata = dict(backend=backend, allocator_backend=self.allocator_backend,
                             world_size=self.world, group_size=self.gs,
                             padded_numel=self.groups * self.gs, input_numel=self.n,
                             quantized_input_bytes_per_rank=self.groups * self.gs,
                             scale_bytes_per_rank=self.groups * 4,
                             switch_accumulator="fp16" if backend == "switch" else None,
                             sm_accumulator="fp32" if backend == "sm" else None,
                             status="experimental", barriers_per_call=4 if backend == "switch" else 3,
                             workspace_shared=workspace_shared, workspace_max_numel=workspace.max_numel,
                             workspace_symmetric_allocation_count=len(self.tensors),
                             workspace_symmetric_bytes_per_rank=workspace.metadata["symmetric_bytes_per_rank"],
                             symmetric_view_bytes_per_rank=sum(t.numel() * t.element_size() for t in self.views),
                             owned_output_bytes_per_rank=self.out.numel() * self.out.element_size())

    def __call__(self, x, out=None):
        out = self.out if out is None else out
        for t in (x, out):
            if t.shape != self.shape or t.device != self.device or t.dtype != torch.bfloat16 or not t.is_contiguous():
                raise ValueError("input/output metadata must match constructor input")
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.local(_cute(x.view(-1)), self.q, self.stats, stream)
        self.sync(self.flags, stream)
        if self.backend == "switch":
            self.quant(_cute(x.view(-1)), self.q, self.sp, self.scales, stream)
            self.sync(self.flags, stream)
            self.reduce(self.mc_in, self.mc_out, stream)
        else:
            self.reduce(self.qp, self.sp, self.r, self.scales, stream)
        self.sync(self.flags, stream)
        self.decode(self.rp, self.op, _cute(out.view(-1)), stream)
        # This last barrier protects buffers from the next invocation/replay.
        self.sync(self.flags, stream)
        return out
