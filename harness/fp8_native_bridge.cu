// Companion benchmark only. Original kernels and Fp4ArVariant ABI stay unchanged.
#include "fp8_native_bridge.h"
#include "fp4ar_api.h"
#include "ref.cuh"
#include <nccl_device/core.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>

#if NCCL_MAJOR < 2 || (NCCL_MAJOR == 2 && NCCL_MINOR < 31)
#error "fp8_native_bridge requires NCCL >= 2.31 headers"
#endif

const Fp4ArVariant* fp4ar_register_L2();
const Fp4ArVariant* fp4ar_register_L4();

namespace {
thread_local char error_text[1024] = {};
constexpr int kHidden = 6144;
struct Bridge {
  Fp4ArCtx ctx{};
  const Fp4ArVariant* variant = nullptr;
  int max_rows = 0;
  void* window_base = nullptr;
  float* alignment = nullptr;
  char* reference_scratch = nullptr;
  void* registered_input = nullptr;
  void* registered_output = nullptr;
};

int fail(const char* where, const char* message) {
  std::snprintf(error_text, sizeof(error_text), "%s: %s", where, message);
  return -1;
}
int cuda_status(cudaError_t rc, const char* where) {
  return rc == cudaSuccess ? 0 : fail(where, cudaGetErrorString(rc));
}
int nccl_status(ncclResult_t rc, const char* where) {
  return rc == ncclSuccess ? 0 : fail(where, ncclGetErrorString(rc));
}
#define CUDA_OK(expr) do { if (cuda_status((expr), #expr)) return -1; } while (0)
#define NCCL_OK(expr) do { if (nccl_status((expr), #expr)) return -1; } while (0)
#define ABI_BEGIN try { error_text[0] = '\0';
#define ABI_END } catch (const std::exception& e) { return fail(__func__, e.what()); } \
                  catch (...) { return fail(__func__, "unknown C++ exception"); }

Bridge* checked(void* handle, int rows = 1) {
  Bridge* b = static_cast<Bridge*>(handle);
  if (!b) { fail("handle", "null handle"); return nullptr; }
  if (rows <= 0 || rows > b->max_rows) {
    fail("rows", "rows must be positive and <= max_rows"); return nullptr;
  }
  return b;
}
bool valid_algo(int algo) { return algo == 1 || algo == 2; }

// The original reference packs ranks at R * 3460 bytes. Its uint4 value
// accesses require 16-byte source bases, so that layout only works for R % 4
// == 0. Pad between ranks for the companion's small-row cases; the contents
// and planar offsets inside each ref::Slot remain unchanged.
size_t reference_stride(int rows) {
  return (ref::slot_bytes(rows) + size_t{15}) & ~size_t{15};
}

// Same codec operations and rank order as ref::ref_sum_pair; only the
// inter-source address stride differs. Keep reference kernels outside timing.
__device__ __forceinline__ void reference_sum_pair(char* scratch, size_t stride,
                                                  int n, int rows, int row,
                                                  int pair, float* acc) {
  fp4c::zero32(acc);
  for (int src = 0; src < n; ++src) {
    ref::Slot slot{scratch + size_t(src) * stride, rows};
    fp4c::dec_acc(acc, slot.q(row)[pair], slot.sf(row)[pair], slot.rs()[row]);
  }
}

__global__ void reference_oneshot(char* scratch, size_t stride, int n, int rows,
                                  __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
       row < rows; row += gridDim.x * (blockDim.x >> 5)) {
    for (int j = 0; j < 6; ++j) {
      const int pair = lane + 32 * j;
      float acc[32];
      reference_sum_pair(scratch, stride, n, rows, row, pair, acc);
      fp4c::store_out(out + size_t(row) * fp4c::H + 32 * pair, acc);
    }
  }
}

__global__ void reference_twoshot(char* scratch, size_t stride, int n, int rows,
                                  __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  for (int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
       row < rows; row += gridDim.x * (blockDim.x >> 5)) {
    float am = 0.f, acc[32];
    for (int j = 0; j < 6; ++j) {
      reference_sum_pair(scratch, stride, n, rows, row, lane + 32 * j, acc);
      am = fp4c::amax32(acc, am);
    }
    float inv, rs2;
    fp4c::row_scales(fp4c::warp_amax(am), inv, rs2);
    for (int j = 0; j < 6; ++j) {
      const int pair = lane + 32 * j;
      reference_sum_pair(scratch, stride, n, rows, row, pair, acc);
      uint4 q;
      uint16_t sf;
      fp4c::enc_pair_f(acc, inv, q, sf);
      fp4c::zero32(acc);
      fp4c::dec_acc(acc, q, sf, rs2);
      fp4c::store_out(out + size_t(row) * fp4c::H + 32 * pair, acc);
    }
  }
}

// Cleanup also supports partially constructed handles. Preserve the first error.
int release(Bridge* b, bool abort_comm) {
  if (!b) return 0;
  int result = 0;
  auto cu = [&](cudaError_t e, const char* where) {
    if (e != cudaSuccess && !result) { result = cuda_status(e, where); }
  };
  auto nc = [&](ncclResult_t e, const char* where) {
    if (e != ncclSuccess && !result) { result = nccl_status(e, where); }
  };
  cu(cudaSetDevice(b->ctx.device), "destroy cudaSetDevice");
  // A peer can fail during collective setup. Abort before any CUDA free or
  // deregistration that could wait on its unfinished communicator work.
  if (abort_comm && b->ctx.comm) {
    nc(ncclCommAbort(b->ctx.comm), "communicator abort");
    b->ctx.comm = nullptr;
  }
  if (!abort_comm) {
    cu(cudaDeviceSynchronize(), "destroy cudaDeviceSynchronize");
    if (b->registered_input) nc(ncclCommDeregister(b->ctx.comm, b->registered_input), "input deregister");
    if (b->registered_output) nc(ncclCommDeregister(b->ctx.comm, b->registered_output), "output deregister");
  }
  if (b->variant && b->ctx.priv) b->variant->fini(&b->ctx);
  // CommAbort releases communicator-owned registration state; never pass its
  // invalidated communicator/window handles to explicit deregistration calls.
  if (!abort_comm && b->ctx.win) nc(ncclCommWindowDeregister(b->ctx.comm, b->ctx.win), "window deregister");
  if (b->window_base) nc(ncclMemFree(b->window_base), "window free");
  if (b->reference_scratch) cu(cudaFree(b->reference_scratch), "reference scratch free");
  if (b->alignment) cu(cudaFree(b->alignment), "alignment free");
  if (b->ctx.comm) nc(ncclCommDestroy(b->ctx.comm), "communicator cleanup");
  return result;
}
struct PendingBridge {
  Bridge* ptr;
  ~PendingBridge() {
    if (!ptr) return;
    char saved[sizeof(error_text)]; std::memcpy(saved, error_text, sizeof(saved));
    release(ptr, true);
    delete ptr;
    std::memcpy(error_text, saved, sizeof(saved));
  }
};
}  // namespace

extern "C" const char* fp8_bridge_last_error() { return error_text; }
extern "C" size_t fp8_bridge_uid_size() { return sizeof(ncclUniqueId); }
extern "C" int fp8_bridge_get_unique_id(void* uid) {
  ABI_BEGIN
  if (!uid) return fail(__func__, "null uid");
  ncclUniqueId id;
  NCCL_OK(ncclGetUniqueId(&id));
  std::memcpy(uid, &id, sizeof(id));
  return 0;
  ABI_END
}
extern "C" int fp8_bridge_nccl_version(int* version) {
  ABI_BEGIN
  if (!version) return fail(__func__, "null version output");
  NCCL_OK(ncclGetVersion(version));
  return 0;
  ABI_END
}
extern "C" int fp8_bridge_create(void** out, int rank, int world, int device,
                                 int max_rows, const char* variant, const void* uid) {
  ABI_BEGIN
  if (!out) return fail(__func__, "null output handle");
  *out = nullptr;
  if (!uid || !variant || (world != 2 && world != 4) || rank < 0 || rank >= world ||
      device < 0 || max_rows <= 0 || max_rows > std::numeric_limits<int>::max() / kHidden)
    return fail(__func__, "invalid uid/variant/rank/world/device/max_rows");
  const Fp4ArVariant* v = nullptr;
  if (!std::strcmp(variant, "L2")) v = fp4ar_register_L2();
  else if (!std::strcmp(variant, "L4")) {
    if (world != 4) return fail(__func__, "L4 requires world=4");
    v = fp4ar_register_L4();
  } else if (std::strcmp(variant, "none")) return fail(__func__, "variant must be L2, L4, or none");
  int version = 0;
  NCCL_OK(ncclGetVersion(&version));
  if (version < 23100) return fail(__func__, "NCCL >=2.31 is required");
  CUDA_OK(cudaSetDevice(device));
  Bridge* b = new Bridge{};
  PendingBridge pending{b};
  b->max_rows = max_rows; b->variant = v;
  b->ctx.rank = rank; b->ctx.nranks = world; b->ctx.device = device;
  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop, device));
  b->ctx.sm_count = prop.multiProcessorCount;
  ncclUniqueId id; std::memcpy(&id, uid, sizeof(id));
  NCCL_OK(ncclCommInitRank(&b->ctx.comm, world, id, rank));
  CUDA_OK(cudaMalloc(&b->alignment, sizeof(float)));
  CUDA_OK(cudaMemset(b->alignment, 0, sizeof(float)));
  CUDA_OK(cudaMalloc(&b->reference_scratch, size_t(world) * reference_stride(max_rows)));
  if (v) {
    b->ctx.win_bytes = v->window_bytes(max_rows, kHidden, world);
    if (!b->ctx.win_bytes) return fail(__func__, "variant rejected window dimensions");
    constexpr size_t page = size_t{1} << 21;
    const size_t allocated = (b->ctx.win_bytes + page - 1) / page * page;
    NCCL_OK(ncclMemAlloc(&b->window_base, allocated));
    CUDA_OK(cudaMemset(b->window_base, 0, allocated));
    CUDA_OK(cudaDeviceSynchronize());
    NCCL_OK(ncclCommWindowRegister(b->ctx.comm, b->window_base, allocated,
                                 &b->ctx.win, NCCL_WIN_COLL_SYMMETRIC));
    const ncclTeam_t lsa = ncclTeamLsa(b->ctx.comm);
    if (lsa.nRanks != world) return fail(__func__, "all ranks must be in one LSA team");
    for (int r = 0; r < world; ++r) {
      void* ptr = nullptr;
      NCCL_OK(ncclGetLsaDevicePointer(b->ctx.win, 0, r, &ptr));
      b->ctx.peer[r] = static_cast<char*>(ptr);
    }
    const int rc = v->init(&b->ctx, max_rows, kHidden);
    if (rc) { char msg[96]; std::snprintf(msg, sizeof(msg), "variant init returned %d", rc); return fail(__func__, msg); }
  }
  // Also makes construction collective for variant=none and establishes a ready
  // input value for the alignment allreduce before later CUDA graph captures.
  NCCL_OK(ncclAllReduce(b->alignment, b->alignment, 1, ncclFloat, ncclSum, b->ctx.comm, nullptr));
  CUDA_OK(cudaDeviceSynchronize());
  *out = b; pending.ptr = nullptr;
  return 0;
  ABI_END
}

extern "C" int fp8_bridge_launch_fp4(void* handle, const void* x, void* out,
                                     int rows, int algo, const int* tune4, void* stream) {
  ABI_BEGIN
  Bridge* b = checked(handle, rows); if (!b) return -1;
  if (!b->variant || !x || !out || x == out || !valid_algo(algo))
    return fail(__func__, "requires variant, distinct nonnull x/out, algo 1 or 2");
  if (algo == 2 && !std::strcmp(b->variant->name, "L4") && rows % 4)
    return fail(__func__, "L4 two-shot requires rows divisible by 4");
  Fp4ArArgs a{};
  a.x = static_cast<const __nv_bfloat16*>(x); a.out = static_cast<__nv_bfloat16*>(out);
  a.rows = rows; a.hidden = kHidden; a.mode = FP4AR_FUSED;
  a.algo = algo == 1 ? FP4AR_ONESHOT : FP4AR_TWOSHOT;
  if (tune4) for (int i = 0; i < 4; ++i) a.tune[i] = tune4[i];
  const int rc = b->variant->launch(&b->ctx, &a, static_cast<cudaStream_t>(stream));
  if (rc) { char msg[96]; std::snprintf(msg, sizeof(msg), "variant launch returned %d", rc); return fail(__func__, msg); }
  return 0;
  ABI_END
}
extern "C" int fp8_bridge_allreduce_bf16(void* handle, const void* x, void* out,
                                        int rows, void* stream) {
  ABI_BEGIN
  Bridge* b = checked(handle, rows); if (!b) return -1;
  if (!x || !out || x == out) return fail(__func__, "requires distinct nonnull input/output");
  NCCL_OK(ncclAllReduce(x, out, size_t(rows) * kHidden, ncclBfloat16, ncclSum,
                       b->ctx.comm, static_cast<cudaStream_t>(stream)));
  return 0;
  ABI_END
}
extern "C" int fp8_bridge_align(void* handle, void* stream) {
  ABI_BEGIN
  Bridge* b = checked(handle); if (!b) return -1;
  NCCL_OK(ncclAllReduce(b->alignment, b->alignment, 1, ncclFloat, ncclSum,
                       b->ctx.comm, static_cast<cudaStream_t>(stream)));
  return 0;
  ABI_END
}
extern "C" int fp8_bridge_generate(void* handle, void* x, int src, int rows,
                                   uint32_t seed, void* stream) {
  ABI_BEGIN
  Bridge* b = checked(handle, rows); if (!b) return -1;
  if (!x || src < 0 || src >= b->ctx.nranks) return fail(__func__, "invalid input/source");
  auto* bf = static_cast<__nv_bfloat16*>(x); auto st = static_cast<cudaStream_t>(stream);
  ref::gen_input<<<1024, 256, 0, st>>>(bf, src, rows, seed);
  CUDA_OK(cudaGetLastError());
  ref::gen_outliers<<<1, 32, 0, st>>>(bf, src, rows, seed);
  CUDA_OK(cudaGetLastError());
  return 0;
  ABI_END
}
extern "C" int fp8_bridge_reference(void* handle, const void* all_sources,
                                    void* codec_out, float* fp32_out, int rows,
                                    int algo, void* stream) {
  ABI_BEGIN
  Bridge* b = checked(handle, rows); if (!b) return -1;
  if (!all_sources || (!codec_out && !fp32_out) || !valid_algo(algo))
    return fail(__func__, "requires source/output and algo 1 or 2");
  const auto* x = static_cast<const __nv_bfloat16*>(all_sources);
  const size_t n = size_t(rows) * kHidden; auto st = static_cast<cudaStream_t>(stream);
  if (codec_out) {
    const size_t stride = reference_stride(rows);
    for (int src = 0; src < b->ctx.nranks; ++src) {
      ref::ref_encode<<<2048, 256, 0, st>>>(x + size_t(src) * n,
          b->reference_scratch + size_t(src) * stride, rows);
      CUDA_OK(cudaGetLastError());
    }
    auto* out = static_cast<__nv_bfloat16*>(codec_out);
    if (algo == 1) reference_oneshot<<<2048, 256, 0, st>>>(b->reference_scratch, stride, b->ctx.nranks, rows, out);
    else reference_twoshot<<<2048, 256, 0, st>>>(b->reference_scratch, stride, b->ctx.nranks, rows, out);
    CUDA_OK(cudaGetLastError());
  }
  if (fp32_out) {
    ref::ref_fp32<<<2048, 256, 0, st>>>(x, b->ctx.nranks, n, fp32_out);
    CUDA_OK(cudaGetLastError());
  }
  return 0;
  ABI_END
}
extern "C" int fp8_bridge_register_nccl_buffers(void* handle, void* input,
                                                void* output, size_t bytes) {
  ABI_BEGIN
  Bridge* b = checked(handle); if (!b) return -1;
  if (!input || !output || input == output || !bytes || b->registered_input || b->registered_output)
    return fail(__func__, "requires fresh, distinct nonnull buffers and positive bytes");
  NCCL_OK(ncclCommRegister(b->ctx.comm, input, bytes, &b->registered_input));
  const ncclResult_t rc = ncclCommRegister(b->ctx.comm, output, bytes, &b->registered_output);
  if (rc != ncclSuccess) {
    ncclCommDeregister(b->ctx.comm, b->registered_input); b->registered_input = nullptr;
    return nccl_status(rc, "output ncclCommRegister");
  }
  return 0;
  ABI_END
}
extern "C" int fp8_bridge_unregister_nccl_buffers(void* handle) {
  ABI_BEGIN
  Bridge* b = checked(handle); if (!b) return -1;
  if (b->registered_input) { NCCL_OK(ncclCommDeregister(b->ctx.comm, b->registered_input)); b->registered_input = nullptr; }
  if (b->registered_output) { NCCL_OK(ncclCommDeregister(b->ctx.comm, b->registered_output)); b->registered_output = nullptr; }
  return 0;
  ABI_END
}
extern "C" int fp8_bridge_destroy(void* handle) {
  ABI_BEGIN
  if (!handle) return 0;
  std::unique_ptr<Bridge> b(static_cast<Bridge*>(handle));
  return release(b.get(), false);
  ABI_END
}
