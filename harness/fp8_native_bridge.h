// C ABI for the companion FP8/FP4/BF16 benchmark. No PyTorch dependency.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Every int return is 0 on success and -1 on failure. last_error is thread-local
// and valid until the next bridge call. CUDA execution failures are asynchronous.
const char* fp8_bridge_last_error(void);
size_t fp8_bridge_uid_size(void);
int fp8_bridge_get_unique_id(void* uid);
int fp8_bridge_nccl_version(int* version);

// Collective across world independent processes. The caller sets its current
// CUDA device before using the bridge; create also sets device explicitly.
// max_rows > 0; hidden is fixed at 6144. variant is "L2", "L4", or "none".
// L4 requires world=4; L2 accepts world=2 or 4. All ranks use identical settings.
// uid points at uid_size() host bytes from get_unique_id. *out is null on error.
// Setup allocates the native window, reference scratch and alignment scalar.
// Call outside CUDA graph capture. The handle owns its NCCL communicator.
int fp8_bridge_create(void** out, int rank, int world, int device, int max_rows,
                      const char* variant, const void* uid);

// All stream arguments are the cudaStream_t value (e.g. torch's cuda_stream).
// Launches enqueue work without host synchronization or allocation. Input and
// output are distinct contiguous CUDA BF16 buffers [rows,6144], retained until
// stream/graph work completes. algo uses CLI values: 1=one-shot, 2=two-shot.
// tune4 may be null for defaults. FP4 includes encode, transport, and decode.
int fp8_bridge_launch_fp4(void* handle, const void* x, void* out, int rows,
                          int algo, const int* tune4, void* stream);
int fp8_bridge_allreduce_bf16(void* handle, const void* x, void* out, int rows,
                             void* stream);
// One FP32 NCCL allreduce, to align ranks BEFORE the start timing event.
int fp8_bridge_align(void* handle, void* stream);

// Exact existing harness input kernels, including 32 outliers. src is [0,world).
// Existing harness seed for input set s is 1000u + 17u*rows + 7777u*s.
int fp8_bridge_generate(void* handle, void* x, int src, int rows, uint32_t seed,
                        void* stream);
// all_sources is contiguous BF16 [world,rows,6144], with no max_rows padding.
// codec_out is BF16 [rows,6144], fp32_out is FP32 [rows,6144]. Either output may
// be null to skip that reference. Reference scratch is serial per handle.
int fp8_bridge_reference(void* handle, const void* all_sources, void* codec_out,
                         float* fp32_out, int rows, int algo, void* stream);

// Optional communicator registration, outside capture/timing; one pair/handle.
// Successful registration does not itself guarantee an NVLS algorithm is used.
// The caller owns the storage and must keep it alive until deregister/destroy.
int fp8_bridge_register_nccl_buffers(void* handle, void* input, void* output,
                                     size_t bytes);
int fp8_bridge_unregister_nccl_buffers(void* handle);
// Synchronizes the current device, deregisters/frees native resources, and
// destroys the communicator. Call collectively after destroying captured graphs
// referencing this handle. No Python/C++ exceptions cross the C ABI.
int fp8_bridge_destroy(void* handle);

#ifdef __cplusplus
}
#endif
