#!/usr/bin/env bash
# Build the companion C ABI with the original L2/L4 kernels, without PyTorch.
# Usage: NCCL_INC=/nccl/include NCCL_LIB=/nccl/lib [NVCC=...] [BUILD_DIR=...]
#        [FP8_BRIDGE_LIB=/output/libfp8_native_bridge.so]
#        scripts/build_fp8_bridge.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${NCCL_INC:?Set NCCL_INC to NCCL >= 2.31 headers}"
: "${NCCL_LIB:?Set NCCL_LIB to the directory containing matching libnccl.so.2}"
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-100a}
OUT=${BUILD_DIR:-$ROOT/build/fp8_bridge}
LIBRARY=${FP8_BRIDGE_LIB:-$OUT/libfp8_native_bridge.so}
mkdir -p "$OUT" "$(dirname "$LIBRARY")"
[[ -f "$NCCL_INC/nccl.h" ]] || { echo "Missing $NCCL_INC/nccl.h" >&2; exit 1; }
if [[ -f "$NCCL_LIB/libnccl.so" ]]; then
  NCCL_LINK=-lnccl
elif [[ -f "$NCCL_LIB/libnccl.so.2" ]]; then
  NCCL_LINK=-l:libnccl.so.2
else
  echo "Missing libnccl.so or libnccl.so.2 in $NCCL_LIB" >&2
  exit 1
fi
FLAGS=(-O3 -std=c++17 -gencode "arch=compute_$ARCH,code=sm_$ARCH" -lineinfo
       -Xcompiler -fPIC -I"$ROOT/include" -I"$ROOT/harness" -I"$NCCL_INC")
"$NVCC" --version > "$OUT/toolchain.txt"
"$NVCC" "${FLAGS[@]}" -DFP4AR_NAME=L2 -DFP4AR_REGISTER=fp4ar_register_L2 -DLAMPORT_V2=1 \
  -c "$ROOT/kernels/push/tp2_l2.cu" -o "$OUT/L2.o" > "$OUT/L2.log" 2>&1 || { cat "$OUT/L2.log" >&2; exit 1; }
"$NVCC" "${FLAGS[@]}" -DFP4AR_NAME=L4 -DFP4AR_REGISTER=fp4ar_register_L4 \
  -c "$ROOT/kernels/push/tp4_l4.cu" -o "$OUT/L4.o" > "$OUT/L4.log" 2>&1 || { cat "$OUT/L4.log" >&2; exit 1; }
"$NVCC" "${FLAGS[@]}" -c "$ROOT/harness/fp8_native_bridge.cu" -o "$OUT/fp8_native_bridge.o" \
  > "$OUT/bridge.log" 2>&1 || { cat "$OUT/bridge.log" >&2; exit 1; }
"$NVCC" -shared --cudart shared -o "$LIBRARY" "$OUT/fp8_native_bridge.o" "$OUT/L2.o" "$OUT/L4.o" \
  -L"$NCCL_LIB" "$NCCL_LINK" -Xlinker -rpath -Xlinker "$NCCL_LIB" \
  -Xlinker --no-undefined \
  > "$OUT/link.log" 2>&1 || { cat "$OUT/link.log" >&2; exit 1; }
echo "$LIBRARY"
