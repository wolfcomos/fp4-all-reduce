"""NumPy precision models for the BF16 -> FP8 communication prototype.

These models describe numerical contracts, not communication or synchronization.
The switch model uses one *illustrative* rank-order FP16 accumulation. It is not a
bit-exact emulator of multimem.ld_reduce: hardware reduction order, intermediate
rounding, and overflow behavior require GPU validation. Its final FP8 conversion
uses finite saturation; fp8_range_exceeded separately records out-of-range sums.
"""

from dataclasses import dataclass

import numpy as np


FP8_MAX = np.float32(448.0)
SCALE_FLOOR = np.float32(2.0**-120)


def fp8_e4m3fn_decode(bits):
    """Decode E4M3FN bytes to float32, including signed zero and NaN."""
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> 3) & 15).astype(np.int32)
    mantissa = (bits & 7).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    magnitude = np.where(exponent == 0, mantissa * np.float32(2.0**-9), normal)
    magnitude = np.where((bits & 127) == 127, np.float32(np.nan), magnitude)
    return np.copysign(magnitude, np.where((bits & 128) != 0, -1.0, 1.0)).astype(
        np.float32
    )


_POSITIVE_FP8 = fp8_e4m3fn_decode(np.arange(127, dtype=np.uint8))


def fp8_e4m3fn_encode(values):
    """Encode float32 with round-to-nearest-even and finite saturation.

    Finite overflow and infinities saturate to +/-448. NaNs stay NaNs. This is
    specifically a conversion model, not a claim about FP8 arithmetic overflow.
    """
    values = np.asarray(values, dtype=np.float32)
    magnitude = np.abs(values)
    magnitude = np.where(np.isnan(magnitude), np.float32(0), magnitude)
    magnitude = np.minimum(magnitude, FP8_MAX)
    upper = np.minimum(np.searchsorted(_POSITIVE_FP8, magnitude, side="left"), 126)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitude - _POSITIVE_FP8[lower]
    upper_distance = _POSITIVE_FP8[upper] - magnitude
    # Adjacent finite encodings have alternating significand least-significant
    # bits, including at exponent boundaries and at the subnormal boundary.
    use_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    encoded = np.where(use_upper, upper, lower).astype(np.uint8)
    encoded = np.where(np.isnan(values), np.uint8(127), encoded).astype(np.uint8)
    return encoded | (np.signbit(values).astype(np.uint8) << 7)


def fp8_e4m3fn_round(values):
    """Return the decoded values of finite-saturating E4M3FN RNE conversion."""
    return fp8_e4m3fn_decode(fp8_e4m3fn_encode(values))


def bf16_round(values):
    """Round float32 to BF16 with ties-to-even, returning float32 storage."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(
        0xFFFF0000
    )
    # Preserve NaNs, including payloads whose high BF16 payload bits were zero.
    rounded = np.where(
        np.isnan(values), (bits & np.uint32(0xFFFF0000)) | np.uint32(0x00400000), rounded
    ).astype(np.uint32)
    return rounded.view(np.float32)


def _prepare_inputs(inputs, group_size):
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] < 1:
        raise ValueError("inputs must have shape [world_size >= 1, numel >= 1]")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("only finite BF16 inputs are supported")
    inputs = bf16_round(inputs)
    if not np.all(np.isfinite(inputs)):
        raise ValueError("input overflowed during BF16 conversion")
    world_size, numel = inputs.shape
    padded_numel = ((numel + group_size - 1) // group_size) * group_size
    padded = np.pad(inputs, ((0, 0), (0, padded_numel - numel)))
    return inputs, padded.reshape(world_size, -1, group_size)


def _scales(amax, headroom=1):
    # Divide first to avoid an unnecessary intermediate overflow in amax * R.
    scale = np.maximum((amax / FP8_MAX) * np.float32(headroom), SCALE_FLOOR)
    if not np.all(np.isfinite(scale)):
        raise ValueError("scale overflow: input range/world size is unsupported")
    return scale.astype(np.float32)


@dataclass
class TwoShotResult:
    output: np.ndarray
    input_scales: np.ndarray
    quantized_inputs: np.ndarray
    reduced_fp32: np.ndarray
    output_scales: np.ndarray
    quantized_output: np.ndarray


def two_shot_reference(inputs, group_size=128):
    """Independent rank scales, FP32 local sum, then FP8 redistribution.

    Inputs have shape [world_size, numel] and are rounded to BF16 first. Groups
    pad their tails with zero. Every rank, including the owning rank, receives
    the same second-stage quantize/dequantize result before BF16 conversion.
    Scales are FP32. Returned FP8 arrays contain decoded float32 values.
    """
    inputs, blocks = _prepare_inputs(inputs, group_size)
    world_size, numel = inputs.shape
    scales = _scales(np.max(np.abs(blocks), axis=2))
    quantized = fp8_e4m3fn_round(blocks / scales[:, :, None])
    reduced = np.zeros(blocks.shape[1:], dtype=np.float32)
    for rank in range(world_size):
        reduced = np.add(reduced, quantized[rank] * scales[rank, :, None], dtype=np.float32)
    if not np.all(np.isfinite(reduced)):
        raise ValueError("FP32 reduction overflow: reference requires finite sums")
    output_scales = _scales(np.max(np.abs(reduced), axis=1))
    quantized_output = fp8_e4m3fn_round(reduced / output_scales[:, None])
    output = bf16_round(quantized_output * output_scales[:, None]).reshape(-1)[:numel]
    return TwoShotResult(
        output=np.broadcast_to(output, inputs.shape).copy(),
        input_scales=scales,
        quantized_inputs=quantized.reshape(world_size, -1)[:, :numel],
        reduced_fp32=reduced.reshape(-1)[:numel],
        output_scales=output_scales,
        quantized_output=quantized_output.reshape(-1)[:numel],
    )


@dataclass
class SwitchResult:
    output: np.ndarray
    shared_scales: np.ndarray
    headroom: int
    quantized_inputs: np.ndarray
    accumulated_fp16: np.ndarray
    quantized_sum: np.ndarray
    fp8_range_exceeded: np.ndarray


def switch_reference(inputs, group_size=128, headroom=True):
    """Model common-scale FP8 sum with an illustrative FP16 accumulator.

    scale[group] = (max_rank_amax[group] / 448) * next_power_of_two(world_size),
    clamped to SCALE_FLOOR. Normalized values are explicitly clamped to
    +/-448/headroom, matching the GPU boundary protection. For the prototype's
    2/4/8 ranks these bounds are exactly representable in both FP8 and FP16.
    Any FP16 addition tree is range-safe: the bound for every k-leaf subtree,
    k * 448/headroom, is exactly representable in FP16, so monotone rounding
    cannot cross it. This assumes no FP8 requantization of internal subtrees.
    Actual parallel reduction order and rounding still need GPU validation.
    Set headroom=False only to expose the overflow failure mode. Computing
    the shared maximum requires communication; this model does not time it.

    Power-of-two headroom shifts exponents without removing FP8 mantissa bits
    while values stay normal. It can increase subnormal/underflow exposure;
    sharing scales also couples ranks with very different magnitudes.

    The accumulator is rounded to FP16 after each rank in input order. The
    FP8 result uses finite saturation. fp8_range_exceeded marks where that
    conversion sees magnitude > 448, regardless of hardware overflow rules.
    """
    inputs, blocks = _prepare_inputs(inputs, group_size)
    world_size, numel = inputs.shape
    factor = 1 << (world_size - 1).bit_length() if headroom else 1
    scales = _scales(np.max(np.abs(blocks), axis=(0, 2)), factor)
    limit = FP8_MAX / np.float32(factor)
    normalized = np.clip(blocks / scales[None, :, None], -limit, limit)
    quantized = fp8_e4m3fn_round(normalized)
    accumulated = np.zeros(blocks.shape[1:], dtype=np.float16)
    with np.errstate(over="ignore"):
        for rank in range(world_size):
            accumulated = np.add(accumulated, quantized[rank].astype(np.float16), dtype=np.float16)
    accumulated = accumulated.astype(np.float32)
    outside_fp8_range = np.abs(accumulated) > FP8_MAX
    quantized_sum = fp8_e4m3fn_round(accumulated)
    output = bf16_round(quantized_sum * scales[:, None]).reshape(-1)[:numel]
    return SwitchResult(
        output=np.broadcast_to(output, inputs.shape).copy(),
        shared_scales=scales,
        headroom=factor,
        quantized_inputs=quantized.reshape(world_size, -1)[:, :numel],
        accumulated_fp16=accumulated.reshape(-1)[:numel],
        quantized_sum=quantized_sum.reshape(-1)[:numel],
        fp8_range_exceeded=outside_fp8_range.reshape(-1)[:numel],
    )
