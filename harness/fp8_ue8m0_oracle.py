"""Independent NumPy oracle for a group-128 E4M3/UE8M0 TP2 codec.

This is deliberately not standard MXFP8 (whose groups have 32 elements).
Inputs and outputs are BF16 values held in NumPy float32 arrays. Scale selection
uses float64 frexp and exact mathematical thresholds, not the kernel's FP32 bit
algorithm. Format-conversion primitives are shared with the existing CPU oracle.
Finite inputs are accepted even when quantization/dequantization or addition
overflows; those failures remain explicit and fail the finite-output contract.
"""

import numpy as np

try:
    from .cute_fp8.reference import bf16_round, fp8_e4m3fn_decode, fp8_e4m3fn_encode
except ImportError:
    from cute_fp8.reference import bf16_round, fp8_e4m3fn_decode, fp8_e4m3fn_encode


UE8M0_CONTRACT = "tp2_fp8_ue8m0_group128_oneshot_v1"
UE8M0_CONTRACT_METADATA = {
    "tp_size": 2,
    "group_size": 128,
    "standard_mxfp8_group32": False,
    "input_output_dtype": "bfloat16",
    "quantized_dtype": "float8_e4m3fn",
    "scale_dtype": "ue8m0",
    "scale_rule": "e=max(-120,ceil(log2(amax/448))); scale=2**e; byte=e+127",
    "zero_group_scale_byte": 7,
    "scale_floor_exponent": -120,
    "normalization": "FP32 mul.rn with exact reciprocal 2**(-e), then E4M3 RNE",
    "accumulation": "separate FP32 mul.rn/add.rn; +0 seed, source rank0 then rank1",
    "output_rounding": "BF16 RNE",
    "second_output_quantization": False,
    "bitwise_reference_required": True,
    "finite_output_required": True,
    "finite_input_does_not_guarantee_finite_output": True,
    "model_accuracy_qualified": False,
}


def ue8m0_decode_scale_codes(codes):
    """Decode UE8M0 bytes: 0..254 represent 2**(byte-127); 255 is NaN."""
    codes = np.asarray(codes)
    if codes.dtype != np.uint8:
        raise ValueError("UE8M0 codes must have uint8 dtype")
    exponent = codes.astype(np.int32) - 127
    with np.errstate(over="ignore"):
        values = np.ldexp(np.ones(codes.shape, dtype=np.float32), exponent)
    return np.where(codes == 255, np.float32(np.nan), values).astype(np.float32)


def ue8m0_scales(amax):
    """Return exact exponent, byte, scale and reciprocal for nonnegative amax.

    frexp gives amax=m*2**p, .5<=m<1. The smaller candidate e=p-9
    suffices exactly when amax<=448*2**e; otherwise increment e. Both sides
    of this comparison are exactly representable in float64 for BF16 inputs.
    Zero is handled explicitly, and the floor is applied after selection.
    """
    amax = np.asarray(amax, dtype=np.float64)
    if not np.isfinite(amax).all() or (amax < 0).any():
        raise ValueError("amax must be finite and nonnegative")
    if (amax > np.finfo(np.float32).max).any():
        raise ValueError("amax exceeds the finite FP32 input domain")
    _, power = np.frexp(amax)
    candidate = power.astype(np.int32) - 9
    threshold = np.ldexp(np.full(amax.shape, 448.0, dtype=np.float64), candidate)
    candidate += (amax > threshold).astype(np.int32)
    exponent = np.where(amax == 0, -120, np.maximum(candidate, -120)).astype(np.int32)
    if (exponent > 120).any():
        raise ValueError("scale exponent exceeds the finite BF16 domain")
    codes = (exponent + 127).astype(np.uint8)
    scales = ue8m0_decode_scale_codes(codes)
    reciprocal = np.ldexp(np.ones(amax.shape, dtype=np.float32), -exponent)
    return dict(exponents=exponent, codes=codes, scales=scales, reciprocals=reciprocal)


def ue8m0_reference(inputs, group_size=128):
    with np.errstate(over="ignore", invalid="ignore"):
        inputs = np.asarray(inputs, dtype=np.float32)
    if (group_size != 128 or inputs.ndim != 2 or inputs.shape[0] != 2
            or inputs.shape[1] == 0 or inputs.shape[1] % 128):
        raise ValueError("UE8M0 reference requires complete [2,K*128] input groups")
    if not np.isfinite(inputs).all():
        raise ValueError("UE8M0 reference requires finite input")
    blocks = bf16_round(inputs).reshape(2, -1, 128)
    if not np.isfinite(blocks).all():
        raise ValueError("input overflowed during BF16 conversion")
    scale = ue8m0_scales(np.max(np.abs(blocks), axis=2))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = np.multiply(blocks, scale["reciprocals"][:, :, None], dtype=np.float32)
        normalized = np.clip(normalized, np.float32(-448), np.float32(448))
        q = fp8_e4m3fn_encode(normalized)
        decoded = fp8_e4m3fn_decode(q)
        products = np.multiply(decoded, scale["scales"][:, :, None], dtype=np.float32)
        total = np.zeros(blocks.shape[1:], dtype=np.float32)
        for rank in range(2):
            total = np.add(total, products[rank], dtype=np.float32)
        output = bf16_round(total).reshape(-1)
    diagnostics = {
        "normalized_nonfinite_count": int(np.count_nonzero(~np.isfinite(normalized))),
        "dequant_nonfinite_count_per_source": np.count_nonzero(~np.isfinite(products), axis=(1, 2)).tolist(),
        "fp32_reduction_nonfinite_count": int(np.count_nonzero(~np.isfinite(total))),
        "fp32_reduction_nan_count": int(np.count_nonzero(np.isnan(total))),
        "bf16_output_nonfinite_count": int(np.count_nonzero(~np.isfinite(output))),
        "bf16_conversion_overflow_count": int(np.count_nonzero(np.isfinite(total.reshape(-1)) & ~np.isfinite(output))),
    }
    finite = bool(np.isfinite(products).all() and np.isfinite(total).all() and np.isfinite(output).all())
    return dict(output=np.broadcast_to(output, inputs.shape).copy(),
                quantized_inputs=q.reshape(inputs.shape), input_scale_codes=scale["codes"],
                input_scale_exponents=scale["exponents"], input_scales=scale["scales"],
                input_reciprocals=scale["reciprocals"], reduced_fp32=total.reshape(-1),
                reference_finite=finite, overflow_diagnostics=diagnostics)


def _json_number(value):
    value = float(value)
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "+Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return value


def ue8m0_contract(inputs, outputs, group_size=128):
    reference = ue8m0_reference(inputs, group_size)
    outputs = np.asarray(outputs, dtype=np.float32)
    expected = reference["output"]
    if outputs.shape != expected.shape:
        raise ValueError("outputs must match [2,K*128] input shape")
    finite = bool(np.isfinite(outputs).all())
    mismatch = outputs.view(np.uint32) != expected.view(np.uint32)
    failures = np.argwhere(mismatch)
    both_finite = finite and reference["reference_finite"]
    return dict(contract=UE8M0_CONTRACT, passed=both_finite and not bool(mismatch.any()),
                comparison="exact_bf16_bits_in_float32_storage", finite=finite,
                reference_finite=reference["reference_finite"],
                overflow_diagnostics=reference["overflow_diagnostics"],
                mismatch_count_per_tp_rank=mismatch.sum(axis=1).tolist(),
                max_abs_codec_error=float(np.max(np.abs(outputs.astype(np.float64) - expected))) if both_finite else None,
                failure_examples=[dict(tp_rank=int(rank), sample_index=int(index),
                    expected=_json_number(expected[rank, index]), actual=_json_number(outputs[rank, index]),
                    expected_bits=int(expected.view(np.uint32)[rank, index] >> np.uint32(16)),
                    actual_bits=int(outputs.view(np.uint32)[rank, index] >> np.uint32(16)))
                    for rank, index in failures[:16]],
                second_output_quantization=False,
                scope="Complete sampled group128 codec; transport and full-tensor protocol gates are separate.")


def received_payload_contract(reference, received_q, received_scale_codes, received_scales=None):
    """Check gathered receiver-order payloads against the opposite source.

    No FP32 scale is transmitted. By default decoded scales are a CPU decode of
    the received UE8M0 bytes, explicitly not a readback of kernel registers.
    The final BF16 bit gate checks the kernel's decoded-scale arithmetic.
    """
    q, codes = np.asarray(received_q), np.asarray(received_scale_codes)
    expected_q = reference["quantized_inputs"][[1, 0]]
    expected_codes = reference["input_scale_codes"][[1, 0]]
    expected_scales = reference["input_scales"][[1, 0]]
    if q.dtype != np.uint8 or codes.dtype != np.uint8:
        raise ValueError("received q and scale codes must have uint8 dtype")
    if q.shape != expected_q.shape or codes.shape != expected_codes.shape:
        raise ValueError("received payload shapes must match the reference in receiver order")
    if received_scales is None:
        scales = ue8m0_decode_scale_codes(codes)
        scope = "CPU decode of received UE8M0 bytes; actual kernel arithmetic is checked by final BF16 output"
    else:
        scales = np.asarray(received_scales, dtype=np.float32)
        scope = "Caller-supplied decoded scale readback"
    if scales.shape != expected_scales.shape:
        raise ValueError("received decoded scales have the wrong shape")
    q_mismatch, code_mismatch = q != expected_q, codes != expected_codes
    scale_mismatch = scales.view(np.uint32) != expected_scales.view(np.uint32)
    finite = bool(np.isfinite(scales).all() and (scales > 0).all())
    return dict(passed=finite and not bool(q_mismatch.any() or code_mismatch.any() or scale_mismatch.any()),
                source_order_per_receiving_rank=[1, 0],
                quantized_byte_mismatches_per_rank=q_mismatch.sum(axis=1).tolist(),
                scale_code_byte_mismatches_per_rank=code_mismatch.sum(axis=1).tolist(),
                scale_bit_mismatches_per_rank=scale_mismatch.sum(axis=1).tolist(),
                decoded_scale_finite_positive=finite, decoded_scale_scope=scope,
                negative_zero_byte_count_per_rank=(q == 0x80).sum(axis=1).tolist(),
                expected_negative_zero_byte_count_per_rank=(expected_q == 0x80).sum(axis=1).tolist(),
                scale_floor_count_per_rank=(codes == 7).sum(axis=1).tolist(),
                expected_scale_floor_count_per_rank=(expected_codes == 7).sum(axis=1).tolist())
