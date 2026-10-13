"""Independent NumPy oracle for the fused TP2 FP8 one-shot codec.

The CPU model shares format conversion primitives with the existing independent
reference, but does not import or execute a CuTe kernel. Inputs/outputs are BF16
values represented by float32 NumPy arrays. Complete group samples are valid
because neither scale calculation nor reduction crosses a group boundary.
"""
import numpy as np

try:
    from .cute_fp8.reference import bf16_round, fp8_e4m3fn_decode, fp8_e4m3fn_encode
except ImportError:
    from cute_fp8.reference import bf16_round, fp8_e4m3fn_decode, fp8_e4m3fn_encode


ONESHOT_CONTRACT = "tp2_fp8_oneshot_fp32_rank_order_v1"
ONESHOT_CONTRACT_METADATA = {
    "tp_size": 2,
    "group_size": 128,
    "input_output_dtype": "bfloat16",
    "quantized_dtype": "float8_e4m3fn",
    "scales": "FP32 max(abs(group))/448, floored at 2**-120, separately per source",
    "rounding": "FP8 RNE with finite clamp; separate FP32 mul.rn/add.rn; final BF16 RNE",
    "accumulation": "FP32 +0 seed, then source rank 0 and source rank 1",
    "second_output_quantization": False,
    "bitwise_reference_required": True,
    "model_accuracy_qualified": False,
}


def one_shot_reference(inputs, group_size=128):
    inputs = np.asarray(inputs, dtype=np.float32)
    if (group_size != 128 or inputs.ndim != 2 or inputs.shape[0] != 2
            or inputs.shape[1] == 0 or inputs.shape[1] % group_size):
        raise ValueError("one-shot reference requires complete [2,K*128] input groups")
    if not np.isfinite(inputs).all():
        raise ValueError("one-shot reference requires finite input")
    blocks = bf16_round(inputs).reshape(2, -1, group_size)
    scales = np.maximum(np.max(np.abs(blocks), axis=2) / np.float32(448.0),
                        np.float32(2.0 ** -120)).astype(np.float32)
    normalized = np.divide(blocks, scales[:, :, None], dtype=np.float32)
    normalized = np.clip(normalized, np.float32(-448), np.float32(448))
    encoded = fp8_e4m3fn_encode(normalized)
    decoded = fp8_e4m3fn_decode(encoded)
    total = np.zeros(blocks.shape[1:], dtype=np.float32)
    for rank in range(2):
        product = np.multiply(decoded[rank], scales[rank, :, None], dtype=np.float32)
        total = np.add(total, product, dtype=np.float32)
    output = bf16_round(total).reshape(-1)
    return dict(output=np.broadcast_to(output, inputs.shape).copy(),
                quantized_inputs=encoded.reshape(inputs.shape), input_scales=scales,
                reduced_fp32=total.reshape(-1))


def one_shot_contract(inputs, outputs, group_size=128):
    outputs = np.asarray(outputs, dtype=np.float32)
    reference = one_shot_reference(inputs, group_size)
    expected = reference["output"]
    if outputs.shape != expected.shape:
        raise ValueError("outputs must match [2,K*128] input shape")
    finite = bool(np.isfinite(outputs).all())
    mismatch = outputs.view(np.uint32) != expected.view(np.uint32)
    failures = np.argwhere(mismatch)
    return dict(contract=ONESHOT_CONTRACT, passed=finite and not bool(mismatch.any()),
                comparison="exact_bf16_bits_in_float32_storage", finite=finite,
                mismatch_count_per_tp_rank=mismatch.sum(axis=1).tolist(),
                max_abs_codec_error=float(np.max(np.abs(outputs - expected))) if finite else None,
                failure_examples=[dict(tp_rank=int(rank), sample_index=int(index),
                    expected=float(expected[rank, index]), actual=float(outputs[rank, index]),
                    expected_bits=int(expected.view(np.uint32)[rank, index] >> np.uint32(16)),
                    actual_bits=int(outputs.view(np.uint32)[rank, index] >> np.uint32(16)))
                    for rank, index in failures[:16]],
                second_output_quantization=False,
                scope="Complete sampled groups only; full-tensor protocol checks are separate.")
