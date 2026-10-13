"""CPU-only sampled stage checks for the companion TP2 FP8 benchmark.

The numerical gate is extracted from cute-fp8-allreduce-prototype's
ops/workspace_regression.py. It preserves that empirical adjacent-E4M3
contract; it is neither an RNE hardware emulator nor an FP4 bitwise oracle.
GPU gathering, full-output agreement and repeat/graph checks belong to the
driver. Supply complete quantization groups, in identical order on both ranks.
"""

import numpy as np

try:
    from .cute_fp8.reference import (
        bf16_round,
        fp8_e4m3fn_decode,
        fp8_e4m3fn_encode,
        switch_reference,
    )
except ImportError:
    # Support adding harness/ to sys.path and importing fp8_oracle directly.
    from cute_fp8.reference import (
        bf16_round,
        fp8_e4m3fn_decode,
        fp8_e4m3fn_encode,
        switch_reference,
    )


SWITCH_CONTRACT = "tp2_fp8_faithful_interval_empirical_v1"
SWITCH_CONTRACT_METADATA = {
    "changed_from_rne_equivalence": True,
    "weaker_than_rne_equivalence": True,
    "observed_scope": {
        "device": "NVIDIA GB200",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "cutlass_dsl_version": "4.6.2",
        "tp_size": 2,
        "group_size": 128,
    },
    "model_accuracy_qualified": False,
    "batch_invariance_qualified": False,
    "architectural_guarantee": False,
    "scope": (
        "Empirical adjacent-E4M3 bound on this runtime, with strict input "
        "quantization, scale, stats and BF16 decode checks. Neighbor context "
        "can change rounding. Prior RNE failures remain failures under the "
        "previous contract."
    ),
}


def _hash32(value):
    """The uint32 hash in the unmodified harness/ref.cuh."""
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def original_outlier_positions(numel, input_seed):
    """Unique element positions written by ref::gen_outliers for this set.

    The uint32 seed arithmetic wraps before hashing. The subsequent position
    multiplication is 64-bit in CUDA. Positions are shared across source ranks;
    only each rank's outlier amplitude differs.
    """
    if type(numel) is not int or numel <= 0 or type(input_seed) is not int:
        raise ValueError("numel must be positive and input_seed must be an integer")
    return sorted({
        (_hash32(input_seed * 7919 + 104729 * (thread + 1)) * 2654435761) % numel
        for thread in range(32)
    })


def sample_group_indices(numel, *, input_seed, group_size=128, max_groups=128):
    """Return sorted deterministic complete-group indices, at most max_groups.

    Includes head, tail, middle and every original outlier-containing group.
    Additional indices use a fixed hash sequence independent of input values.
    input_seed must match the actual generator invocation; for set s the
    original harness uses 1000 + 17 * rows + 7777 * s, modulo 2**32.
    A budget too small to retain all mandatory groups is rejected.
    """
    if (type(numel) is not int or numel <= 0 or type(group_size) is not int
            or group_size <= 0 or numel % group_size or type(max_groups) is not int
            or not 1 <= max_groups <= 128 or type(input_seed) is not int):
        raise ValueError("requires complete positive groups, integer seed and max_groups in [1,128]")
    groups = numel // group_size
    if groups <= max_groups:
        return list(range(groups))
    selected = {0, 1, groups // 2, groups - 2, groups - 1}
    selected.update(position // group_size for position in original_outlier_positions(numel, input_seed))
    selected = {index for index in selected if 0 <= index < groups}
    if len(selected) > max_groups:
        raise ValueError("max_groups cannot retain every mandatory/outlier group")
    counter = 0
    while len(selected) < max_groups:
        selected.add(_hash32(0xC0DEC0DE + counter) % groups)
        counter += 1
    return sorted(selected)


def rne_diagnostic(actual, expected):
    """Preserve the previous two-BF16-ULP diagnostic without gating switch."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)

    def codes(values):
        bits = values.view(np.uint32) >> 16
        magnitude = (bits & np.uint32(0x7FFF)).astype(np.int64)
        return np.where(bits & np.uint32(0x8000), -magnitude, magnitude)

    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    distance = int(np.abs(codes(actual) - codes(expected)).max()) if finite else None
    return {
        "max_bf16_ulp_distance": distance,
        "max_abs_codec_error": float(np.max(np.abs(actual - expected))) if finite else None,
        "tolerance_bf16_ulps": 2,
        "passed_under_previous_gate": finite and distance <= 2,
    }


def faithful_interval(q_bytes, r_bytes):
    """Check adjacent finite FP8 endpoints of the exact two-input sum.

    Every finite E4M3 value is a multiple of 2**-9, so float32 adds a pair
    exactly. Both zero signs are numerically zero here; byte-level rank,
    repeat, allocation-mode and BF16-decode checks remain separate gates.
    """
    q = fp8_e4m3fn_decode(q_bytes)
    r = fp8_e4m3fn_decode(r_bytes)
    if q.ndim != 2 or q.shape[0] != 2 or r.shape != q.shape or q.shape[1] == 0:
        raise ValueError("interval requires matching nonempty [2, numel] q/r arrays")
    exact = np.add(q[0], q[1], dtype=np.float32)
    values = fp8_e4m3fn_decode(np.arange(256, dtype=np.uint8))
    values = np.unique(values[np.isfinite(values)])
    finite_inputs = bool(np.isfinite(q).all())
    headroom = finite_inputs and bool((np.abs(q) <= 224).all())
    in_range = np.isfinite(exact) & (exact >= values[0]) & (exact <= values[-1])
    upper = values[np.clip(np.searchsorted(values, exact, side="left"), 0, len(values) - 1)]
    lower = values[np.clip(np.searchsorted(values, exact, side="right") - 1, 0, len(values) - 1)]
    within = np.isfinite(r) & in_range[None, :] & (r >= lower) & (r <= upper)
    exact_representable = in_range & (lower == upper)
    changed_exact = exact_representable[None, :] & (r != exact)
    failures = np.argwhere(~within)
    return {
        "passed": headroom and bool(within.all()),
        "finite_inputs": finite_inputs,
        "tp2_input_headroom_preserved": headroom,
        "nonfinite_output_count": int((~np.isfinite(r)).sum()),
        "out_of_range_exact_sum_count": int((~in_range).sum()),
        "outside_adjacent_fp8_bracket_count_per_tp_rank": (~within).sum(axis=1).tolist(),
        "exactly_representable_sum_count": int(exact_representable.sum()),
        "changed_exactly_representable_sum_count_per_tp_rank": changed_exact.sum(axis=1).tolist(),
        "failure_examples": [
            {"tp_rank": int(rank), "sample_index": int(index), "exact_sum": float(exact[index]),
             "lower": float(lower[index]), "upper": float(upper[index]), "actual": float(r[rank, index])}
            for rank, index in failures[:12]
        ],
        "signed_zero_policy": "Numerically equal in interval only; other byte/bit gates remain strict.",
    }


def switch_stage_contract(inputs, q_bytes, r_bytes, stats, scales, outputs, gs=128):
    """Validate host samples: values [2,K*128], stats/scales [2,K].

    Float arrays use float32 storage (BF16 values converted exactly to float32),
    while q_bytes/r_bytes contain raw E4M3 bytes. This does not check runtime
    versions: the driver must record the observed scope and any deviation.
    """
    inputs, stats, scales, outputs = [np.asarray(a, dtype=np.float32) for a in (inputs, stats, scales, outputs)]
    q_bytes, r_bytes = [np.asarray(a, dtype=np.uint8) for a in (q_bytes, r_bytes)]
    if (gs != 128 or inputs.ndim != 2 or inputs.shape[0] != 2 or inputs.shape[1] == 0
            or inputs.shape[1] % gs or any(a.shape != inputs.shape for a in (q_bytes, r_bytes, outputs))
            or stats.shape != (2, inputs.shape[1] // gs) or scales.shape != stats.shape):
        raise ValueError("switch contract is limited to TP2, gs128 and complete matching sampled groups")
    if not np.isfinite(inputs).all():
        return {"contract": SWITCH_CONTRACT, "passed": False, "error": "nonfinite input"}

    def exact_stage(actual, expected, bits=False):
        mismatch = (actual.view(np.uint32) != expected.view(np.uint32)) if bits else (actual != expected)
        return {
            "passed": not bool(mismatch.any()),
            "mismatch_count_per_tp_rank": mismatch.sum(axis=1).tolist(),
            "comparison": "exact_fp32_bits" if bits else "exact_e4m3_bytes",
        }

    reference = switch_reference(inputs, group_size=gs)
    expected_stats = np.max(np.abs(inputs.reshape(2, -1, gs)), axis=2)
    expected_scales = np.broadcast_to(reference.shared_scales, scales.shape).copy()
    decoded = bf16_round(fp8_e4m3fn_decode(r_bytes).reshape(2, -1, gs) * scales[:, :, None]).reshape(2, -1)
    stages = {
        "input_quantization": exact_stage(q_bytes, fp8_e4m3fn_encode(reference.quantized_inputs)),
        "local_stats": exact_stage(stats, expected_stats, bits=True),
        "common_scales": exact_stage(scales, expected_scales, bits=True),
        "reduction_interval": faithful_interval(q_bytes, r_bytes),
        "bf16_decode": exact_stage(outputs, decoded, bits=True),
    }
    diagnostic = rne_diagnostic(outputs, reference.output)
    diagnostic.update(reference="reference.switch_reference", hard_gate=False, diagnostic_only=True,
                      scope="Previous RNE-equivalence gate; failures are retained and do not become RNE passes.")
    return {
        "contract": SWITCH_CONTRACT,
        "passed": all(stage["passed"] for stage in stages.values()),
        "stages": stages,
        "rne_diagnostic": diagnostic,
        "scope": "Sampled complete groups only; no model or batch-invariance qualification.",
    }
