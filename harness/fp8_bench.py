#!/usr/bin/env python3
"""Public TP2 FP8 benchmark entry point; delegates all gates to the tested driver.

Default comparison: NCCL BF16, native NVFP4, previous tuned UE8M0 FP8,
and the batched Lamport UE8M0 FP8 kernel. Defaults match the measured GB200
configuration; explicit options override them. Runtime requires two GPUs.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys

import fp8_fused_compare as driver


PUBLIC_BACKENDS = (
    "nccl", "fp4", "fp8-ue8m0-sm", "fp8-ue8m0-tuned",
    "fp8-ue8m0-lamport-batched",
)
DEFAULT_ARGUMENTS = (
    "--rows", "1,4,16,64,1024,4096,16384",
    "--backends", "nccl,fp4,fp8-ue8m0-tuned,fp8-ue8m0-lamport-batched",
    "--fused-tile", "16384", "--fused-ctas", "152",
    "--ue8m0-tile", "32768", "--ue8m0-ctas", "152",
    "--ue8m0-threads", "256",
    "--lamport-tile", "2048", "--lamport-ctas", "152",
    "--lamport-teams", "7",
    "--graph-k", "10", "--iters", "200", "--warmup", "20",
    "--stress", "100",
)


@contextmanager
def _arguments(arguments):
    previous = sys.argv
    sys.argv = [previous[0], *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def prepare_arguments(arguments):
    """Validate with the real driver's parser before any CUDA initialization."""
    forwarded = [*DEFAULT_ARGUMENTS, *arguments]
    with _arguments(forwarded):
        parsed = driver.parse_args()
    # Validate the resolved namespace, including argparse's accepted option
    # abbreviations, so alternate spellings cannot bypass the public allowlist.
    unsupported = set(parsed.backends) - set(PUBLIC_BACKENDS)
    if unsupported:
        raise SystemExit(
            "Unsupported public backend(s): " + ", ".join(sorted(unsupported))
            + ". Choose from: " + ", ".join(PUBLIC_BACKENDS)
        )
    return forwarded


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if "--help" in arguments or "-h" in arguments:
        print(__doc__)
        print("Public backends: " + ", ".join(PUBLIC_BACKENDS))
        print("At least one UE8M0 backend is required. Shared driver options follow.\n")
    forwarded = prepare_arguments(arguments)
    with _arguments(forwarded):
        return driver.main()


if __name__ == "__main__":
    main()
