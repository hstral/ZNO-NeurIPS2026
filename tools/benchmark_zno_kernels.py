"""Time the ZNO recurrent kernels (forward + backward) at the paper's 8.5k layer shape.

Compares the original ``triton`` kernel with ``triton_fast`` over its tile sizes and
warp counts. Needs a CUDA GPU with Triton.

Usage::

    python tools/benchmark_zno_kernels.py [--batch-size 32] [--length 2048]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_1d.models import LowRankRationalLayer1d  # noqa: E402
from benchmark_1d.triton_recurrent import triton_recurrent_scan  # noqa: E402
from benchmark_1d.triton_scan_fast import triton_fast_recurrent_scan  # noqa: E402


def timeit(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=2048)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7:
        raise SystemExit("needs a CUDA GPU of compute capability 7.0 or higher")
    device = torch.device("cuda")
    torch.manual_seed(0)
    # w=20, r=8, K=40 (20 conjugate pairs), F=4.
    layer = LowRankRationalLayer1d(20, 8, 40, 4, 0.999, "save_history", "triton").to(device)
    poles, residues = layer.stable_poles_and_residues()
    poles, residues = poles.detach().requires_grad_(True), residues.detach().requires_grad_(True)
    lane_p = poles.detach()[:, :20].clone().requires_grad_(True)
    lane_c = residues.detach()[:, :20].clone().requires_grad_(True)
    fir = layer.fir_weights.detach().clone().requires_grad_(True)
    u = torch.randn(args.batch_size, args.length, 8, device=device, requires_grad=True)
    g = torch.randn(args.batch_size, args.length, 8, device=device)

    def old_fb():
        triton_recurrent_scan(u, poles, residues, fir).backward(g)

    def old_f():
        with torch.no_grad():
            triton_recurrent_scan(u, poles, residues, fir)

    print(f"triton:      fwd (no grad) {timeit(old_f):7.3f} ms   fwd+bwd {timeit(old_fb):7.3f} ms")
    for block_t in (16, 32, 64):
        for fwd_warps in (1, 2, 4):
            for bwd_warps in (1, 2, 4, 8):
                def new_fb():
                    triton_fast_recurrent_scan(u, lane_p, lane_c, fir, 20, block_t=block_t,
                                               num_warps=bwd_warps, forward_warps=fwd_warps).backward(g)

                def new_f():
                    with torch.no_grad():
                        triton_fast_recurrent_scan(u, lane_p, lane_c, fir, 20, block_t=block_t,
                                                   forward_warps=fwd_warps)
                label = f"triton_fast BLOCK_T={block_t:3d} fwd_warps={fwd_warps} bwd_warps={bwd_warps}"
                try:
                    print(f"{label}: fwd (no grad) {timeit(new_f):7.3f} ms   fwd+bwd {timeit(new_fb):7.3f} ms")
                except Exception as exc:  # resource limits for large tiles
                    print(f"{label}: failed ({type(exc).__name__})")


if __name__ == "__main__":
    main()
