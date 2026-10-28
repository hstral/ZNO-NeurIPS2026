"""Equivalence of the ZNO recurrent backends.

``triton_fast`` (chunked parallel scan, one lane per conjugate pair) must compute the
same layer output and gradients as the original ``triton`` kernel and as a float64
sequential reference. The GPU checks need CUDA, Triton, and compute capability >= 7.0
and are skipped otherwise; the CPU checks always run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark_1d.models import LowRankRationalLayer1d  # noqa: E402
from benchmark_1d.triton_scan_fast import triton_fast_available  # noqa: E402

# (batch, time, channels, rank, poles, fir, radius_low, radius_high, block_t)
GPU_CONFIGS = [
    (4, 2048, 20, 8, 40, 4, None, None, 32),      # 8.5k configuration of the paper
    (4, 2048, 20, 8, 40, 4, 0.98, 0.998, 32),     # near-unit-circle poles
    (3, 2048, 20, 12, 64, 0, None, None, 32),     # no FIR branch
    (2, 1000, 20, 5, 7, 2, None, None, 32),       # odd pole count, ragged last tile
    (2, 2048, 20, 8, 40, 4, None, None, 16),      # other tile sizes
    (2, 2048, 20, 8, 40, 4, None, None, 64),
]


def _gpu_ok() -> bool:
    return (torch.cuda.is_available() and triton_fast_available()
            and torch.cuda.get_device_capability()[0] >= 7)


def reference_f64(layer, x):
    """Sequential float64 evaluation of the layer (same math as the CPU path)."""
    poles, residues = layer.stable_poles_and_residues()
    poles, residues = poles.to(torch.complex128), residues.to(torch.complex128)
    w_in = torch.cat((layer.in_proj.weight, layer.skip.weight), 0).double()
    b_in = torch.cat((layer.skip.bias.new_zeros(layer.rank), layer.skip.bias)).double()
    proj_skip = torch.nn.functional.linear(x.double(), w_in, b_in)
    u, skip = proj_skip[..., : layer.rank], proj_skip[..., layer.rank:]
    batch, steps, rank = u.shape
    state = torch.zeros(batch, rank, poles.size(-1), dtype=torch.complex128, device=x.device)
    fir = layer.fir_weights.double() if layer.fir_order > 0 else None
    out = []
    for n in range(steps):
        state = state * poles + u[:, n, :, None].to(torch.complex128)
        y = (state * residues).sum(-1).real
        if fir is not None:
            for j in range(fir.size(1)):
                if n - j >= 0:
                    y = y + fir[:, j] * u[:, n - j, :]
        out.append(y)
    return torch.nn.functional.linear(torch.stack(out, 1), layer.out_proj.weight.double()) + skip


def _run(layer, x, w, backend):
    layer.kernel_backend = backend
    layer.zero_grad(set_to_none=True)
    xg = x.clone().requires_grad_(True)
    y = reference_f64(layer, xg) if backend == "f64" else layer.scan_sequence(xg, layer.scan_params())
    (y.double() * w).sum().backward()
    grads = {"input": xg.grad.detach().double()}
    grads.update({name: p.grad.detach().double() for name, p in layer.named_parameters()})
    return y.detach().double(), grads


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


@pytest.mark.parametrize("num_poles", [7, 8, 1])
def test_lane_layout_matches_full_layout(num_poles):
    torch.manual_seed(0)
    layer = LowRankRationalLayer1d(6, 3, num_poles, 2, 0.999, "save_history", "triton")
    full_p, full_c = layer.stable_poles_and_residues()
    lane_p, lane_c = layer.lane_poles_and_residues()
    pairs = layer.num_pairs
    assert lane_p.shape[-1] == pairs + layer.has_real_pole
    assert torch.equal(lane_p[..., :pairs], full_p[..., :pairs])
    assert torch.equal(full_p[..., pairs:2 * pairs], lane_p[..., :pairs].conj())
    assert torch.equal(lane_c[..., :pairs], full_c[..., :pairs])
    if layer.has_real_pole:
        assert torch.equal(lane_p[..., -1], full_p[..., -1])


def test_triton_fast_cpu_fallback_matches_reference():
    """On CPU, triton_fast falls back to the reference scan with the full pole layout."""
    torch.manual_seed(0)
    layer = LowRankRationalLayer1d(6, 3, 7, 2, 0.999, "save_history", "triton")
    x = torch.randn(2, 40, 6)
    w = torch.randn(2, 40, 6, dtype=torch.float64)
    y_ref, g_ref = _run(layer, x, w, "triton")
    y_fast, g_fast = _run(layer, x, w, "triton_fast")
    assert torch.equal(y_ref, y_fast)
    for key in g_ref:
        assert torch.equal(g_ref[key], g_fast[key]), key


@pytest.mark.skipif(not _gpu_ok(), reason="needs CUDA, Triton, and compute capability >= 7.0")
@pytest.mark.parametrize("config", GPU_CONFIGS)
def test_triton_fast_matches_triton_and_float64(config):
    import benchmark_1d.triton_scan_fast as fast

    batch, steps, chans, rank, poles, fir, lo, hi, block_t = config
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    torch.manual_seed(1234)
    layer = LowRankRationalLayer1d(chans, rank, poles, fir, 0.999, "save_history", "triton",
                                   pole_init_radius_low=lo, pole_init_radius_high=hi).to(device)
    old_block = fast.DEFAULT_BLOCK_T
    fast.DEFAULT_BLOCK_T = block_t
    try:
        x = torch.randn(batch, steps, chans, device=device)
        w = torch.randn(batch, steps, chans, device=device, dtype=torch.float64)
        y_old, g_old = _run(layer, x, w, "triton")
        y_new, g_new = _run(layer, x, w, "triton_fast")
        y_ref, g_ref = _run(layer, x, w, "f64")
    finally:
        fast.DEFAULT_BLOCK_T = old_block
    rows = [("output", y_old, y_new, y_ref)] + [(k, g_old[k], g_new[k], g_ref[k]) for k in g_ref]
    for name, a_old, a_new, a_ref in rows:
        e_old, e_new = _rel(a_old, a_ref), _rel(a_new, a_ref)
        assert e_new <= max(4 * e_old, 1e-6), (name, e_old, e_new)
        assert _rel(a_new, a_old) <= 1e-4, name
