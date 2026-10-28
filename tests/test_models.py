"""CPU unit tests for the PyTorch models and the conventions used in the paper.

Run with ``python -m pytest tests`` (or ``python tests/test_models.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark_1d.parameters import parameter_counts  # noqa: E402
from benchmark_discrete.configs import build_case_registry  # noqa: E402
from benchmark_discrete.models import build_model  # noqa: E402
from benchmark_discrete.train import LpLoss, _override_config  # noqa: E402

REGISTRY = build_case_registry(REPO_ROOT)
GRID = torch.arange(32, dtype=torch.float32)


def make(case: str, model: str, input_channels: int = 1, **overrides):
    config = _override_config(REGISTRY[case].models[model], **overrides)
    torch.manual_seed(0)
    return build_model(model_name=model, config=config, grid=GRID, use_grid=False,
                       input_channels=input_channels)


def test_real_parameter_counts_of_original_operating_point():
    fno = make("resonant_arma_near_unit_circle_v4", "fno", 7, width=4, depth=4, modes=256)
    assert parameter_counts(fno) == {"parameter_count": 16845, "parameter_count_elements": 8589}
    s4d = make("resonant_arma_near_unit_circle_v4", "s4d_official", 7, width=20, depth=4, state_dim=12)
    assert parameter_counts(s4d)["parameter_count"] == 8577
    zno = make("resonant_arma_near_unit_circle_v4", "zno", 7, width=20, depth=4, rank=8,
               num_poles=40, fir_order=4)
    counts = parameter_counts(zno)
    assert counts["parameter_count"] == counts["parameter_count_elements"] == 8657


def _all_pole_radii(model) -> torch.Tensor:
    radii = []
    for layer in model.layers:
        poles, _ = layer.stable_poles_and_residues()
        radii.append(poles.detach().abs().reshape(-1))
    return torch.cat(radii)


def test_bounded_radius_map_respects_ceiling():
    model = make("narx_softplus", "zno", rank=4, num_poles=8, radius_limit=0.999)
    with torch.no_grad():
        for layer in model.layers:
            layer.pair_radius.fill_(50.0)  # saturate the sigmoid
    assert float(_all_pole_radii(model).max()) <= 0.999 + 1e-6


def test_unbounded_radius_map_is_positive_and_unconstrained():
    model = make("narx_softplus", "zno", rank=4, num_poles=8, pole_parameterization="z_unbounded")
    assert float(_all_pole_radii(model).min()) > 0.0
    with torch.no_grad():
        for layer in model.layers:
            layer.pair_radius.fill_(3.0)  # softplus(3) > 1: no hard ceiling
    assert float(_all_pole_radii(model).max()) > 1.0


def test_pole_initialization_range():
    default = make("narx_softplus", "zno", rank=8, num_poles=32)
    radii = _all_pole_radii(default)
    assert 0.30 - 1e-5 <= float(radii.min()) and float(radii.max()) <= 0.95 + 1e-5
    high = make("narx_softplus", "zno", rank=8, num_poles=32,
                pole_init_radius_low=0.90, pole_init_radius_high=0.995)
    radii = _all_pole_radii(high)
    assert 0.90 - 1e-5 <= float(radii.min()) and float(radii.max()) <= 0.995 + 1e-5


def test_stabilized_lno_poles_lie_in_left_half_plane():
    model = make("narx_softplus", "lno_stable", width=4, modes=4)
    with torch.no_grad():
        for layer in model.layers:
            layer.raw_decay.normal_(0.0, 5.0)
    assert float(model.stability_stats()["max_continuous_pole_real_part"]) < 0.0


def test_zno_is_causal():
    model = make("narx_softplus", "zno", width=8, depth=2, rank=2, num_poles=4, fir_order=2).eval()
    x = torch.randn(2, 32, 1)
    perturbed = x.clone()
    perturbed[:, 20:, :] += 10.0
    with torch.no_grad():
        baseline, changed = model(x), model(perturbed)
    assert torch.allclose(baseline[:, :20], changed[:, :20], atol=1e-5)
    assert not torch.allclose(baseline[:, 20:], changed[:, 20:])


def test_relative_l2_is_per_example():
    loss = LpLoss(size_average=True)
    target = torch.tensor([[[1.0], [0.0]], [[100.0], [0.0]]])
    prediction = target + torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
    # Mean of per-example ratios (1/1 and 1/100), not one norm over the batch.
    assert np.isclose(float(loss(prediction, target)), (1.0 + 0.01) / 2, atol=1e-6)


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("passed", name)
