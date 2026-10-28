"""Checks for the official-S5 adapter (skipped when JAX or the S5 checkout is missing).

Verifies the pinned commit, metric parity with the PyTorch loss, causality,
variable-length inputs, and the closed-form parameter count used in
``experiments/verify_parameter_counts.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import jax
    import jax.numpy as jnp

    from s5_baseline.metrics import relative_l2_jax, relative_l2_numpy
    from s5_baseline.model import (S5RegressionConfig, S5SequenceRegressor,
                                   count_real_parameters, initialize_model)
    from s5_baseline.official_imports import OFFICIAL_S5_COMMIT, OFFICIAL_S5_ROOT
    AVAILABLE = True
except (ImportError, FileNotFoundError):  # pragma: no cover - optional dependency
    AVAILABLE = False

if not AVAILABLE:
    import pytest
    pytest.skip("JAX/Flax or the official S5 checkout is not available", allow_module_level=True)

from s5_baseline.closed_form import s5_real_parameters  # noqa: E402


def test_pinned_commit():
    if (OFFICIAL_S5_ROOT / ".git").exists():
        actual = subprocess.check_output(["git", "-C", str(OFFICIAL_S5_ROOT), "rev-parse", "HEAD"],
                                         text=True).strip()
        assert actual == OFFICIAL_S5_COMMIT


def test_metric_matches_numpy_reference():
    rng = np.random.default_rng(0)
    prediction = rng.normal(size=(5, 17, 3)).astype(np.float32)
    target = rng.normal(size=(5, 17, 3)).astype(np.float32)
    for prefix in (0, 4):
        expected = relative_l2_numpy(prediction, target, prefix)
        actual = float(relative_l2_jax(jnp.asarray(prediction), jnp.asarray(target), prefix))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_causality_and_variable_length():
    config = S5RegressionConfig(d_model=8, n_layers=2, ssm_size_base=16)
    _, params = initialize_model(config, input_channels=3, output_channels=1,
                                 sequence_length=16, batch_size=2, seed=0)
    model = S5SequenceRegressor(config=config, output_channels=1, training=False)
    inputs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 3))
    perturbed = np.asarray(inputs).copy()
    perturbed[:, 10:, :] += 100.0
    baseline = np.asarray(model.apply({"params": params}, inputs))
    changed = np.asarray(model.apply({"params": params}, jnp.asarray(perturbed)))
    assert np.allclose(baseline[:, :10], changed[:, :10], atol=2e-5)
    assert model.apply({"params": params}, inputs[:, :8, :]).shape == (2, 8, 1)


def test_closed_form_parameter_count():
    for input_channels, expected in ((7, 8741), (16, 8921), (1, 8621)):
        config = S5RegressionConfig(d_model=20, n_layers=4, ssm_size_base=40)
        _, params = initialize_model(config, input_channels=input_channels, output_channels=1,
                                     sequence_length=8, batch_size=1, seed=0)
        assert count_real_parameters(params) == expected
        assert s5_real_parameters(input_channels=input_channels, d_model=20, n_layers=4,
                                  ssm_size_base=40) == expected


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("passed", name)
