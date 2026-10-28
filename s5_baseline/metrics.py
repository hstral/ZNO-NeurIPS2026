from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def relative_l2_jax(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    ignore_prefix_steps: int = 0,
) -> jnp.ndarray:
    if ignore_prefix_steps:
        prediction = prediction[:, ignore_prefix_steps:, :]
        target = target[:, ignore_prefix_steps:, :]
    batch_size = prediction.shape[0]
    diff = (prediction - target).reshape(batch_size, -1)
    flat_target = target.reshape(batch_size, -1)
    ratios = jnp.linalg.norm(diff, axis=1) / (
        jnp.linalg.norm(flat_target, axis=1) + 1e-8
    )
    return jnp.mean(ratios)


def relative_l2_numpy(
    prediction: np.ndarray,
    target: np.ndarray,
    ignore_prefix_steps: int = 0,
) -> float:
    if ignore_prefix_steps:
        prediction = prediction[:, ignore_prefix_steps:, :]
        target = target[:, ignore_prefix_steps:, :]
    batch_size = prediction.shape[0]
    diff = (prediction - target).reshape(batch_size, -1)
    flat_target = target.reshape(batch_size, -1)
    ratios = np.linalg.norm(diff, axis=1) / (
        np.linalg.norm(flat_target, axis=1) + 1e-8
    )
    return float(ratios.mean())

