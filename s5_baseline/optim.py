"""Optimizer recipe of the official S5 code base.

SSM parameters (B, Lambda, log-step) are trained with Adam without weight
decay; all other parameters use AdamW. The learning rate follows a one-epoch
linear warm-up and a cosine decay to 1e-6.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import optax
from flax import traverse_util
from flax.core import FrozenDict, freeze


SSM_PARAMETER_NAMES = {"B", "Lambda_re", "Lambda_im", "log_step"}


def official_parameter_labels(params):
    flat = traverse_util.flatten_dict(params)
    labels = {
        path: "ssm" if path[-1] in SSM_PARAMETER_NAMES else "regular"
        for path in flat
    }
    restored = traverse_util.unflatten_dict(labels)
    return freeze(restored) if isinstance(params, FrozenDict) else restored


def official_warmup_cosine_schedule(
    base_learning_rate: float,
    *,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int = 1,
    minimum_learning_rate: float = 1e-6,
):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, epochs * steps_per_epoch)
    cosine_steps = total_steps - warmup_steps

    def schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        warmup = base_learning_rate * (step + 1.0) / float(warmup_steps)
        progress = jnp.clip((step - warmup_steps) / float(cosine_steps), 0.0, 1.0)
        cosine = minimum_learning_rate + 0.5 * (
            base_learning_rate - minimum_learning_rate
        ) * (1.0 + jnp.cos(math.pi * progress))
        return jnp.where(step < warmup_steps, warmup, cosine)

    return schedule


def build_optimizer(
    params,
    *,
    learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
    weight_decay: float,
    grad_clip: float | None,
):
    schedule = official_warmup_cosine_schedule(
        learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )
    grouped = optax.multi_transform(
        {
            "ssm": optax.adam(schedule),
            "regular": optax.adamw(schedule, weight_decay=weight_decay),
        },
        official_parameter_labels(params),
    )
    if grad_clip is not None:
        return optax.chain(optax.clip_by_global_norm(grad_clip), grouped), schedule
    return grouped, schedule
