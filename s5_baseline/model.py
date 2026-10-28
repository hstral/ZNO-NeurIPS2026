from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.linalg import block_diag

from .official_imports import configure_official_s5_import


configure_official_s5_import()

from s5.seq_model import StackedEncoderModel  # noqa: E402
from s5.ssm import init_S5SSM  # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO  # noqa: E402


@dataclass(frozen=True)
class S5RegressionConfig:
    d_model: int = 20
    n_layers: int = 4
    ssm_size_base: int = 40
    blocks: int = 1
    activation: str = "half_glu1"
    dropout: float = 0.0
    prenorm: bool = True
    batchnorm: bool = False
    c_init: str = "lecun_normal"
    discretization: str = "zoh"
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    conj_sym: bool = True
    clip_eigs: bool = True
    bidirectional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_official_s5_ssm(config: S5RegressionConfig):
    if config.ssm_size_base % config.blocks:
        raise ValueError("ssm_size_base must be divisible by blocks")

    ssm_size = config.ssm_size_base
    block_size = ssm_size // config.blocks
    if config.conj_sym and (block_size % 2 or ssm_size % 2):
        raise ValueError("Conjugate symmetry requires even block and state sizes")

    eigenvalues, _, _, eigenvectors, _ = make_DPLR_HiPPO(block_size)
    if config.conj_sym:
        block_size //= 2
        ssm_size //= 2
        eigenvalues = eigenvalues[:block_size]
        eigenvectors = eigenvectors[:, :block_size]
        inverse_eigenvectors = eigenvectors.conj().T
    else:
        inverse_eigenvectors = eigenvectors.conj().T

    eigenvalues = (eigenvalues * jnp.ones((config.blocks, block_size))).ravel()
    eigenvectors = block_diag(*([eigenvectors] * config.blocks))
    inverse_eigenvectors = block_diag(*([inverse_eigenvectors] * config.blocks))

    return init_S5SSM(
        H=config.d_model,
        P=ssm_size,
        Lambda_re_init=eigenvalues.real,
        Lambda_im_init=eigenvalues.imag,
        V=eigenvectors,
        Vinv=inverse_eigenvectors,
        C_init=config.c_init,
        discretization=config.discretization,
        dt_min=config.dt_min,
        dt_max=config.dt_max,
        conj_sym=config.conj_sym,
        clip_eigs=config.clip_eigs,
        bidirectional=config.bidirectional,
    )


class S5SequenceRegressor(nn.Module):
    config: S5RegressionConfig
    output_channels: int
    training: bool = True

    def setup(self) -> None:
        ssm = build_official_s5_ssm(self.config)
        batch_encoder = nn.vmap(
            StackedEncoderModel,
            in_axes=(0, 0),
            out_axes=0,
            variable_axes={
                "params": None,
                "dropout": None,
                "batch_stats": None,
                "cache": 0,
                "prime": None,
            },
            split_rngs={"params": False, "dropout": True},
            axis_name="batch",
        )
        self.encoder = batch_encoder(
            ssm=ssm,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            activation=self.config.activation,
            dropout=self.config.dropout,
            training=self.training,
            prenorm=self.config.prenorm,
            batchnorm=self.config.batchnorm,
        )
        self.readout = nn.Dense(self.output_channels)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        integration_steps = jnp.ones(inputs.shape[:2], dtype=inputs.dtype)
        hidden = self.encoder(inputs, integration_steps)
        return self.readout(hidden)


def initialize_model(
    config: S5RegressionConfig,
    *,
    input_channels: int,
    output_channels: int,
    sequence_length: int,
    batch_size: int,
    seed: int = 0,
):
    model = S5SequenceRegressor(
        config=config,
        output_channels=output_channels,
        training=True,
    )
    key = jax.random.PRNGKey(seed)
    params_key, dropout_key = jax.random.split(key)
    dummy = jnp.zeros(
        (batch_size, sequence_length, input_channels),
        dtype=jnp.float32,
    )
    variables = model.init(
        {"params": params_key, "dropout": dropout_key},
        dummy,
    )
    return model, variables["params"]


def count_real_parameters(params) -> int:
    count = 0
    for value in jax.tree_util.tree_leaves(params):
        multiplier = 2 if jnp.issubdtype(value.dtype, jnp.complexfloating) else 1
        count += int(value.size) * multiplier
    return count

