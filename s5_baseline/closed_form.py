"""Closed-form real parameter count of the S5 sequence-regression adapter.

Input lift (``input_channels -> H``), four official S5 blocks (complex B and
C stored as real pairs, eigenvalues, log-steps, feedthrough D, ``half_glu1``
dense layer, layer norm), and a scalar readout. Pure Python; no JAX needed.
"""


def s5_real_parameters(*, input_channels: int, d_model: int, n_layers: int, ssm_size_base: int) -> int:
    if n_layers != 4:
        raise ValueError("The closed form is written for four layers")
    h, p = d_model, ssm_size_base
    return 4 * h * h + (18 + input_channels) * h + 8 * p * h + 6 * p + 1
