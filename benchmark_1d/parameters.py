"""Parameter counting conventions used throughout the paper.

All parameter counts reported in the paper are *real degrees of freedom*: a
complex-valued tensor element counts as two real parameters. PyTorch's
``sum(p.numel())`` counts a complex element once, which undercounts models
with native complex weights (FNO spectral weights, LNO residues). Both values
are written to every run summary so that the two conventions can be compared.
"""

from typing import Dict

import torch


def count_parameter_elements(model: torch.nn.Module) -> int:
    """Number of stored tensor elements (complex elements counted once)."""
    return int(sum(param.numel() for param in model.parameters()))


def count_real_parameters(model: torch.nn.Module) -> int:
    """Number of real degrees of freedom (complex elements counted twice)."""
    return int(
        sum(
            param.numel() * (2 if param.is_complex() else 1)
            for param in model.parameters()
        )
    )


def parameter_counts(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "parameter_count": count_real_parameters(model),
        "parameter_count_elements": count_parameter_elements(model),
    }
