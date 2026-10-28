"""Locate and import the pinned upstream S5 package without modifying it."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp


OFFICIAL_S5_URL = "https://github.com/lindermanlab/S5"
OFFICIAL_S5_COMMIT = "3c18fdb6b06414da35e77b94b9cd855f6a95ef17"
OFFICIAL_S5_ROOT = Path(
    os.environ.get(
        "ZNO_S5_ROOT",
        Path(__file__).resolve().parents[1] / "third_party" / "S5",
    )
)


def configure_official_s5_import() -> None:
    """Expose the pinned upstream package on ``sys.path``."""
    if not (OFFICIAL_S5_ROOT / "s5").is_dir():
        raise FileNotFoundError(
            f"Official S5 checkout not found at {OFFICIAL_S5_ROOT}. "
            "Run `bash scripts/fetch_official_s5.sh` or set ZNO_S5_ROOT."
        )
    root = str(OFFICIAL_S5_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    # The pinned source uses the pre-JAX-0.4 type alias only in annotations.
    if not hasattr(jnp, "DeviceArray"):
        setattr(jnp, "DeviceArray", jax.Array)


configure_official_s5_import()
