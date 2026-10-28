"""Diagnostics of trained public-benchmark checkpoints (Appendix K)."""

from __future__ import annotations

import json
from pathlib import Path

from ..protocols import DIAGNOSTIC_SEEDS


def seed_checkpoints(directory: Path, seeds=DIAGNOSTIC_SEEDS) -> list[Path]:
    """``model_best.pt`` of the runs in ``directory`` whose seed is in ``seeds``.

    ``runs/public`` holds seeds 0-7 of each selected configuration; the
    diagnostics use seeds 0-4.
    """
    paths = []
    for path in sorted(Path(directory).glob("*/model_best.pt")):
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        if int(summary["seed"]) in seeds:
            paths.append(path)
    return paths
