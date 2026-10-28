"""Count the parameters of a configuration exactly as the runner builds it.

Reports both conventions: real degrees of freedom (complex elements count
twice; used in the paper) and stored tensor elements (``sum(p.numel())``).
No data or GPU are needed.

Examples::

    python tools/count_parameters.py --case resonant_arma_near_unit_circle_v4 \
        --model fno --input-channels 7 -- --width 4 --depth 4 --modes 256
    python tools/count_parameters.py --case narx_softplus --model zno \
        --input-channels 1 -- --width 20 --rank 8 --num-poles 40 --fir-order 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_1d.parameters import parameter_counts  # noqa: E402
from benchmark_discrete.configs import build_case_registry  # noqa: E402
from benchmark_discrete.models import build_model  # noqa: E402
from benchmark_discrete.train import _override_config  # noqa: E402

FLAG_FIELDS = {
    "--width": ("width", int), "--depth": ("depth", int), "--rank": ("rank", int),
    "--modes": ("modes", int), "--num-poles": ("num_poles", int),
    "--fir-order": ("fir_order", int), "--state-dim": ("state_dim", int),
    "--kernel-size": ("kernel_size", int), "--learning-rate": ("learning_rate", float),
    "--radius-limit": ("radius_limit", float), "--safe-radius": ("safe_radius", float),
    "--zno-pole-parameterization": ("pole_parameterization", str),
    "--zno-pole-init-radius-low": ("pole_init_radius_low", float),
    "--zno-pole-init-radius-high": ("pole_init_radius_high", float),
    "--zno-kernel-backend": ("kernel_backend", str), "--grad-clip": ("grad_clip", float),
}
IGNORED_WITH_VALUE = {"--lambda-roll", "--lambda-pole", "--rollout-context-steps", "--epochs",
                      "--batch-size"}


def overrides_from_args(model_args: list[str]) -> dict:
    overrides, i = {}, 0
    while i < len(model_args):
        flag = model_args[i]
        if flag in FLAG_FIELDS:
            name, cast = FLAG_FIELDS[flag]
            overrides[name] = cast(model_args[i + 1])
            i += 2
        elif flag in IGNORED_WITH_VALUE:
            i += 2
        else:
            raise ValueError(f"Unsupported flag for parameter counting: {flag}")
    return overrides


def count(case: str, model: str, input_channels: int, model_args: list[str]) -> dict:
    registry = build_case_registry(REPO_ROOT)
    base = registry[case].models[model]
    config = _override_config(base, **overrides_from_args(model_args))
    built = build_model(model_name=model, config=config, grid=torch.arange(16, dtype=torch.float32),
                        use_grid=False, input_channels=input_channels)
    return parameter_counts(built)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-channels", type=int, required=True)
    parser.add_argument("model_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    model_args = [a for a in args.model_args if a != "--"]
    print(count(args.case, args.model, args.input_channels, model_args))


if __name__ == "__main__":
    main()
