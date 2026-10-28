import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_1d.fused_recurrent import precompile_fused_recurrent_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompile the fused recurrent CUDA extension.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backward-mode", choices=["save_history", "recompute"], default="save_history")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precompile_fused_recurrent_extension(device=args.device, backward_mode=args.backward_mode)
    print(
        {
            "status": "ok",
            "device": args.device,
            "backward_mode": args.backward_mode,
        }
    )
