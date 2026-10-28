import argparse
from pathlib import Path

import torch

from benchmark_1d import build_case_registry, run_experiment


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified 1D LNO/FNO/ZNO benchmark runner."
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=[
            "duffing_c0",
            "duffing_c05",
            "pendulum_c0",
            "pendulum_c05",
            "lorenz_rho5",
            "lorenz_rho10",
        ],
    )
    parser.add_argument("--model", required=True, choices=["lno", "fno", "zno"])
    parser.add_argument("--run-name", default="default")
    parser.add_argument(
        "--output-dir",
        default="runs/one_d",
        help="Root directory for saved runs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device, e.g. cpu or cuda.",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--modes", type=int)
    parser.add_argument("--num-poles", type=int)
    parser.add_argument("--fir-order", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--lambda-pole", type=float, default=1e-3)
    parser.add_argument("--lambda-roll", type=float, default=1e-2)
    parser.add_argument("--rollout-context-steps", type=int)
    parser.add_argument("--safe-radius", type=float)
    parser.add_argument(
        "--zno-fused-backward-mode",
        choices=["save_history", "recompute"],
        help="CUDA fused backward mode for ZNO.",
    )
    parser.add_argument(
        "--zno-kernel-backend",
        choices=["cuda", "triton", "triton_fast"],
        help="Kernel backend for the ZNO recurrent scan (GPU). triton: the kernel used for the "
        "paper's runs; triton_fast: chunked parallel scan, about 1.5x faster training, equal up "
        "to float32 rounding; cuda: fused CUDA/C++ extension. CPU runs use the PyTorch reference scan.",
    )
    return parser.parse_args(argv)


def experiment_kwargs(argv=None) -> dict:
    """Keyword arguments of ``run_experiment`` for a command line (without the program name)."""
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    registry = build_case_registry(repo_root)
    return dict(
        repo_root=repo_root,
        case=registry[args.case],
        model_name=args.model,
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        width=args.width,
        depth=args.depth,
        rank=args.rank,
        modes=args.modes,
        num_poles=args.num_poles,
        fir_order=args.fir_order,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        lambda_pole=args.lambda_pole,
        lambda_roll=args.lambda_roll,
        rollout_context_steps=args.rollout_context_steps,
        safe_radius=args.safe_radius,
        fused_backward_mode=args.zno_fused_backward_mode,
        kernel_backend=args.zno_kernel_backend,
    )


def main():
    run_experiment(**experiment_kwargs())


if __name__ == "__main__":
    main()
