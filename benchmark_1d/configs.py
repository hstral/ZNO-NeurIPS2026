from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class OperatorConfig:
    width: int
    depth: int
    activation: str
    batch_size: int
    learning_rate: float
    epochs: int
    step_size: int = 100
    gamma: float = 0.5
    weight_decay: float = 1e-4
    rank: Optional[int] = None
    modes: Optional[int] = None
    num_poles: Optional[int] = None
    fir_order: int = 4
    radius_limit: float = 0.999
    safe_radius: float = 0.95
    fused_backward_mode: str = "save_history"
    kernel_backend: str = "cuda"


@dataclass(frozen=True)
class CaseConfig:
    name: str
    data_path: str
    use_grid: bool
    sequence_length: int
    train_size: int
    val_size: int
    test_size: int
    models: Dict[str, OperatorConfig]

    def to_dict(self):
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        return payload


def _matched_lno_defaults(base: OperatorConfig, width: int) -> OperatorConfig:
    return replace(base, width=width, depth=1)


def _matched_fno_defaults(base: OperatorConfig) -> OperatorConfig:
    return OperatorConfig(
        width=2,
        depth=4,
        activation=base.activation,
        batch_size=base.batch_size,
        learning_rate=0.002,
        epochs=base.epochs,
        modes=1025,
    )


def _zno_defaults(base: OperatorConfig, num_poles: int) -> OperatorConfig:
    return OperatorConfig(
        width=20,
        depth=4,
        activation=base.activation,
        batch_size=base.batch_size,
        learning_rate=base.learning_rate,
        epochs=base.epochs,
        step_size=base.step_size,
        gamma=base.gamma,
        weight_decay=base.weight_decay,
        rank=8,
        num_poles=num_poles,
        fir_order=4,
        radius_limit=0.999,
        safe_radius=0.95,
        kernel_backend="triton",
    )


def build_case_registry(repo_root: Path) -> Dict[str, CaseConfig]:
    repo_root = Path(repo_root).resolve()

    duffing_c0_lno_base = OperatorConfig(
        width=4,
        depth=1,
        activation="sin",
        batch_size=20,
        learning_rate=0.002,
        epochs=1000,
        modes=16,
    )
    duffing_c05_lno_base = OperatorConfig(
        width=4,
        depth=1,
        activation="sin",
        batch_size=20,
        learning_rate=0.002,
        epochs=1000,
        modes=16,
    )
    pendulum_c0_lno_base = OperatorConfig(
        width=4,
        depth=1,
        activation="sin",
        batch_size=40,
        learning_rate=0.005,
        epochs=1200,
        modes=20,
    )
    pendulum_c05_lno_base = OperatorConfig(
        width=4,
        depth=1,
        activation="sin",
        batch_size=40,
        learning_rate=0.002,
        epochs=1200,
        modes=8,
    )
    lorenz_rho5_lno_base = OperatorConfig(
        width=4,
        depth=1,
        activation="tanh",
        batch_size=20,
        learning_rate=0.005,
        epochs=1000,
        modes=16,
    )
    lorenz_rho10_lno_base = OperatorConfig(
        width=4,
        depth=1,
        activation="tanh",
        batch_size=10,
        learning_rate=0.002,
        epochs=1000,
        modes=84,
    )
    duffing_c0_lno = _matched_lno_defaults(duffing_c0_lno_base, width=13)
    duffing_c05_lno = _matched_lno_defaults(duffing_c05_lno_base, width=13)
    pendulum_c0_lno = _matched_lno_defaults(pendulum_c0_lno_base, width=12)
    pendulum_c05_lno = _matched_lno_defaults(pendulum_c05_lno_base, width=16)
    lorenz_rho5_lno = _matched_lno_defaults(lorenz_rho5_lno_base, width=13)
    lorenz_rho10_lno = _matched_lno_defaults(lorenz_rho10_lno_base, width=6)

    registry = {
        "duffing_c0": CaseConfig(
            name="duffing_c0",
            data_path=str(repo_root / "1D_Duffing_c0" / "Data" / "data.mat"),
            use_grid=False,
            sequence_length=2048,
            train_size=200,
            val_size=50,
            test_size=130,
            models={
                "lno": duffing_c0_lno,
                "fno": _matched_fno_defaults(duffing_c0_lno_base),
                "zno": _zno_defaults(duffing_c0_lno_base, num_poles=16),
            },
        ),
        "duffing_c05": CaseConfig(
            name="duffing_c05",
            data_path=str(repo_root / "1D_Duffing_c05" / "Data" / "data.mat"),
            use_grid=False,
            sequence_length=2048,
            train_size=200,
            val_size=50,
            test_size=130,
            models={
                "lno": duffing_c05_lno,
                "fno": _matched_fno_defaults(duffing_c05_lno_base),
                "zno": _zno_defaults(duffing_c05_lno_base, num_poles=16),
            },
        ),
        "pendulum_c0": CaseConfig(
            name="pendulum_c0",
            data_path=str(repo_root / "1D_Pendulum_c0" / "Data" / "data.mat"),
            use_grid=True,
            sequence_length=2048,
            train_size=200,
            val_size=50,
            test_size=130,
            models={
                "lno": pendulum_c0_lno,
                "fno": _matched_fno_defaults(pendulum_c0_lno_base),
                "zno": _zno_defaults(pendulum_c0_lno_base, num_poles=20),
            },
        ),
        "pendulum_c05": CaseConfig(
            name="pendulum_c05",
            data_path=str(repo_root / "1D_Pendulum_c05" / "Data" / "data.mat"),
            use_grid=True,
            sequence_length=2048,
            train_size=200,
            val_size=50,
            test_size=130,
            models={
                "lno": pendulum_c05_lno,
                "fno": _matched_fno_defaults(pendulum_c05_lno_base),
                "zno": _zno_defaults(pendulum_c05_lno_base, num_poles=8),
            },
        ),
        "lorenz_rho5": CaseConfig(
            name="lorenz_rho5",
            data_path=str(repo_root / "1D_Lorenz_rho5" / "Data" / "data.mat"),
            use_grid=True,
            sequence_length=2048,
            train_size=200,
            val_size=50,
            test_size=130,
            models={
                "lno": lorenz_rho5_lno,
                "fno": _matched_fno_defaults(lorenz_rho5_lno_base),
                "zno": _zno_defaults(lorenz_rho5_lno_base, num_poles=16),
            },
        ),
        "lorenz_rho10": CaseConfig(
            name="lorenz_rho10",
            data_path=str(repo_root / "1D_Lorenz_rho10" / "Data" / "data.mat"),
            use_grid=True,
            sequence_length=2048,
            train_size=200,
            val_size=50,
            test_size=130,
            models={
                "lno": lorenz_rho10_lno,
                "fno": _matched_fno_defaults(lorenz_rho10_lno_base),
                "zno": _zno_defaults(lorenz_rho10_lno_base, num_poles=32),
            },
        ),
    }

    return registry
