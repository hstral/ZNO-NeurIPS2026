from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np


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
    pole_parameterization: str = "z"
    pole_init_radius_low: Optional[float] = None
    pole_init_radius_high: Optional[float] = None
    fused_backward_mode: str = "save_history"
    kernel_backend: str = "triton"
    state_dim: Optional[int] = None
    kernel_size: Optional[int] = None
    grad_clip: Optional[float] = None


@dataclass(frozen=True)
class CaseConfig:
    name: str
    data_path: str
    use_grid: bool
    sequence_length: int
    train_size: int
    val_size: int
    test_size: int
    dt: float
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
        modes=128,
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


def _s4d_defaults(base: OperatorConfig) -> OperatorConfig:
    return OperatorConfig(
        width=20,
        depth=4,
        activation=base.activation,
        batch_size=base.batch_size,
        learning_rate=5e-4,
        epochs=base.epochs,
        step_size=base.step_size,
        gamma=base.gamma,
        weight_decay=base.weight_decay,
        state_dim=32,
        grad_clip=1.0,
    )


def _linear_fir_defaults(base: OperatorConfig) -> OperatorConfig:
    return OperatorConfig(
        width=1,
        depth=1,
        activation=base.activation,
        batch_size=base.batch_size,
        learning_rate=1e-3,
        epochs=base.epochs,
        step_size=base.step_size,
        gamma=base.gamma,
        weight_decay=0.0,
        kernel_size=129,
    )


def build_case_registry(
    repo_root: Path,
    data_root: Optional[Path] = None,
) -> Dict[str, CaseConfig]:
    repo_root = Path(repo_root).resolve()
    if data_root is None:
        data_root = repo_root / "data" / "controlled"
    else:
        data_root = Path(data_root).resolve()

    base = OperatorConfig(
        width=4,
        depth=1,
        activation="gelu",
        batch_size=32,
        learning_rate=2e-3,
        epochs=600,
        step_size=150,
        gamma=0.5,
        modes=32,
    )

    cases = {
        "arma_stable": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 16,
        },
        "narx_softplus": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 16,
        },
        "iir_filter_bank": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 16,
        },
        "dss_surrogate": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 16,
        },
        "resonant_arma_near_unit_circle": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        },
        "nonlinear_stable_narx_latent": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 24,
        },
        "step_aware_variable_dt_family": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        },
        "resonant_arma_near_unit_circle_v4": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        },
        "nonlinear_stable_narx_latent_v4": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 24,
        },
        "step_aware_variable_dt_family_v4": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        },
        "high_order_iir_cascade_v5": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 48,
        },
        "digital_pid_controlled_plant_v5": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 24,
        },
        "silverbox_public_v1": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        },
        "cascaded_tanks_public_v1": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        },
        "emps_public_v1": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        },
        "ced_public_v1": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        },
        "wiener_hammer_public_v1": {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        },
    }

    for case_name in (
        "resonant_arma_lm_r0900_0930_v1",
        "resonant_arma_lm_r0930_0960_v1",
        "resonant_arma_lm_r0960_0980_v1",
        "resonant_arma_lm_r0980_0990_v1",
        "resonant_arma_lm_r0990_0995_v1",
    ):
        cases[case_name] = {
            "sequence_length": 2048,
            "dt": 1.0,
            "lno_width": 12,
            "zno_num_poles": 32,
        }

    registry = {}
    for case_name, meta in cases.items():
        data_path = data_root / case_name / "data.npz"
        sequence_length = meta["sequence_length"]
        train_size = 1024
        val_size = 256
        test_size = 256
        if data_path.exists():
            with np.load(data_path) as raw:
                sequence_length = int(raw["x_train"].shape[1])
                train_size = int(raw["x_train"].shape[0])
                val_size = int(raw["x_val"].shape[0])
                test_size = int(raw["x_test"].shape[0])
        registry[case_name] = CaseConfig(
            name=case_name,
            data_path=str(data_path),
            use_grid=False,
            sequence_length=sequence_length,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            dt=meta["dt"],
            models={
                "lno": _matched_lno_defaults(base, width=meta["lno_width"]),
                "lno_stable": _matched_lno_defaults(base, width=meta["lno_width"]),
                "fno": _matched_fno_defaults(base),
                "zno": _zno_defaults(base, num_poles=meta["zno_num_poles"]),
                "zno_splane_iso": _zno_defaults(base, num_poles=meta["zno_num_poles"]),
                "s4d": _s4d_defaults(base),
                "s4d_official": _s4d_defaults(base),
                "linear_fir": _linear_fir_defaults(base),
            },
        )
    return registry
