import argparse
from pathlib import Path

import numpy as np


def _stable_input_signal(length: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
    signal = (
        0.8 * np.sin(2 * np.pi * rng.uniform(1.0, 5.0) * t + rng.uniform(0, 2 * np.pi))
        + 0.4 * np.sin(2 * np.pi * rng.uniform(5.0, 12.0) * t + rng.uniform(0, 2 * np.pi))
        + 0.2 * rng.standard_normal(length)
    )
    envelope = 0.75 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.25, 1.5) * t + rng.uniform(0, 2 * np.pi))
    return (signal * envelope).astype(np.float32)


def _parameter_prompt(params: np.ndarray, prompt_len: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    prompt = np.zeros(prompt_len, dtype=np.float32)
    block = max(prompt_len // max(len(params), 1), 1)
    for idx, value in enumerate(params):
        start = idx * block
        end = prompt_len if idx == len(params) - 1 else min(prompt_len, start + block)
        prompt[start:end] = np.float32(value)
    return prompt


def _stack_conditioned_input(signal: np.ndarray, params: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1, 1)
    params = np.asarray(params, dtype=np.float32).reshape(1, -1)
    conditioned = np.repeat(params, repeats=signal.shape[0], axis=0)
    return np.concatenate([signal, conditioned], axis=1).astype(np.float32)


def _simulate_arma(length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = _stable_input_signal(length, rng)
    y = np.zeros(length, dtype=np.float32)
    radius = rng.uniform(0.45, 0.92)
    phase = rng.uniform(0.15, np.pi - 0.15)
    a1 = 2.0 * radius * np.cos(phase)
    a2 = -(radius**2)
    b = rng.uniform(-0.25, 0.25, size=3).astype(np.float32)
    b[0] += np.sign(b[0]) * 0.25 if b[0] != 0 else 0.25
    for t in range(length):
        acc = 0.0
        if t - 1 >= 0:
            acc += a1 * y[t - 1]
        if t - 2 >= 0:
            acc += a2 * y[t - 2]
        for j in range(3):
            if t - j >= 0:
                acc += b[j] * x[t - j]
        y[t] = acc
    return x, y


def _simulate_narx(length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = _stable_input_signal(length, rng)
    y = np.zeros(length, dtype=np.float32)
    a1 = rng.uniform(0.2, 0.6)
    a2 = rng.uniform(-0.3, 0.1)
    gain = rng.uniform(0.15, 0.45)
    for t in range(length):
        y_tm1 = y[t - 1] if t - 1 >= 0 else 0.0
        y_tm2 = y[t - 2] if t - 2 >= 0 else 0.0
        x_t = x[t]
        x_tm1 = x[t - 1] if t - 1 >= 0 else 0.0
        nonlin = gain * (np.log1p(np.exp(x_t + 0.5 * x_tm1)) - np.log(2.0))
        cross = 0.08 * np.tanh(y_tm1 * x_t)
        y[t] = a1 * y_tm1 + a2 * y_tm2 + nonlin + cross
    return x, y


def _simulate_iir_filter_bank(length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = _stable_input_signal(length, rng)
    y = x.copy()
    num_sections = 2
    for _ in range(num_sections):
        theta = rng.uniform(0.1, np.pi - 0.1)
        radius = rng.uniform(0.65, 0.95)
        a1 = -2 * radius * np.cos(theta)
        a2 = radius**2
        b0 = rng.uniform(0.1, 0.8)
        b1 = rng.uniform(-0.6, 0.6)
        b2 = rng.uniform(-0.4, 0.4)
        out = np.zeros(length, dtype=np.float32)
        for t in range(length):
            x0 = y[t]
            x1 = y[t - 1] if t - 1 >= 0 else 0.0
            x2 = y[t - 2] if t - 2 >= 0 else 0.0
            y1 = out[t - 1] if t - 1 >= 0 else 0.0
            y2 = out[t - 2] if t - 2 >= 0 else 0.0
            out[t] = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y = out
    return x, y


def _simulate_dss(length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = _stable_input_signal(length, rng)
    state_dim = 6
    eigs = rng.uniform(0.45, 0.95, size=state_dim).astype(np.float32)
    q, _ = np.linalg.qr(rng.standard_normal((state_dim, state_dim)))
    A = q @ np.diag(eigs) @ np.linalg.inv(q)
    B = rng.uniform(-0.6, 0.6, size=(state_dim, 1)).astype(np.float32)
    C = rng.uniform(-0.6, 0.6, size=(1, state_dim)).astype(np.float32)
    D = rng.uniform(-0.2, 0.2)
    state = np.zeros((state_dim,), dtype=np.float32)
    y = np.zeros(length, dtype=np.float32)
    for t in range(length):
        state = A @ state + (B[:, 0] * x[t])
        y[t] = float((C @ state)[0] + D * x[t])
    return x, y


def _simulate_resonant_arma_near_unit_circle(
    length: int,
    rng: np.random.Generator,
    prompt_len: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    forcing = _stable_input_signal(length - prompt_len, rng)
    radius = rng.uniform(0.9, 0.995)
    phase = rng.uniform(0.05 * np.pi, 0.45 * np.pi)
    a1 = 2.0 * radius * np.cos(phase)
    a2 = -(radius**2)
    b = rng.uniform(-0.18, 0.18, size=4).astype(np.float32)
    b[0] += np.sign(b[0]) * 0.35 if b[0] != 0 else 0.35
    prompt = _parameter_prompt(
        np.array(
            [
                2.0 * radius - 1.0,
                phase / np.pi,
                *b,
            ],
            dtype=np.float32,
        ),
        prompt_len,
    )
    x = np.concatenate([prompt, forcing], axis=0).astype(np.float32)
    y = np.zeros(length, dtype=np.float32)
    for t in range(prompt_len, length):
        acc = 0.0
        if t - 1 >= prompt_len:
            acc += a1 * y[t - 1]
        if t - 2 >= prompt_len:
            acc += a2 * y[t - 2]
        for j in range(4):
            idx = t - j
            if idx >= prompt_len:
                acc += b[j] * x[idx]
        y[t] = acc
    return x, y


def _simulate_nonlinear_stable_narx_latent(
    length: int,
    rng: np.random.Generator,
    prompt_len: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    forcing = _stable_input_signal(length - prompt_len, rng)
    radii = rng.uniform(0.85, 0.98, size=2).astype(np.float32)
    gains = rng.uniform(0.15, 0.45, size=2).astype(np.float32)
    readout = rng.uniform(-0.6, 0.6, size=2).astype(np.float32)
    nonlinear_gain = float(rng.uniform(0.08, 0.22))
    prompt = _parameter_prompt(
        np.array(
            [
                *radii,
                *gains,
                *readout,
                nonlinear_gain,
            ],
            dtype=np.float32,
        ),
        prompt_len,
    )
    x = np.concatenate([prompt, forcing], axis=0).astype(np.float32)
    y = np.zeros(length, dtype=np.float32)
    h = np.zeros(2, dtype=np.float32)
    for t in range(prompt_len, length):
        u_t = x[t]
        u_tm1 = x[t - 1] if t - 1 >= prompt_len else 0.0
        h[0] = radii[0] * h[0] + gains[0] * u_t
        h[1] = radii[1] * h[1] + gains[1] * (u_tm1 + 0.3 * h[0])
        linear = float(np.dot(readout, h))
        nonlinear = nonlinear_gain * np.tanh(h[0] * h[1]) + 0.08 * np.log1p(np.exp(u_t))
        y[t] = linear + nonlinear
    return x, y


def _simulate_step_aware_variable_dt_family(
    length: int,
    rng: np.random.Generator,
    dt_range: tuple[float, float],
    prompt_len: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    forcing = _stable_input_signal(length - prompt_len, rng)
    dt = float(rng.uniform(*dt_range))
    sigma = float(rng.uniform(-0.08, -0.01))
    omega = float(rng.uniform(0.35, 1.75))
    pole_radius = float(np.exp(sigma * dt))
    pole_phase = float(omega * dt)
    a1 = 2.0 * pole_radius * np.cos(pole_phase)
    a2 = -(pole_radius**2)
    b = rng.uniform(-0.12, 0.12, size=3).astype(np.float32)
    b[0] += np.sign(b[0]) * 0.28 if b[0] != 0 else 0.28
    prompt = _parameter_prompt(
        np.array(
            [
                dt,
                sigma,
                omega,
                *b,
            ],
            dtype=np.float32,
        ),
        prompt_len,
    )
    x = np.concatenate([prompt, forcing], axis=0).astype(np.float32)
    y = np.zeros(length, dtype=np.float32)
    for t in range(prompt_len, length):
        acc = 0.0
        if t - 1 >= prompt_len:
            acc += a1 * y[t - 1]
        if t - 2 >= prompt_len:
            acc += a2 * y[t - 2]
        for j in range(3):
            idx = t - j
            if idx >= prompt_len:
                acc += b[j] * x[idx]
        y[t] = acc
    return x, y


def _simulate_resonant_arma_near_unit_circle_v4(
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    return _simulate_resonant_arma_conditioned_v4(
        length,
        rng,
        radius_range=(0.9, 0.995),
        phase_range=(0.05 * np.pi, 0.45 * np.pi),
    )


def _simulate_resonant_arma_conditioned_v4(
    length: int,
    rng: np.random.Generator,
    radius_range: tuple[float, float],
    phase_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    forcing = _stable_input_signal(length, rng)
    radius = rng.uniform(*radius_range)
    phase = rng.uniform(*phase_range)
    a1 = 2.0 * radius * np.cos(phase)
    a2 = -(radius**2)
    b = rng.uniform(-0.18, 0.18, size=4).astype(np.float32)
    b[0] += np.sign(b[0]) * 0.35 if b[0] != 0 else 0.35
    x = _stack_conditioned_input(
        forcing,
        np.array([radius, phase / np.pi, *b], dtype=np.float32),
    )
    y = np.zeros(length, dtype=np.float32)
    for t in range(length):
        acc = 0.0
        if t - 1 >= 0:
            acc += a1 * y[t - 1]
        if t - 2 >= 0:
            acc += a2 * y[t - 2]
        for j in range(4):
            idx = t - j
            if idx >= 0:
                acc += b[j] * forcing[idx]
        y[t] = acc
    return x, y


NEAR_UNIT_CIRCLE_DIFFICULTY_SPECS = {
    "resonant_arma_lm_r0900_0930_v1": {
        "radius_range": (0.90, 0.93),
        "phase_range": (0.05 * np.pi, 0.45 * np.pi),
        "difficulty_index": 0,
    },
    "resonant_arma_lm_r0930_0960_v1": {
        "radius_range": (0.93, 0.96),
        "phase_range": (0.05 * np.pi, 0.45 * np.pi),
        "difficulty_index": 1,
    },
    "resonant_arma_lm_r0960_0980_v1": {
        "radius_range": (0.96, 0.98),
        "phase_range": (0.05 * np.pi, 0.45 * np.pi),
        "difficulty_index": 2,
    },
    "resonant_arma_lm_r0980_0990_v1": {
        "radius_range": (0.98, 0.99),
        "phase_range": (0.05 * np.pi, 0.45 * np.pi),
        "difficulty_index": 3,
    },
    "resonant_arma_lm_r0990_0995_v1": {
        "radius_range": (0.99, 0.995),
        "phase_range": (0.05 * np.pi, 0.45 * np.pi),
        "difficulty_index": 4,
    },
}


def _simulate_nonlinear_stable_narx_latent_v4(
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    forcing = _stable_input_signal(length, rng)
    radii = rng.uniform(0.85, 0.98, size=2).astype(np.float32)
    gains = rng.uniform(0.15, 0.45, size=2).astype(np.float32)
    readout = rng.uniform(-0.6, 0.6, size=2).astype(np.float32)
    nonlinear_gain = float(rng.uniform(0.08, 0.22))
    x = _stack_conditioned_input(
        forcing,
        np.array([*radii, *gains, *readout, nonlinear_gain], dtype=np.float32),
    )
    y = np.zeros(length, dtype=np.float32)
    h = np.zeros(2, dtype=np.float32)
    for t in range(length):
        u_t = forcing[t]
        u_tm1 = forcing[t - 1] if t - 1 >= 0 else 0.0
        h[0] = radii[0] * h[0] + gains[0] * u_t
        h[1] = radii[1] * h[1] + gains[1] * (u_tm1 + 0.3 * h[0])
        linear = float(np.dot(readout, h))
        nonlinear = nonlinear_gain * np.tanh(h[0] * h[1]) + 0.08 * np.log1p(np.exp(u_t))
        y[t] = linear + nonlinear
    return x, y


def _simulate_step_aware_variable_dt_family_v4(
    length: int,
    rng: np.random.Generator,
    dt_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    forcing = _stable_input_signal(length, rng)
    dt = float(rng.uniform(*dt_range))
    sigma = float(rng.uniform(-0.08, -0.01))
    omega = float(rng.uniform(0.35, 1.75))
    pole_radius = float(np.exp(sigma * dt))
    pole_phase = float(omega * dt)
    a1 = 2.0 * pole_radius * np.cos(pole_phase)
    a2 = -(pole_radius**2)
    b = rng.uniform(-0.12, 0.12, size=3).astype(np.float32)
    b[0] += np.sign(b[0]) * 0.28 if b[0] != 0 else 0.28
    x = _stack_conditioned_input(
        forcing,
        np.array([dt, sigma, omega, *b], dtype=np.float32),
    )
    y = np.zeros(length, dtype=np.float32)
    for t in range(length):
        acc = 0.0
        if t - 1 >= 0:
            acc += a1 * y[t - 1]
        if t - 2 >= 0:
            acc += a2 * y[t - 2]
        for j in range(3):
            idx = t - j
            if idx >= 0:
                acc += b[j] * forcing[idx]
        y[t] = acc
    return x, y


def _simulate_high_order_iir_cascade_v5(
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    forcing = _stable_input_signal(length, rng)
    num_sections = 3
    params = []
    section_inputs = forcing.astype(np.float32)
    for _ in range(num_sections):
        radius = float(rng.uniform(0.88, 0.995))
        phase = float(rng.uniform(0.04 * np.pi, 0.35 * np.pi))
        a1 = 2.0 * radius * np.cos(phase)
        a2 = -(radius**2)
        b0 = float(rng.uniform(0.18, 0.85))
        b1 = float(rng.uniform(-0.35, 0.35))
        b2 = float(rng.uniform(-0.2, 0.2))
        params.extend([radius, phase / np.pi, b0, b1, b2])
        out = np.zeros(length, dtype=np.float32)
        for t in range(length):
            acc = 0.0
            if t - 1 >= 0:
                acc += a1 * out[t - 1]
            if t - 2 >= 0:
                acc += a2 * out[t - 2]
            x0 = section_inputs[t]
            x1 = section_inputs[t - 1] if t - 1 >= 0 else 0.0
            x2 = section_inputs[t - 2] if t - 2 >= 0 else 0.0
            acc += b0 * x0 + b1 * x1 + b2 * x2
            out[t] = acc
        section_inputs = out
    x = _stack_conditioned_input(forcing, np.array(params, dtype=np.float32))
    y = section_inputs.astype(np.float32)
    return x, y


def _simulate_digital_pid_controlled_plant_v5(
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
    reference = (
        0.8 * np.sign(np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t + rng.uniform(0, 2 * np.pi)))
        + 0.35 * np.sin(2 * np.pi * rng.uniform(2.0, 5.0) * t + rng.uniform(0, 2 * np.pi))
        + 0.1 * rng.standard_normal(length)
    ).astype(np.float32)
    radius = float(rng.uniform(0.82, 0.97))
    phase = float(rng.uniform(0.05 * np.pi, 0.3 * np.pi))
    a1 = 2.0 * radius * np.cos(phase)
    a2 = -(radius**2)
    b0 = float(rng.uniform(0.04, 0.18))
    b1 = float(rng.uniform(0.0, 0.12))
    kp = float(rng.uniform(0.4, 1.4))
    ki = float(rng.uniform(0.01, 0.12))
    kd = float(rng.uniform(0.0, 0.2))
    anti_windup = float(rng.uniform(1.5, 3.5))

    x = _stack_conditioned_input(
        reference,
        np.array([radius, phase / np.pi, b0, b1, kp, ki, kd, anti_windup], dtype=np.float32),
    )

    y = np.zeros(length, dtype=np.float32)
    control = np.zeros(length, dtype=np.float32)
    integral = 0.0
    prev_error = 0.0
    for t_idx in range(length):
        y_prev = y[t_idx - 1] if t_idx - 1 >= 0 else 0.0
        error = reference[t_idx] - y_prev
        integral = float(np.clip(integral + error, -anti_windup, anti_windup))
        derivative = error - prev_error
        prev_error = error
        control[t_idx] = kp * error + ki * integral + kd * derivative

        plant = 0.0
        if t_idx - 1 >= 0:
            plant += a1 * y[t_idx - 1]
            plant += b1 * control[t_idx - 1]
        if t_idx - 2 >= 0:
            plant += a2 * y[t_idx - 2]
        plant += b0 * control[t_idx]
        y[t_idx] = plant
    return x, y


SUITES = {
    "arma_stable": _simulate_arma,
    "narx_softplus": _simulate_narx,
    "iir_filter_bank": _simulate_iir_filter_bank,
    "dss_surrogate": _simulate_dss,
    "resonant_arma_near_unit_circle": _simulate_resonant_arma_near_unit_circle,
    "nonlinear_stable_narx_latent": _simulate_nonlinear_stable_narx_latent,
    "step_aware_variable_dt_family": lambda length, rng: _simulate_step_aware_variable_dt_family(
        length, rng, dt_range=(0.8, 1.2)
    ),
    "resonant_arma_near_unit_circle_v4": _simulate_resonant_arma_near_unit_circle_v4,
    "nonlinear_stable_narx_latent_v4": _simulate_nonlinear_stable_narx_latent_v4,
    "step_aware_variable_dt_family_v4": lambda length, rng: _simulate_step_aware_variable_dt_family_v4(
        length, rng, dt_range=(0.8, 1.2)
    ),
    "high_order_iir_cascade_v5": _simulate_high_order_iir_cascade_v5,
    "digital_pid_controlled_plant_v5": _simulate_digital_pid_controlled_plant_v5,
}
for suite_name, spec in NEAR_UNIT_CIRCLE_DIFFICULTY_SPECS.items():
    SUITES[suite_name] = (
        lambda length, rng, spec=spec: _simulate_resonant_arma_conditioned_v4(
            length,
            rng,
            radius_range=spec["radius_range"],
            phase_range=spec["phase_range"],
        )
    )


def _build_split(sim_fn, count: int, length: int, seed: int):
    rng = np.random.default_rng(seed)
    first_x, first_y = sim_fn(length, rng)
    first_x = np.asarray(first_x, dtype=np.float32)
    if first_x.ndim == 1:
        first_x = first_x[:, None]
    first_y = np.asarray(first_y, dtype=np.float32).reshape(length, 1)
    xs = np.zeros((count, length, first_x.shape[1]), dtype=np.float32)
    ys = np.zeros((count, length, 1), dtype=np.float32)
    xs[0] = first_x
    ys[0] = first_y
    for idx in range(1, count):
        x, y = sim_fn(length, rng)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        xs[idx] = x
        ys[idx, :, 0] = y
    return xs, ys


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ZNO-native discrete-time benchmark suites.")
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--output-dir", default="data/controlled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--train-sequence-length", type=int)
    parser.add_argument("--val-sequence-length", type=int)
    parser.add_argument("--test-sequence-length", type=int)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve() / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_fn = SUITES[args.suite]
    train_length = args.train_sequence_length or args.sequence_length
    val_length = args.val_sequence_length or args.sequence_length
    test_length = args.test_sequence_length or args.sequence_length
    ready_file = out_dir / "_READY"
    if ready_file.exists():
        ready_file.unlink()
    if not args.overwrite and (out_dir / "data.npz").exists():
        print(f"Suite already exists, skip regenerate: {out_dir}")
        ready_file.write_text("ready\n", encoding="utf-8")
        return

    x_train, y_train = _build_split(sim_fn, args.train_size, train_length, args.seed)
    x_val, y_val = _build_split(sim_fn, args.val_size, val_length, args.seed + 1)
    if args.suite == "step_aware_variable_dt_family":
        test_sim_fn = lambda length, rng: _simulate_step_aware_variable_dt_family(
            length, rng, dt_range=(0.25, 0.5) if rng.uniform() < 0.5 else (1.5, 2.25)
        )
        x_test, y_test = _build_split(test_sim_fn, args.test_size, test_length, args.seed + 2)
    elif args.suite == "step_aware_variable_dt_family_v4":
        test_sim_fn = lambda length, rng: _simulate_step_aware_variable_dt_family_v4(
            length, rng, dt_range=(0.25, 0.5) if rng.uniform() < 0.5 else (1.5, 2.25)
        )
        x_test, y_test = _build_split(test_sim_fn, args.test_size, test_length, args.seed + 2)
    else:
        x_test, y_test = _build_split(sim_fn, args.test_size, test_length, args.seed + 2)
    grid = np.arange(test_length, dtype=np.float32)

    tmp_data_path = out_dir / "data.tmp.npz"
    np.savez_compressed(
        tmp_data_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        grid=grid,
    )
    tmp_data_path.replace(out_dir / "data.npz")

    meta = {
        "suite": args.suite,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "train_sequence_length": train_length,
        "val_sequence_length": val_length,
        "test_sequence_length": test_length,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "output_dir": str(out_dir),
    }
    if args.suite in {"step_aware_variable_dt_family", "step_aware_variable_dt_family_v4"}:
        meta["train_dt_range"] = [0.8, 1.2]
        meta["test_dt_ranges"] = [[0.25, 0.5], [1.5, 2.25]]
    if args.suite in NEAR_UNIT_CIRCLE_DIFFICULTY_SPECS:
        spec = NEAR_UNIT_CIRCLE_DIFFICULTY_SPECS[args.suite]
        radius_lo, radius_hi = spec["radius_range"]
        phase_lo, phase_hi = spec["phase_range"]
        meta["radius_range"] = [radius_lo, radius_hi]
        meta["phase_range_over_pi"] = [phase_lo / np.pi, phase_hi / np.pi]
        meta["memory_horizon_range"] = [1.0 / (1.0 - radius_lo), 1.0 / (1.0 - radius_hi)]
        meta["difficulty_index"] = spec["difficulty_index"]
    (out_dir / "meta.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
    ready_file.write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved discrete suite to: {out_dir}")


if __name__ == "__main__":
    main()
