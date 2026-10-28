import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import OperatorConfig
from .fused_recurrent import fused_recurrent_scan
from .triton_recurrent import triton_recurrent_available, triton_recurrent_scan
from .triton_scan_fast import triton_fast_available, triton_fast_recurrent_scan

try:
    from neuralop.models import FNO as NeuralOpFNO
    NEURALOP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    NeuralOpFNO = None
    NEURALOP_IMPORT_ERROR = exc


def apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "sin":
        return torch.sin(x)
    if activation == "tanh":
        return torch.tanh(x)
    if activation == "relu":
        return F.relu(x)
    if activation == "gelu":
        return F.gelu(x)
    raise ValueError(f"Unsupported activation: {activation}")


def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


class InputGridMixin:
    def _append_grid(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_grid:
            return x
        grid = self.grid.view(1, -1, 1).expand(x.size(0), -1, -1)
        return torch.cat((x, grid), dim=-1)


class PoleResidue1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, grid: torch.Tensor):
        super().__init__()
        self.modes = modes
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        scale = 1 / (in_channels * out_channels)
        self.weights_pole = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        self.weights_residue = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.grid.to(x.device)
        dt = (t[1] - t[0]).item()
        alpha = torch.fft.fft(x, dim=-1)
        lambdas = torch.fft.fftfreq(t.numel(), d=dt).to(x.device) * (2j * math.pi)
        kernel = self.weights_residue.unsqueeze(0) / (
            lambdas[:, None, None, None] - self.weights_pole.unsqueeze(0)
        )
        steady_residue = torch.einsum("bit,tiom->bot", alpha, kernel)
        transient_residue = torch.einsum("bit,tiom->bom", alpha, -kernel)

        steady = torch.fft.ifft(steady_residue, n=x.size(-1), dim=-1).real
        exp_term = torch.exp(
            self.weights_pole.unsqueeze(-1) * t.to(torch.cfloat).view(1, 1, 1, -1)
        )
        transient = torch.einsum("bom,iomt->bot", transient_residue, exp_term).real
        return steady + transient / x.size(-1)


class StablePoleResidue1d(nn.Module):
    """The LNO pole-residue transform with poles restricted to the left half-plane."""

    def __init__(self, in_channels: int, out_channels: int, modes: int, grid: torch.Tensor):
        super().__init__()
        self.modes = modes
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        scale = 1 / (in_channels * out_channels)
        decay_init = scale * torch.rand(in_channels, out_channels, modes) + 1e-4
        self.raw_decay = nn.Parameter(_softplus_inverse(decay_init))
        self.frequency = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes)
        )
        self.weights_residue = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def poles(self) -> torch.Tensor:
        return torch.complex(-F.softplus(self.raw_decay), self.frequency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.grid.to(x.device)
        dt = (t[1] - t[0]).item()
        poles = self.poles()
        alpha = torch.fft.fft(x, dim=-1)
        lambdas = torch.fft.fftfreq(t.numel(), d=dt).to(x.device) * (2j * math.pi)
        kernel = self.weights_residue.unsqueeze(0) / (
            lambdas[:, None, None, None] - poles.unsqueeze(0)
        )
        steady_residue = torch.einsum("bit,tiom->bot", alpha, kernel)
        transient_residue = torch.einsum("bit,tiom->bom", alpha, -kernel)

        steady = torch.fft.ifft(steady_residue, n=x.size(-1), dim=-1).real
        exp_term = torch.exp(
            poles.unsqueeze(-1) * t.to(torch.cfloat).view(1, 1, 1, -1)
        )
        transient = torch.einsum("bom,iomt->bot", transient_residue, exp_term).real
        return steady + transient / x.size(-1)


class LNO1d(nn.Module, InputGridMixin):
    def __init__(self, config: OperatorConfig, grid: torch.Tensor, use_grid: bool):
        super().__init__()
        self.use_grid = use_grid
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        normalized_grid = torch.linspace(0.0, 1.0, steps=grid.numel(), dtype=torch.float32)
        self.register_buffer("input_grid", normalized_grid, persistent=False)
        input_dim = 1 + int(use_grid)
        self.activation = config.activation
        self.fc0 = nn.Linear(input_dim, config.width)
        self.layers = nn.ModuleList(
            [PoleResidue1d(config.width, config.width, config.modes, grid) for _ in range(config.depth)]
        )
        self.skips = nn.ModuleList(
            [nn.Conv1d(config.width, config.width, 1) for _ in range(config.depth)]
        )
        self.fc1 = nn.Linear(config.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grid:
            grid = self.input_grid.view(1, -1, 1).expand(x.size(0), -1, -1)
            x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        for index, (layer, skip) in enumerate(zip(self.layers, self.skips)):
            x = layer(x) + skip(x)
            if index < len(self.layers) - 1:
                x = apply_activation(x, self.activation)
        x = x.permute(0, 2, 1)
        x = apply_activation(self.fc1(x), self.activation)
        return self.fc2(x)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x, dim=-1)
        usable_modes = min(self.modes, x_ft.size(-1))
        out_ft = torch.zeros(
            x.size(0),
            self.weights.size(1),
            x_ft.size(-1),
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :usable_modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :usable_modes], self.weights[:, :, :usable_modes]
        )
        return torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)


def package_activation(activation: str):
    if activation == "sin":
        return torch.sin
    if activation == "tanh":
        return torch.tanh
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {activation}")


class FNO1d(nn.Module, InputGridMixin):
    def __init__(self, config: OperatorConfig, grid: torch.Tensor, use_grid: bool):
        super().__init__()
        if NeuralOpFNO is None:
            raise ImportError(
                "FNO baseline now uses the official neuraloperator package. "
                "Install/fix it in the active environment with `pip install neuraloperator`. "
                f"Original import error: {NEURALOP_IMPORT_ERROR}"
            )

        self.use_grid = use_grid
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        self.model = NeuralOpFNO(
            n_modes=(config.modes,),
            in_channels=1,
            out_channels=1,
            hidden_channels=config.width,
            n_layers=config.depth,
            positional_embedding="grid" if use_grid else None,
            non_linearity=package_activation(config.activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.model(x)
        return x.permute(0, 2, 1)


class LowRankRationalLayer1d(nn.Module):
    def __init__(
        self,
        channels: int,
        rank: int,
        num_poles: int,
        fir_order: int,
        radius_limit: float,
        fused_backward_mode: str,
        kernel_backend: str,
        pole_parameterization: str = "z",
        dt: float = 1.0,
        pole_init_radius_low: float | None = None,
        pole_init_radius_high: float | None = None,
    ):
        super().__init__()
        if num_poles < 1:
            raise ValueError("num_poles must be positive")
        if rank < 1:
            raise ValueError("rank must be positive")

        self.channels = channels
        self.rank = rank
        self.num_poles = num_poles
        self.fir_order = fir_order
        self.radius_limit = radius_limit
        self.fused_backward_mode = fused_backward_mode
        self.kernel_backend = kernel_backend
        self.pole_parameterization = pole_parameterization
        self.dt = float(dt)
        self.num_pairs = num_poles // 2
        self.has_real_pole = num_poles % 2

        scale = 1 / max(channels, rank)
        self.in_proj = nn.Linear(channels, rank, bias=False)
        self.out_proj = nn.Linear(rank, channels, bias=False)
        self.skip = nn.Linear(channels, channels)
        init_radius_low = (
            float(pole_init_radius_low)
            if pole_init_radius_low is not None
            else (0.3 if radius_limit > 0.3 else max(radius_limit * 0.25, 1e-3))
        )
        init_radius_high = (
            float(pole_init_radius_high)
            if pole_init_radius_high is not None
            else min(0.95, radius_limit - 1e-3)
        )
        if init_radius_high <= init_radius_low:
            if pole_init_radius_low is not None or pole_init_radius_high is not None:
                raise ValueError("pole_init_radius_high must exceed pole_init_radius_low")
            init_radius_low = max(radius_limit * 0.3, 1e-3)
            init_radius_high = max(
                radius_limit - 1e-3, init_radius_low + 1e-3
            )
        if pole_parameterization == "z" and init_radius_high >= radius_limit:
            raise ValueError(
                "Bounded z-pole initialization must lie below radius_limit"
            )

        if pole_parameterization not in {"z", "z_unbounded", "s_exp"}:
            raise ValueError(f"Unsupported pole_parameterization: {pole_parameterization}")

        if self.num_pairs:
            pair_radius_init = torch.empty(rank, self.num_pairs).uniform_(init_radius_low, init_radius_high)
            pair_phase_init = torch.empty(rank, self.num_pairs).uniform_(1e-3, math.pi - 1e-3)
            if self.pole_parameterization == "z":
                self.pair_radius = nn.Parameter(torch.logit(pair_radius_init / radius_limit))
                self.pair_phase = nn.Parameter(pair_phase_init)
                self.register_parameter("pair_log_decay", None)
                self.register_parameter("pair_frequency", None)
            elif self.pole_parameterization == "z_unbounded":
                self.pair_radius = nn.Parameter(_softplus_inverse(pair_radius_init))
                self.pair_phase = nn.Parameter(pair_phase_init)
                self.register_parameter("pair_log_decay", None)
                self.register_parameter("pair_frequency", None)
            else:
                pair_decay_init = (-torch.log(pair_radius_init)).clamp_min(1e-4) / max(self.dt, 1e-6)
                self.register_parameter("pair_radius", None)
                self.register_parameter("pair_phase", None)
                self.pair_log_decay = nn.Parameter(_softplus_inverse(pair_decay_init))
                self.pair_frequency = nn.Parameter(pair_phase_init / max(self.dt, 1e-6))
            self.pair_residue_real = nn.Parameter(scale * torch.randn(rank, self.num_pairs))
            self.pair_residue_imag = nn.Parameter(scale * torch.randn(rank, self.num_pairs))
        else:
            self.register_parameter("pair_radius", None)
            self.register_parameter("pair_phase", None)
            self.register_parameter("pair_log_decay", None)
            self.register_parameter("pair_frequency", None)
            self.register_parameter("pair_residue_real", None)
            self.register_parameter("pair_residue_imag", None)

        if self.has_real_pole:
            real_radius_init = torch.empty(rank, 1).uniform_(init_radius_low, init_radius_high)
            if self.pole_parameterization == "z":
                self.real_radius = nn.Parameter(torch.logit(real_radius_init / radius_limit))
                self.register_parameter("real_log_decay", None)
            elif self.pole_parameterization == "z_unbounded":
                self.real_radius = nn.Parameter(_softplus_inverse(real_radius_init))
                self.register_parameter("real_log_decay", None)
            else:
                real_decay_init = (-torch.log(real_radius_init)).clamp_min(1e-4) / max(self.dt, 1e-6)
                self.register_parameter("real_radius", None)
                self.real_log_decay = nn.Parameter(_softplus_inverse(real_decay_init))
            self.real_residue = nn.Parameter(scale * torch.randn(rank, 1))
        else:
            self.register_parameter("real_radius", None)
            self.register_parameter("real_log_decay", None)
            self.register_parameter("real_residue", None)

        if fir_order > 0:
            self.fir_weights = nn.Parameter(scale * torch.randn(rank, fir_order + 1))
        else:
            self.register_parameter("fir_weights", None)

    def _pole_terms(self):
        """Pair poles/residues and the real pole/residue (None when absent)."""
        pair_poles = pair_residues = real_pole = real_residue = None
        if self.num_pairs:
            if self.pole_parameterization == "z":
                radius = self.radius_limit * torch.sigmoid(self.pair_radius)
                pair_poles = torch.polar(radius, self.pair_phase)
            elif self.pole_parameterization == "z_unbounded":
                radius = F.softplus(self.pair_radius)
                pair_poles = torch.polar(radius, self.pair_phase)
            else:
                decay = F.softplus(self.pair_log_decay)
                radius = torch.exp(-decay * self.dt)
                pair_poles = torch.polar(radius, self.pair_frequency * self.dt)
            pair_residues = torch.complex(self.pair_residue_real, self.pair_residue_imag)
        if self.has_real_pole:
            if self.pole_parameterization == "z":
                real_radius = self.radius_limit * torch.sigmoid(self.real_radius)
            elif self.pole_parameterization == "z_unbounded":
                real_radius = F.softplus(self.real_radius)
            else:
                real_decay = F.softplus(self.real_log_decay)
                real_radius = torch.exp(-real_decay * self.dt)
            real_pole = torch.complex(real_radius, torch.zeros_like(real_radius))
            real_residue = torch.complex(self.real_residue, torch.zeros_like(self.real_residue))
        return pair_poles, pair_residues, real_pole, real_residue

    def stable_poles_and_residues(self):
        pair_poles, pair_residues, real_pole, real_residue = self._pole_terms()
        poles = []
        residues = []
        if pair_poles is not None:
            poles.extend([pair_poles, pair_poles.conj()])
            residues.extend([pair_residues, pair_residues.conj()])
        if real_pole is not None:
            poles.append(real_pole)
            residues.append(real_residue)
        return torch.cat(poles, dim=-1), torch.cat(residues, dim=-1)

    def lane_poles_and_residues(self):
        """One lane per conjugate pair followed by the real pole: the input of the triton_fast scan."""
        pair_poles, pair_residues, real_pole, real_residue = self._pole_terms()
        if real_pole is None:
            return pair_poles, pair_residues
        if pair_poles is None:
            return real_pole, real_residue
        return torch.cat((pair_poles, real_pole), dim=-1), torch.cat((pair_residues, real_residue), dim=-1)

    def init_state(self, batch_size: int, device: torch.device):
        pole_state = torch.zeros(
            batch_size,
            self.rank,
            self.num_poles,
            dtype=torch.cfloat,
            device=device,
        )
        fir_state = torch.zeros(
            batch_size,
            self.rank,
            self.fir_order + 1 if self.fir_order > 0 else 0,
            device=device,
        )
        return pole_state, fir_state

    def scan_params(self):
        if self.kernel_backend == "triton_fast":
            # Lane layout (no conjugate copies); scan_sequence rebuilds the full layout on CPU.
            poles, residues = self.lane_poles_and_residues()
        else:
            poles, residues = self.stable_poles_and_residues()
        input_weight = torch.cat((self.in_proj.weight, self.skip.weight), dim=0)
        input_bias = torch.cat(
            (
                self.skip.bias.new_zeros(self.rank),
                self.skip.bias,
            )
        )
        fir_weights = self.fir_weights.unsqueeze(0) if self.fir_order > 0 else None
        return poles.unsqueeze(0), residues.unsqueeze(0), fir_weights, input_weight, input_bias

    def scan_sequence(self, x: torch.Tensor, scan_params) -> torch.Tensor:
        poles, residues, fir_weights, input_weight, input_bias = scan_params
        projected_and_skip = F.linear(x, input_weight, input_bias)
        projected = projected_and_skip[..., : self.rank]
        skip_sequence = projected_and_skip[..., self.rank :]
        if self.kernel_backend == "triton_fast" and not projected.is_cuda:
            full_poles, full_residues = self.stable_poles_and_residues()
            poles, residues = full_poles.unsqueeze(0), full_residues.unsqueeze(0)
        if projected.is_cuda:
            if self.kernel_backend == "triton_fast":
                if not triton_fast_available():
                    raise RuntimeError("triton_fast backend requested but Triton is not available.")
                filtered_sequence = triton_fast_recurrent_scan(
                    projected,
                    poles.squeeze(0),
                    residues.squeeze(0),
                    self.fir_weights,
                    num_pairs=self.num_pairs,
                )
            elif self.kernel_backend == "triton":
                if not triton_recurrent_available():
                    raise RuntimeError("Triton backend requested but Triton is not available.")
                filtered_sequence = triton_recurrent_scan(
                    projected,
                    poles.squeeze(0),
                    residues.squeeze(0),
                    self.fir_weights,
                )
            else:
                filtered_sequence = fused_recurrent_scan(
                    projected,
                    poles.squeeze(0),
                    residues.squeeze(0),
                    self.fir_weights,
                    backward_mode=self.fused_backward_mode,
                )
        else:
            pole_state, fir_state = self.init_state(x.size(0), x.device)
            filtered_sequence = projected.new_empty(projected.shape)

            for step in range(projected.size(1)):
                projected_t = projected[:, step, :]
                pole_state = pole_state * poles + projected_t.to(torch.cfloat).unsqueeze(-1)
                filtered_t = (pole_state * residues).sum(dim=-1).real

                if self.fir_order > 0:
                    next_fir_state = torch.empty_like(fir_state)
                    next_fir_state[..., 0] = projected_t
                    next_fir_state[..., 1:] = fir_state[..., : self.fir_order]
                    fir_state = next_fir_state
                    filtered_t = filtered_t + (fir_state * fir_weights).sum(dim=-1)

                filtered_sequence[:, step, :] = filtered_t

        return self.out_proj(filtered_sequence) + skip_sequence

    def pole_penalty(self, safe_radius: float) -> torch.Tensor:
        poles, _ = self.stable_poles_and_residues()
        return F.relu(torch.abs(poles) - safe_radius).pow(2).mean()

    def max_pole_radius(self) -> float:
        poles, _ = self.stable_poles_and_residues()
        return float(torch.abs(poles).max().detach().cpu().item())

    def mean_pole_radius(self) -> float:
        poles, _ = self.stable_poles_and_residues()
        return float(torch.abs(poles).mean().detach().cpu().item())


class ZNO1d(nn.Module, InputGridMixin):
    def __init__(self, config: OperatorConfig, grid: torch.Tensor, use_grid: bool):
        super().__init__()
        self.use_grid = use_grid
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        input_dim = 1 + int(use_grid)
        self.activation = config.activation
        self.safe_radius = config.safe_radius
        self.rank = config.rank if config.rank is not None else config.width
        self.fc0 = nn.Linear(input_dim, config.width)
        self.layers = nn.ModuleList(
            [
                LowRankRationalLayer1d(
                    channels=config.width,
                    rank=self.rank,
                    num_poles=config.num_poles,
                    fir_order=config.fir_order,
                    radius_limit=config.radius_limit,
                    fused_backward_mode=config.fused_backward_mode,
                    kernel_backend=config.kernel_backend,
                )
                for _ in range(config.depth)
            ]
        )
        self.fc1 = nn.Linear(config.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def _apply_output_head(self, hidden_sequence: torch.Tensor) -> torch.Tensor:
        return self.fc2(apply_activation(self.fc1(hidden_sequence), self.activation))

    def _scan_hidden_sequence(self, x: torch.Tensor, collect_from: int = 0) -> torch.Tensor:
        x = self._append_grid(x)
        hidden_sequence = self.fc0(x)
        sequence_length = hidden_sequence.size(1)
        collect_from = max(0, min(collect_from, sequence_length - 1))
        for layer in self.layers:
            hidden_sequence = layer.scan_sequence(hidden_sequence, layer.scan_params())
            hidden_sequence = apply_activation(hidden_sequence, self.activation)
        return hidden_sequence[:, collect_from:, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_sequence = self._scan_hidden_sequence(x, collect_from=0)
        return self._apply_output_head(hidden_sequence)

    def rollout(self, x: torch.Tensor, context_steps: int) -> torch.Tensor:
        context_steps = max(0, min(context_steps, x.size(1) - 1))
        hidden_sequence = self._scan_hidden_sequence(x, collect_from=context_steps)
        return self._apply_output_head(hidden_sequence)

    def pole_penalty(self) -> torch.Tensor:
        penalties = [layer.pole_penalty(self.safe_radius) for layer in self.layers]
        if not penalties:
            return torch.tensor(0.0, device=self.fc0.weight.device)
        return torch.stack(penalties).mean()

    def stability_stats(self) -> Dict[str, float]:
        max_radius = max(layer.max_pole_radius() for layer in self.layers)
        mean_radius = sum(layer.mean_pole_radius() for layer in self.layers) / len(self.layers)
        return {
            "max_pole_radius": max_radius,
            "mean_pole_radius": mean_radius,
        }


def build_model(
    model_name: str,
    config: OperatorConfig,
    grid: torch.Tensor,
    use_grid: bool,
) -> nn.Module:
    if model_name == "lno":
        return LNO1d(config=config, grid=grid, use_grid=use_grid)
    if model_name == "fno":
        return FNO1d(config=config, grid=grid, use_grid=use_grid)
    if model_name == "zno":
        return ZNO1d(config=config, grid=grid, use_grid=use_grid)
    raise ValueError(f"Unsupported model: {model_name}")
