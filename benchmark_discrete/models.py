from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_1d.models import (
    LowRankRationalLayer1d,
    NeuralOpFNO,
    NEURALOP_IMPORT_ERROR,
    PoleResidue1d,
    StablePoleResidue1d,
    apply_activation,
    package_activation,
)

from .configs import OperatorConfig


class InputGridMixin:
    def _append_grid(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_grid:
            return x
        grid = self.grid.view(1, -1, 1).expand(x.size(0), -1, -1)
        return torch.cat((x, grid), dim=-1)


class DiscreteLNO1d(nn.Module, InputGridMixin):
    def __init__(self, config: OperatorConfig, grid: torch.Tensor, use_grid: bool, input_channels: int):
        super().__init__()
        self.use_grid = use_grid
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        normalized_grid = torch.linspace(0.0, 1.0, steps=grid.numel(), dtype=torch.float32)
        self.register_buffer("input_grid", normalized_grid, persistent=False)
        input_dim = input_channels + int(use_grid)
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


class DiscreteStableLNO1d(nn.Module, InputGridMixin):
    def __init__(self, config: OperatorConfig, grid: torch.Tensor, use_grid: bool, input_channels: int):
        super().__init__()
        self.use_grid = use_grid
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        normalized_grid = torch.linspace(0.0, 1.0, steps=grid.numel(), dtype=torch.float32)
        self.register_buffer("input_grid", normalized_grid, persistent=False)
        input_dim = input_channels + int(use_grid)
        self.activation = config.activation
        self.fc0 = nn.Linear(input_dim, config.width)
        self.layers = nn.ModuleList(
            [
                StablePoleResidue1d(config.width, config.width, config.modes, grid)
                for _ in range(config.depth)
            ]
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

    def stability_stats(self) -> Dict[str, float]:
        real_parts = torch.cat(
            [layer.poles().real.reshape(-1) for layer in self.layers]
        )
        return {
            "max_continuous_pole_real_part": float(real_parts.max().detach().cpu()),
            "mean_continuous_pole_real_part": float(real_parts.mean().detach().cpu()),
        }


class DiscreteFNO1d(nn.Module, InputGridMixin):
    def __init__(self, config: OperatorConfig, grid: torch.Tensor, use_grid: bool, input_channels: int):
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
            in_channels=input_channels,
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


class DiscreteZNO1d(nn.Module, InputGridMixin):
    def __init__(
        self,
        config: OperatorConfig,
        grid: torch.Tensor,
        use_grid: bool,
        input_channels: int,
        pole_parameterization: str = "z",
        dt: float = 1.0,
    ):
        super().__init__()
        self.use_grid = use_grid
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        input_dim = input_channels + int(use_grid)
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
                    pole_parameterization=pole_parameterization,
                    dt=dt,
                    pole_init_radius_low=config.pole_init_radius_low,
                    pole_init_radius_high=config.pole_init_radius_high,
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


class S4DLayer1d(nn.Module):
    def __init__(self, channels: int, state_dim: int):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        scale = 0.2 / max(channels, state_dim)

        pole_init = torch.empty(channels, state_dim).uniform_(-0.75, 0.75)
        self.pole_param = nn.Parameter(torch.atanh(pole_init / 0.999))
        self.b = nn.Parameter(scale * torch.randn(channels, state_dim))
        self.c = nn.Parameter(scale * torch.randn(channels, state_dim))
        self.d = nn.Parameter(torch.zeros(channels))
        self.out_proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)

    def pole_radius_stats(self) -> Dict[str, float]:
        poles = (0.999 * torch.tanh(self.pole_param)).detach().abs()
        return {
            "mean_pole_radius": float(poles.mean().item()),
            "max_pole_radius": float(poles.max().item()),
        }

    def _kernel(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        poles = (0.999 * torch.tanh(self.pole_param)).to(device=device, dtype=dtype)
        coeff = (self.b * self.c).to(device=device, dtype=dtype)
        n = torch.arange(length, device=device, dtype=dtype)
        powers = poles.unsqueeze(-1).pow(n)
        return torch.einsum("cm,cmt->ct", coeff, powers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        length = x.size(1)
        x_perm = x.permute(0, 2, 1)
        kernel = self._kernel(length, x.device, x.dtype)
        fft_size = 1
        while fft_size < 2 * length:
            fft_size *= 2
        x_ft = torch.fft.rfft(x_perm, n=fft_size, dim=-1)
        k_ft = torch.fft.rfft(kernel, n=fft_size, dim=-1).unsqueeze(0)
        y = torch.fft.irfft(x_ft * k_ft, n=fft_size, dim=-1)[..., :length]
        y = y + self.d.view(1, -1, 1) * x_perm
        return (self.out_proj(y) + x_perm).permute(0, 2, 1)


class LinearFIR1d(nn.Module, InputGridMixin):
    def __init__(self, config: OperatorConfig, grid: torch.Tensor, use_grid: bool, input_channels: int):
        super().__init__()
        self.use_grid = use_grid
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        input_dim = input_channels + int(use_grid)
        kernel_size = config.kernel_size if config.kernel_size is not None else 129
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(input_dim, 1, self.kernel_size, bias=True)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._append_grid(x)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x).transpose(1, 2)


class S4D1d(nn.Module, InputGridMixin):
    def __init__(self, config: OperatorConfig, grid: torch.Tensor, use_grid: bool, input_channels: int):
        super().__init__()
        self.use_grid = use_grid
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        input_dim = input_channels + int(use_grid)
        self.activation = config.activation
        self.fc0 = nn.Linear(input_dim, config.width)
        state_dim = config.state_dim if config.state_dim is not None else config.width
        self.layers = nn.ModuleList(
            [S4DLayer1d(config.width, state_dim=state_dim) for _ in range(config.depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(config.depth)])
        self.fc1 = nn.Linear(config.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._append_grid(x)
        x = self.fc0(x)
        for layer, norm in zip(self.layers, self.norms):
            x = apply_activation(norm(layer(x)), self.activation)
        x = apply_activation(self.fc1(x), self.activation)
        return self.fc2(x)

    def stability_stats(self) -> Dict[str, float]:
        stats = [layer.pole_radius_stats() for layer in self.layers]
        if not stats:
            return {}
        mean_radius = sum(item["mean_pole_radius"] for item in stats) / len(stats)
        max_radius = max(item["max_pole_radius"] for item in stats)
        return {
            "mean_pole_radius": mean_radius,
            "max_pole_radius": max_radius,
        }


class OfficialS4DKernel1d(nn.Module):
    """Local wrapper around the official state-spaces/s4 S4D parameterization."""

    def __init__(
        self,
        channels: int,
        state_dim: int,
        *,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        param_lr_scale: float = 0.5,
    ):
        super().__init__()
        half_state = max(1, state_dim // 2)
        log_dt = torch.rand(channels) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))) + torch.log(
            torch.tensor(dt_min)
        )
        c = torch.randn(channels, half_state, dtype=torch.cfloat)
        log_a_real = torch.log(0.5 * torch.ones(channels, half_state))
        a_imag = torch.pi * torch.arange(half_state, dtype=torch.float32).view(1, -1).repeat(channels, 1)

        self.c = nn.Parameter(torch.view_as_real(c))
        self._register_optim_parameter("log_dt", nn.Parameter(log_dt), lr_scale=param_lr_scale)
        self._register_optim_parameter("log_a_real", nn.Parameter(log_a_real), lr_scale=param_lr_scale)
        self._register_optim_parameter("a_imag", nn.Parameter(a_imag), lr_scale=param_lr_scale)

    def _register_optim_parameter(self, name: str, param: nn.Parameter, *, lr_scale: float):
        param._optim = {  # type: ignore[attr-defined]
            "lr_scale": float(lr_scale),
            "weight_decay": 0.0,
        }
        self.register_parameter(name, param)

    def pole_radius_stats(self) -> Dict[str, float]:
        with torch.no_grad():
            dt = torch.exp(self.log_dt)
            a = -torch.exp(self.log_a_real) + 1j * self.a_imag
            poles = torch.exp(a * dt.unsqueeze(-1)).abs()
            return {
                "mean_pole_radius": float(poles.mean().item()),
                "max_pole_radius": float(poles.max().item()),
            }

    def forward(self, length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dt = torch.exp(self.log_dt).to(device=device, dtype=dtype)
        c = torch.view_as_complex(self.c).to(device=device, dtype=torch.cfloat if dtype == torch.float32 else torch.complex128)
        a = (
            -torch.exp(self.log_a_real).to(device=device, dtype=dtype)
            + 1j * self.a_imag.to(device=device, dtype=dtype)
        )
        dt_a = a * dt.unsqueeze(-1)
        n = torch.arange(length, device=device, dtype=dtype)
        kernel = dt_a.unsqueeze(-1) * n
        coeff = c * ((torch.exp(dt_a) - 1.0) / a)
        conv_kernel = 2 * torch.einsum("hn,hnl->hl", coeff, torch.exp(kernel)).real
        return conv_kernel.to(dtype=dtype)


class OfficialS4DLayer1d(nn.Module):
    def __init__(self, channels: int, state_dim: int, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        self.kernel = OfficialS4DKernel1d(channels, state_dim)
        self.d = nn.Parameter(torch.randn(channels))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(channels, 2 * channels, kernel_size=1),
            nn.GLU(dim=1),
        )

    def pole_radius_stats(self) -> Dict[str, float]:
        return self.kernel.pole_radius_stats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        length = x.size(1)
        x_perm = x.permute(0, 2, 1)
        kernel = self.kernel(length, device=x.device, dtype=x.dtype)
        fft_size = 1
        while fft_size < 2 * length:
            fft_size *= 2
        x_ft = torch.fft.rfft(x_perm, n=fft_size, dim=-1)
        k_ft = torch.fft.rfft(kernel, n=fft_size, dim=-1).unsqueeze(0)
        y = torch.fft.irfft(x_ft * k_ft, n=fft_size, dim=-1)[..., :length]
        y = y + x_perm * self.d.view(1, -1, 1)
        y = self.output_linear(self.dropout(self.activation(y)))
        return y.permute(0, 2, 1)


class OfficialS4D1d(nn.Module, InputGridMixin):
    def __init__(self, config: OperatorConfig, grid: torch.Tensor, use_grid: bool, input_channels: int):
        super().__init__()
        self.use_grid = use_grid
        self.register_buffer("grid", grid.reshape(-1).float(), persistent=False)
        input_dim = input_channels + int(use_grid)
        self.fc0 = nn.Linear(input_dim, config.width)
        state_dim = config.state_dim if config.state_dim is not None else config.width
        self.layers = nn.ModuleList(
            [OfficialS4DLayer1d(config.width, state_dim=state_dim) for _ in range(config.depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(config.depth)])
        self.fc1 = nn.Linear(config.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.activation = config.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._append_grid(x)
        x = self.fc0(x)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))
        x = apply_activation(self.fc1(x), self.activation)
        return self.fc2(x)

    def stability_stats(self) -> Dict[str, float]:
        stats = [layer.pole_radius_stats() for layer in self.layers]
        if not stats:
            return {}
        mean_radius = sum(item["mean_pole_radius"] for item in stats) / len(stats)
        max_radius = max(item["max_pole_radius"] for item in stats)
        return {
            "mean_pole_radius": mean_radius,
            "max_pole_radius": max_radius,
        }


def build_model(
    model_name: str,
    config: OperatorConfig,
    grid: torch.Tensor,
    use_grid: bool,
    input_channels: int = 1,
    dt: float = 1.0,
) -> nn.Module:
    if model_name == "lno":
        return DiscreteLNO1d(config=config, grid=grid, use_grid=use_grid, input_channels=input_channels)
    if model_name == "lno_stable":
        return DiscreteStableLNO1d(
            config=config,
            grid=grid,
            use_grid=use_grid,
            input_channels=input_channels,
        )
    if model_name == "fno":
        return DiscreteFNO1d(config=config, grid=grid, use_grid=use_grid, input_channels=input_channels)
    if model_name == "zno":
        return DiscreteZNO1d(
            config=config,
            grid=grid,
            use_grid=use_grid,
            input_channels=input_channels,
            pole_parameterization=config.pole_parameterization,
            dt=dt,
        )
    if model_name == "zno_splane_iso":
        return DiscreteZNO1d(
            config=config,
            grid=grid,
            use_grid=use_grid,
            input_channels=input_channels,
            pole_parameterization="s_exp",
            dt=dt,
        )
    if model_name == "s4d":
        return S4D1d(config=config, grid=grid, use_grid=use_grid, input_channels=input_channels)
    if model_name == "s4d_official":
        return OfficialS4D1d(config=config, grid=grid, use_grid=use_grid, input_channels=input_channels)
    if model_name == "linear_fir":
        return LinearFIR1d(config=config, grid=grid, use_grid=use_grid, input_channels=input_channels)
    raise ValueError(f"Unsupported model: {model_name}")


def stability_stats_if_available(model: nn.Module) -> Dict[str, float]:
    if hasattr(model, "stability_stats"):
        return model.stability_stats()
    return {}
