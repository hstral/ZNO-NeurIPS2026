import os
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load


_EXTENSION_NAME = "zno_fused_recurrent_ext"
_THIS_DIR = Path(__file__).resolve().parent
_CSRC_DIR = _THIS_DIR / "csrc"


def _visible_arch_list() -> str | None:
    if not torch.cuda.is_available():
        return None
    archs = sorted(
        {
            f"{major}.{minor}"
            for major, minor in (torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count()))
        }
    )
    return ";".join(archs) if archs else None


@lru_cache(maxsize=1)
def _load_extension():
    build_directory = Path(tempfile.gettempdir()) / _EXTENSION_NAME
    build_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(build_directory))
    visible_arch_list = _visible_arch_list()
    if visible_arch_list is not None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", visible_arch_list)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix is not None:
            candidate = Path(conda_prefix)
            if (candidate / "bin" / "nvcc").exists():
                cuda_home = str(candidate)
                os.environ["CUDA_HOME"] = cuda_home
                cpp_extension.CUDA_HOME = cuda_home
    return load(
        name=_EXTENSION_NAME,
        sources=[
            str(_CSRC_DIR / "fused_recurrent.cpp"),
            str(_CSRC_DIR / "fused_recurrent_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=str(build_directory),
        verbose=False,
    )


class FusedRecurrentSaveHistoryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        projected: torch.Tensor,
        poles_real: torch.Tensor,
        poles_imag: torch.Tensor,
        residues_real: torch.Tensor,
        residues_imag: torch.Tensor,
        fir_weights: torch.Tensor,
    ):
        extension = _load_extension()
        filtered, state_real_hist, state_imag_hist = extension.forward(
            projected.contiguous(),
            poles_real.contiguous(),
            poles_imag.contiguous(),
            residues_real.contiguous(),
            residues_imag.contiguous(),
            fir_weights.contiguous(),
        )
        ctx.save_for_backward(
            projected,
            poles_real,
            poles_imag,
            residues_real,
            residues_imag,
            fir_weights,
            state_real_hist,
            state_imag_hist,
        )
        return filtered

    @staticmethod
    def backward(ctx, grad_filtered: torch.Tensor):
        extension = _load_extension()
        (
            projected,
            poles_real,
            poles_imag,
            residues_real,
            residues_imag,
            fir_weights,
            state_real_hist,
            state_imag_hist,
        ) = ctx.saved_tensors
        (
            grad_projected,
            grad_poles_real,
            grad_poles_imag,
            grad_residues_real,
            grad_residues_imag,
            grad_fir_weights,
        ) = extension.backward(
            grad_filtered.contiguous(),
            projected.contiguous(),
            poles_real.contiguous(),
            poles_imag.contiguous(),
            residues_real.contiguous(),
            residues_imag.contiguous(),
            fir_weights.contiguous(),
            state_real_hist.contiguous(),
            state_imag_hist.contiguous(),
        )
        if fir_weights.numel() == 0:
            grad_fir_weights = None
        return (
            grad_projected,
            grad_poles_real,
            grad_poles_imag,
            grad_residues_real,
            grad_residues_imag,
            grad_fir_weights,
        )


class FusedRecurrentRecomputeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        projected: torch.Tensor,
        poles_real: torch.Tensor,
        poles_imag: torch.Tensor,
        residues_real: torch.Tensor,
        residues_imag: torch.Tensor,
        fir_weights: torch.Tensor,
    ):
        extension = _load_extension()
        filtered = extension.forward_no_history(
            projected.contiguous(),
            poles_real.contiguous(),
            poles_imag.contiguous(),
            residues_real.contiguous(),
            residues_imag.contiguous(),
            fir_weights.contiguous(),
        )
        ctx.save_for_backward(
            projected,
            poles_real,
            poles_imag,
            residues_real,
            residues_imag,
            fir_weights,
        )
        return filtered

    @staticmethod
    def backward(ctx, grad_filtered: torch.Tensor):
        extension = _load_extension()
        (
            projected,
            poles_real,
            poles_imag,
            residues_real,
            residues_imag,
            fir_weights,
        ) = ctx.saved_tensors
        (
            grad_projected,
            grad_poles_real,
            grad_poles_imag,
            grad_residues_real,
            grad_residues_imag,
            grad_fir_weights,
        ) = extension.backward_recompute(
            grad_filtered.contiguous(),
            projected.contiguous(),
            poles_real.contiguous(),
            poles_imag.contiguous(),
            residues_real.contiguous(),
            residues_imag.contiguous(),
            fir_weights.contiguous(),
        )
        if fir_weights.numel() == 0:
            grad_fir_weights = None
        return (
            grad_projected,
            grad_poles_real,
            grad_poles_imag,
            grad_residues_real,
            grad_residues_imag,
            grad_fir_weights,
        )


def fused_recurrent_scan(
    projected: torch.Tensor,
    poles: torch.Tensor,
    residues: torch.Tensor,
    fir_weights: torch.Tensor | None,
    *,
    backward_mode: str = "save_history",
) -> torch.Tensor:
    if not projected.is_cuda:
        raise RuntimeError("fused_recurrent_scan requires CUDA tensors.")
    if projected.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"Unsupported dtype for fused_recurrent_scan: {projected.dtype}")
    if projected.size(-1) != poles.size(0):
        raise ValueError("Rank mismatch between projected input and poles.")
    fir_weights_tensor = fir_weights
    if fir_weights_tensor is None:
        fir_weights_tensor = projected.new_empty((0,))
    function = {
        "save_history": FusedRecurrentSaveHistoryFunction,
        "recompute": FusedRecurrentRecomputeFunction,
    }.get(backward_mode)
    if function is None:
        raise ValueError(f"Unsupported fused backward mode: {backward_mode}")
    return function.apply(
        projected,
        poles.real.contiguous(),
        poles.imag.contiguous(),
        residues.real.contiguous(),
        residues.imag.contiguous(),
        fir_weights_tensor,
    )


def precompile_fused_recurrent_extension(device: str = "cuda:0", backward_mode: str = "save_history") -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to precompile the fused recurrent extension.")
    torch_device = torch.device(device)
    _load_extension()
    projected = torch.randn(2, 32, 4, device=torch_device)
    poles = torch.randn(4, 8, device=torch_device, dtype=torch.cfloat) * 0.05
    residues = torch.randn(4, 8, device=torch_device, dtype=torch.cfloat)
    fir_weights = torch.randn(4, 3, device=torch_device)
    projected.requires_grad_(True)
    poles_real = poles.real.detach().clone().requires_grad_(True)
    poles_imag = poles.imag.detach().clone().requires_grad_(True)
    residues.requires_grad_(True)
    filtered = fused_recurrent_scan(
        projected,
        torch.complex(poles_real, poles_imag),
        residues,
        fir_weights,
        backward_mode=backward_mode,
    )
    loss = filtered.square().mean()
    loss.backward()
    torch.cuda.synchronize(torch_device)
