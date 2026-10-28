from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    TRITON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = exc


TRITON_MAX_BLOCK_POLES = 512


def triton_recurrent_available() -> bool:
    return triton is not None and tl is not None


def _select_block_p(num_poles: int) -> int:
    block_p = 1 << max(0, int(math.ceil(math.log2(max(1, num_poles)))))
    if block_p > TRITON_MAX_BLOCK_POLES:
        raise ValueError(
            f"Triton recurrent path currently supports num_poles up to {TRITON_MAX_BLOCK_POLES}, "
            f"got {num_poles}."
        )
    return block_p


def _select_block_fir(fir_size: int) -> int:
    if fir_size <= 0:
        return 0
    return 1 << max(0, int(math.ceil(math.log2(fir_size))))


if triton_recurrent_available():
    _TRITON_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]

    @triton.autotune(
        configs=_TRITON_AUTOTUNE_CONFIGS,
        key=["batch_size", "time_steps", "rank", "num_poles"],
    )
    @triton.jit
    def _triton_forward_kernel(
        projected_ptr,
        poles_real_ptr,
        poles_imag_ptr,
        residues_real_ptr,
        residues_imag_ptr,
        fir_weights_ptr,
        filtered_ptr,
        state_real_hist_ptr,
        state_imag_hist_ptr,
        batch_size,
        time_steps,
        rank,
        num_poles,
        BLOCK_P: tl.constexpr,
        FIR_SIZE: tl.constexpr,
        BLOCK_FIR: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_index = pid // rank
        rank_index = pid - batch_index * rank
        offs_p = tl.arange(0, BLOCK_P)
        pole_mask = offs_p < num_poles
        param_base = rank_index * num_poles

        pole_real = tl.load(poles_real_ptr + param_base + offs_p, mask=pole_mask, other=0.0)
        pole_imag = tl.load(poles_imag_ptr + param_base + offs_p, mask=pole_mask, other=0.0)
        residue_real = tl.load(residues_real_ptr + param_base + offs_p, mask=pole_mask, other=0.0)
        residue_imag = tl.load(residues_imag_ptr + param_base + offs_p, mask=pole_mask, other=0.0)
        state_real = tl.zeros((BLOCK_P,), dtype=tl.float32)
        state_imag = tl.zeros((BLOCK_P,), dtype=tl.float32)

        for time_index in tl.range(0, time_steps):
            projected_offset = ((batch_index * time_steps) + time_index) * rank + rank_index
            projected_t = tl.load(projected_ptr + projected_offset)
            next_state_real = state_real * pole_real - state_imag * pole_imag + projected_t
            next_state_imag = state_real * pole_imag + state_imag * pole_real
            state_real = tl.where(pole_mask, next_state_real, 0.0)
            state_imag = tl.where(pole_mask, next_state_imag, 0.0)

            history_offset = (((batch_index * time_steps) + time_index) * rank + rank_index) * num_poles
            tl.store(state_real_hist_ptr + history_offset + offs_p, state_real, mask=pole_mask)
            tl.store(state_imag_hist_ptr + history_offset + offs_p, state_imag, mask=pole_mask)

            filtered_t = tl.sum(state_real * residue_real - state_imag * residue_imag, axis=0)
            if FIR_SIZE > 0:
                fir_base = rank_index * FIR_SIZE
                for fir_index in range(FIR_SIZE):
                    valid = time_index >= fir_index
                    source_time = tl.where(valid, time_index - fir_index, 0)
                    source_offset = ((batch_index * time_steps) + source_time) * rank + rank_index
                    fir_weight = tl.load(fir_weights_ptr + fir_base + fir_index)
                    filtered_t += tl.where(valid, tl.load(projected_ptr + source_offset) * fir_weight, 0.0)
            tl.store(filtered_ptr + projected_offset, filtered_t)

    @triton.autotune(
        configs=_TRITON_AUTOTUNE_CONFIGS,
        key=["batch_size", "time_steps", "rank", "num_poles"],
    )
    @triton.jit
    def _triton_backward_kernel(
        grad_filtered_ptr,
        projected_ptr,
        poles_real_ptr,
        poles_imag_ptr,
        residues_real_ptr,
        residues_imag_ptr,
        fir_weights_ptr,
        state_real_hist_ptr,
        state_imag_hist_ptr,
        grad_projected_ptr,
        partial_grad_poles_real_ptr,
        partial_grad_poles_imag_ptr,
        partial_grad_residues_real_ptr,
        partial_grad_residues_imag_ptr,
        partial_grad_fir_ptr,
        batch_size,
        time_steps,
        rank,
        num_poles,
        BLOCK_P: tl.constexpr,
        FIR_SIZE: tl.constexpr,
        BLOCK_FIR: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        batch_index = pid // rank
        rank_index = pid - batch_index * rank
        offs_p = tl.arange(0, BLOCK_P)
        pole_mask = offs_p < num_poles
        param_base = rank_index * num_poles

        pole_real = tl.load(poles_real_ptr + param_base + offs_p, mask=pole_mask, other=0.0)
        pole_imag = tl.load(poles_imag_ptr + param_base + offs_p, mask=pole_mask, other=0.0)
        residue_real = tl.load(residues_real_ptr + param_base + offs_p, mask=pole_mask, other=0.0)
        residue_imag = tl.load(residues_imag_ptr + param_base + offs_p, mask=pole_mask, other=0.0)

        grad_state_real = tl.zeros((BLOCK_P,), dtype=tl.float32)
        grad_state_imag = tl.zeros((BLOCK_P,), dtype=tl.float32)
        grad_pole_real = tl.zeros((BLOCK_P,), dtype=tl.float32)
        grad_pole_imag = tl.zeros((BLOCK_P,), dtype=tl.float32)
        grad_residue_real = tl.zeros((BLOCK_P,), dtype=tl.float32)
        grad_residue_imag = tl.zeros((BLOCK_P,), dtype=tl.float32)
        if FIR_SIZE > 0:
            offs_f = tl.arange(0, BLOCK_FIR)
            fir_mask = offs_f < FIR_SIZE
            grad_fir = tl.zeros((BLOCK_FIR,), dtype=tl.float32)

        for rev_index in tl.range(0, time_steps):
            time_index = time_steps - 1 - rev_index
            projected_offset = ((batch_index * time_steps) + time_index) * rank + rank_index
            grad_output = tl.load(grad_filtered_ptr + projected_offset)
            history_offset = (((batch_index * time_steps) + time_index) * rank + rank_index) * num_poles
            state_real = tl.load(state_real_hist_ptr + history_offset + offs_p, mask=pole_mask, other=0.0)
            state_imag = tl.load(state_imag_hist_ptr + history_offset + offs_p, mask=pole_mask, other=0.0)

            prev_valid = time_index > 0
            prev_time = tl.where(prev_valid, time_index - 1, 0)
            prev_offset = (((batch_index * time_steps) + prev_time) * rank + rank_index) * num_poles
            prev_mask = pole_mask & prev_valid
            prev_state_real = tl.load(state_real_hist_ptr + prev_offset + offs_p, mask=prev_mask, other=0.0)
            prev_state_imag = tl.load(state_imag_hist_ptr + prev_offset + offs_p, mask=prev_mask, other=0.0)

            grad_residue_real += grad_output * state_real
            grad_residue_imag += -grad_output * state_imag
            grad_state_real += grad_output * residue_real
            grad_state_imag += -grad_output * residue_imag

            grad_projected_t = tl.sum(grad_state_real, axis=0)
            grad_pole_real += grad_state_real * prev_state_real + grad_state_imag * prev_state_imag
            grad_pole_imag += -grad_state_real * prev_state_imag + grad_state_imag * prev_state_real

            next_grad_state_real = grad_state_real * pole_real + grad_state_imag * pole_imag
            next_grad_state_imag = -grad_state_real * pole_imag + grad_state_imag * pole_real
            grad_state_real = tl.where(pole_mask, next_grad_state_real, 0.0)
            grad_state_imag = tl.where(pole_mask, next_grad_state_imag, 0.0)

            if FIR_SIZE > 0:
                fir_base = rank_index * FIR_SIZE
                for fir_index in range(FIR_SIZE):
                    future_valid = time_index + fir_index < time_steps
                    future_time = tl.where(future_valid, time_index + fir_index, 0)
                    future_offset = ((batch_index * time_steps) + future_time) * rank + rank_index
                    fir_weight = tl.load(fir_weights_ptr + fir_base + fir_index)
                    grad_projected_t += tl.where(
                        future_valid,
                        tl.load(grad_filtered_ptr + future_offset) * fir_weight,
                        0.0,
                    )

                    past_valid = time_index >= fir_index
                    past_time = tl.where(past_valid, time_index - fir_index, 0)
                    past_offset = ((batch_index * time_steps) + past_time) * rank + rank_index
                    grad_fir = tl.where(
                        (offs_f == fir_index) & fir_mask,
                        grad_fir + tl.where(past_valid, grad_output * tl.load(projected_ptr + past_offset), 0.0),
                        grad_fir,
                    )

            tl.store(grad_projected_ptr + projected_offset, grad_projected_t)

        partial_base = (batch_index * rank + rank_index) * num_poles
        tl.store(partial_grad_poles_real_ptr + partial_base + offs_p, grad_pole_real, mask=pole_mask)
        tl.store(partial_grad_poles_imag_ptr + partial_base + offs_p, grad_pole_imag, mask=pole_mask)
        tl.store(partial_grad_residues_real_ptr + partial_base + offs_p, grad_residue_real, mask=pole_mask)
        tl.store(partial_grad_residues_imag_ptr + partial_base + offs_p, grad_residue_imag, mask=pole_mask)
        if FIR_SIZE > 0:
            fir_base = (batch_index * rank + rank_index) * FIR_SIZE
            tl.store(partial_grad_fir_ptr + fir_base + offs_f, grad_fir, mask=fir_mask)

class TritonRecurrentSaveHistoryFunction(torch.autograd.Function):
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
        if not triton_recurrent_available():
            raise RuntimeError(f"Triton is not available: {TRITON_IMPORT_ERROR}")
        if projected.dtype != torch.float32:
            raise TypeError(f"Triton recurrent path currently supports float32 only, got {projected.dtype}")
        projected_contiguous = projected.contiguous()
        fir_weights_contiguous = fir_weights.contiguous()
        block_p = _select_block_p(int(poles_real.size(1)))
        block_fir = _select_block_fir(int(fir_weights_contiguous.size(1)) if fir_weights_contiguous.numel() > 0 else 0)
        fir_size = int(fir_weights_contiguous.size(1)) if fir_weights_contiguous.numel() > 0 else 0
        filtered = torch.empty_like(projected_contiguous)
        state_real_hist = torch.empty(
            projected_contiguous.size(0),
            projected_contiguous.size(1),
            projected_contiguous.size(2),
            poles_real.size(1),
            device=projected_contiguous.device,
            dtype=projected_contiguous.dtype,
        )
        state_imag_hist = torch.empty_like(state_real_hist)
        grid = (projected_contiguous.size(0) * projected_contiguous.size(2),)
        _triton_forward_kernel[grid](
            projected_contiguous,
            poles_real,
            poles_imag,
            residues_real,
            residues_imag,
            fir_weights_contiguous,
            filtered,
            state_real_hist,
            state_imag_hist,
            projected_contiguous.size(0),
            projected_contiguous.size(1),
            projected_contiguous.size(2),
            poles_real.size(1),
            BLOCK_P=block_p,
            FIR_SIZE=fir_size,
            BLOCK_FIR=block_fir,
        )
        ctx.save_for_backward(
            projected,
            poles_real,
            poles_imag,
            residues_real,
            residues_imag,
            fir_weights_contiguous,
            state_real_hist,
            state_imag_hist,
        )
        ctx.block_p = block_p
        ctx.fir_size = fir_size
        ctx.block_fir = block_fir
        return filtered

    @staticmethod
    def backward(ctx, grad_filtered: torch.Tensor):
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
        projected_contiguous = projected.contiguous()
        fir_weights_contiguous = fir_weights.contiguous()
        grad_projected = torch.empty_like(projected_contiguous)
        partial_shape = (projected.size(0), projected.size(2), poles_real.size(1))
        partial_grad_poles_real = torch.empty(partial_shape, device=projected.device, dtype=projected.dtype)
        partial_grad_poles_imag = torch.empty_like(partial_grad_poles_real)
        partial_grad_residues_real = torch.empty_like(partial_grad_poles_real)
        partial_grad_residues_imag = torch.empty_like(partial_grad_poles_real)
        if ctx.fir_size > 0:
            partial_grad_fir = torch.empty(
                projected.size(0),
                projected.size(2),
                ctx.fir_size,
                device=projected.device,
                dtype=projected.dtype,
            )
        else:
            partial_grad_fir = projected.new_empty((0,))
        grid = (projected.size(0) * projected.size(2),)
        _triton_backward_kernel[grid](
            grad_filtered.contiguous(),
            projected_contiguous,
            poles_real,
            poles_imag,
            residues_real,
            residues_imag,
            fir_weights_contiguous,
            state_real_hist,
            state_imag_hist,
            grad_projected,
            partial_grad_poles_real,
            partial_grad_poles_imag,
            partial_grad_residues_real,
            partial_grad_residues_imag,
            partial_grad_fir,
            projected.size(0),
            projected.size(1),
            projected.size(2),
            poles_real.size(1),
            BLOCK_P=ctx.block_p,
            FIR_SIZE=ctx.fir_size,
            BLOCK_FIR=ctx.block_fir,
        )
        grad_poles_real = partial_grad_poles_real.sum(dim=0)
        grad_poles_imag = partial_grad_poles_imag.sum(dim=0)
        grad_residues_real = partial_grad_residues_real.sum(dim=0)
        grad_residues_imag = partial_grad_residues_imag.sum(dim=0)
        grad_fir_weights = partial_grad_fir.sum(dim=0) if ctx.fir_size > 0 else None
        return (
            grad_projected,
            grad_poles_real,
            grad_poles_imag,
            grad_residues_real,
            grad_residues_imag,
            grad_fir_weights,
        )
def triton_recurrent_scan(
    projected: torch.Tensor,
    poles: torch.Tensor,
    residues: torch.Tensor,
    fir_weights: torch.Tensor | None,
) -> torch.Tensor:
    if not projected.is_cuda:
        raise RuntimeError("triton_recurrent_scan requires CUDA tensors.")
    fir_weights_tensor = fir_weights if fir_weights is not None else projected.new_empty((0,))
    return TritonRecurrentSaveHistoryFunction.apply(
        projected,
        poles.real.contiguous(),
        poles.imag.contiguous(),
        residues.real.contiguous(),
        residues.imag.contiguous(),
        fir_weights_tensor,
    )
