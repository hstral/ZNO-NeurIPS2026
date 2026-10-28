"""Chunked-scan Triton kernels for the ZNO recurrent branch (backend "triton_fast").

Computes the same operator as ``triton_recurrent.triton_recurrent_scan``:

    s_{k,n} = p_k s_{k,n-1} + u_n,    y_n = sum_k Re(c_k s_{k,n}) + sum_j w_j u_{n-j},

for every (batch, latent channel), with zero initial state, and its exact gradients.
Differences from the reference kernel are purely in the evaluation order:

* Conjugate pole pairs are processed once. The kernels take one lane per conjugate pair
  plus the real pole (if any), instead of the ``[pairs, conj(pairs), real]`` layout of
  ``LowRankRationalLayer1d.stable_poles_and_residues``. The state of a conjugate pole is
  the exact conjugate of its partner's, so each pair contributes ``2 Re(c s)``, and the
  pair gradient is twice the per-entry gradient, which is what autograd accumulates
  through the conjugate copy.
* Time is split into tiles of ``BLOCK_T`` steps. Within a tile the linear recurrence is
  evaluated with ``tl.associative_scan``; the state is carried sequentially across tiles.
* The backward pass recomputes each tile's states from the carried state saved at the
  tile start instead of reading a stored state history, and evaluates the adjoint
  recurrence ``M_n = conj(p) M_{n+1} + g_n`` with a reverse scan.

Results agree with the reference kernel to float32 rounding (see
``tests/test_triton_fast.py``).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_FAST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    triton = None
    tl = None
    TRITON_FAST_IMPORT_ERROR = exc


def triton_fast_available() -> bool:
    return triton is not None and tl is not None


def _next_pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


if triton_fast_available():

    @triton.jit
    def _linrec_combine(a_re, a_im, x_re, x_im, b_re, b_im, y_re, y_im):
        # Composition of affine maps s -> a s + x (earlier) and s -> b s + y (later).
        return (
            b_re * a_re - b_im * a_im,
            b_re * a_im + b_im * a_re,
            b_re * x_re - b_im * x_im + y_re,
            b_re * x_im + b_im * x_re + y_im,
        )

    @triton.jit
    def _tile_states(u_ptr, base, t0, offs_t, time_steps, rank, p_re, p_im, s_in_re, s_in_im,
                     BLOCK_T: tl.constexpr):
        """States s_{t0..t0+BLOCK_T-1} of one tile from the state s_in at t0-1.

        prev_t = s_{t0+t-1} is obtained by a scan seeded with s_in; then
        s_t = p * prev_t + u_t, the recurrence itself. Forward and backward call this
        helper, so the backward recomputation reproduces the forward states.
        """
        t_prev = t0 + offs_t - 1
        u_prev = tl.load(u_ptr + base + t_prev * rank, mask=(offs_t >= 1) & (t_prev < time_steps), other=0.0)
        row0 = (offs_t == 0)[:, None]
        x_re = tl.where(row0, s_in_re[None, :], u_prev[:, None])
        x_im = tl.where(row0, s_in_im[None, :], 0.0)
        a_re = tl.zeros_like(x_re) + p_re[None, :]
        a_im = tl.zeros_like(x_re) + p_im[None, :]
        _, _, prev_re, prev_im = tl.associative_scan((a_re, a_im, x_re, x_im), 0, _linrec_combine)
        t_cur = t0 + offs_t
        u_cur = tl.load(u_ptr + base + t_cur * rank, mask=t_cur < time_steps, other=0.0)
        s_re = p_re[None, :] * prev_re - p_im[None, :] * prev_im + u_cur[:, None]
        s_im = p_re[None, :] * prev_im + p_im[None, :] * prev_re
        return prev_re, prev_im, s_re, s_im, u_cur

    @triton.jit
    def _load_lanes(ptr, r, num_lanes, offs_l, lmask):
        """Real and imaginary parts of the lanes of channel r from an interleaved (rank, lanes, 2) array."""
        addr = ptr + (r * num_lanes + offs_l) * 2
        return tl.load(addr, mask=lmask, other=0.0), tl.load(addr + 1, mask=lmask, other=0.0)

    @triton.jit
    def _fast_forward_kernel(
        u_ptr, poles_ptr, residues_ptr, fir_ptr,
        y_ptr, carry_ptr,
        time_steps, rank, num_lanes, num_pairs, num_tiles,
        BLOCK_T: tl.constexpr, BLOCK_L: tl.constexpr, FIR_SIZE: tl.constexpr, SAVE_CARRY: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // rank
        r = pid - b * rank
        offs_t = tl.arange(0, BLOCK_T)
        offs_l = tl.arange(0, BLOCK_L)
        lmask = offs_l < num_lanes
        p_re, p_im = _load_lanes(poles_ptr, r, num_lanes, offs_l, lmask)
        c_re, c_im = _load_lanes(residues_ptr, r, num_lanes, offs_l, lmask)
        weight = tl.where(offs_l < num_pairs, 2.0, 1.0)
        base = b * time_steps * rank + r
        s_re = tl.zeros((BLOCK_L,), dtype=tl.float32)
        s_im = tl.zeros((BLOCK_L,), dtype=tl.float32)
        # carry layout: (2, batch * rank, num_tiles, num_lanes) for real and imaginary parts
        carry_base = (pid * num_tiles) * num_lanes
        carry_im_off = tl.num_programs(0) * num_tiles * num_lanes
        for tile in range(num_tiles):
            t0 = tile * BLOCK_T
            if SAVE_CARRY:
                tl.store(carry_ptr + carry_base + tile * num_lanes + offs_l, s_re, mask=lmask)
                tl.store(carry_ptr + carry_im_off + carry_base + tile * num_lanes + offs_l, s_im, mask=lmask)
            _, _, st_re, st_im, u_cur = _tile_states(
                u_ptr, base, t0, offs_t, time_steps, rank, p_re, p_im, s_re, s_im, BLOCK_T)
            y = tl.sum(weight[None, :] * (c_re[None, :] * st_re - c_im[None, :] * st_im), axis=1)
            t_cur = t0 + offs_t
            if FIR_SIZE > 0:
                for j in tl.static_range(FIR_SIZE):
                    w_j = tl.load(fir_ptr + r * FIR_SIZE + j)
                    t_src = t_cur - j
                    u_src = tl.load(u_ptr + base + t_src * rank, mask=(t_src >= 0) & (t_cur < time_steps), other=0.0)
                    y += w_j * u_src
            tl.store(y_ptr + base + t_cur * rank, y, mask=t_cur < time_steps)
            last = tl.minimum(time_steps - t0, BLOCK_T) - 1
            sel = (offs_t == last)[:, None]
            s_re = tl.sum(tl.where(sel, st_re, 0.0), axis=0)
            s_im = tl.sum(tl.where(sel, st_im, 0.0), axis=0)

    @triton.jit
    def _store_lane_grads(ptr, pid, num_lanes, offs_l, lmask, weight, g_re, g_im):
        """Write weighted lane gradients into an interleaved (batch * rank, lanes, 2) array."""
        addr = ptr + (pid * num_lanes + offs_l) * 2
        tl.store(addr, weight * g_re, mask=lmask)
        tl.store(addr + 1, weight * g_im, mask=lmask)

    @triton.jit
    def _fast_backward_kernel(
        g_ptr, u_ptr, poles_ptr, residues_ptr, fir_ptr, carry_ptr,
        gu_ptr, gpoles_ptr, gres_ptr, gfir_ptr,
        time_steps, rank, num_lanes, num_pairs, num_tiles,
        BLOCK_T: tl.constexpr, BLOCK_L: tl.constexpr, FIR_SIZE: tl.constexpr, BLOCK_F: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // rank
        r = pid - b * rank
        offs_t = tl.arange(0, BLOCK_T)
        offs_l = tl.arange(0, BLOCK_L)
        lmask = offs_l < num_lanes
        p_re, p_im = _load_lanes(poles_ptr, r, num_lanes, offs_l, lmask)
        c_re, c_im = _load_lanes(residues_ptr, r, num_lanes, offs_l, lmask)
        weight = tl.where(offs_l < num_pairs, 2.0, 1.0)
        base = b * time_steps * rank + r
        carry_base = (pid * num_tiles) * num_lanes
        carry_im_off = tl.num_programs(0) * num_tiles * num_lanes
        m_re = tl.zeros((BLOCK_L,), dtype=tl.float32)
        m_im = tl.zeros((BLOCK_L,), dtype=tl.float32)
        gp_re = tl.zeros((BLOCK_L,), dtype=tl.float32)
        gp_im = tl.zeros((BLOCK_L,), dtype=tl.float32)
        gc_re = tl.zeros((BLOCK_L,), dtype=tl.float32)
        gc_im = tl.zeros((BLOCK_L,), dtype=tl.float32)
        offs_f = tl.arange(0, BLOCK_F)
        gfir = tl.zeros((BLOCK_F,), dtype=tl.float32)
        for it in range(num_tiles):
            tile = num_tiles - 1 - it
            t0 = tile * BLOCK_T
            t_cur = t0 + offs_t
            tmask = t_cur < time_steps
            g = tl.load(g_ptr + base + t_cur * rank, mask=tmask, other=0.0)
            # Adjoint M_t = conj(p) M_{t+1} + g_t, seeded with the carry from the later tile.
            last = tl.minimum(time_steps - t0, BLOCK_T) - 1
            rowl = (offs_t == last)[:, None]
            seed_re = p_re * m_re + p_im * m_im
            seed_im = p_re * m_im - p_im * m_re
            x_re = g[:, None] + tl.where(rowl, seed_re[None, :], 0.0)
            x_im = tl.where(rowl, seed_im[None, :], 0.0)
            x_re = tl.where(tmask[:, None], x_re, 0.0)
            a_re = tl.zeros_like(x_re) + p_re[None, :]
            a_im = tl.zeros_like(x_re) - p_im[None, :]
            _, _, mt_re, mt_im = tl.associative_scan((a_re, a_im, x_re, x_im), 0, _linrec_combine, reverse=True)
            # G = conj(c) M is the gradient with respect to the state of each pair member.
            gs_re = c_re[None, :] * mt_re + c_im[None, :] * mt_im
            gs_im = c_re[None, :] * mt_im - c_im[None, :] * mt_re
            gu = tl.sum(weight[None, :] * gs_re, axis=1)
            if FIR_SIZE > 0:
                for j in tl.static_range(FIR_SIZE):
                    w_j = tl.load(fir_ptr + r * FIR_SIZE + j)
                    t_dst = t_cur + j
                    g_dst = tl.load(g_ptr + base + t_dst * rank, mask=t_dst < time_steps, other=0.0)
                    gu += w_j * g_dst
                    t_src = t_cur - j
                    u_src = tl.load(u_ptr + base + t_src * rank, mask=(t_src >= 0) & tmask, other=0.0)
                    gfir += tl.where(offs_f == j, tl.sum(g * u_src, axis=0), 0.0)
            tl.store(gu_ptr + base + t_cur * rank, gu, mask=tmask)
            # Recompute the tile's states from the carried state at t0-1.
            s_in_re = tl.load(carry_ptr + carry_base + tile * num_lanes + offs_l, mask=lmask, other=0.0)
            s_in_im = tl.load(carry_ptr + carry_im_off + carry_base + tile * num_lanes + offs_l, mask=lmask, other=0.0)
            prev_re, prev_im, st_re, st_im, _ = _tile_states(
                u_ptr, base, t0, offs_t, time_steps, rank, p_re, p_im, s_in_re, s_in_im, BLOCK_T)
            valid = tmask[:, None]
            gp_re += tl.sum(tl.where(valid, gs_re * prev_re + gs_im * prev_im, 0.0), axis=0)
            gp_im += tl.sum(tl.where(valid, gs_im * prev_re - gs_re * prev_im, 0.0), axis=0)
            gc_re += tl.sum(tl.where(valid, g[:, None] * st_re, 0.0), axis=0)
            gc_im += tl.sum(tl.where(valid, -g[:, None] * st_im, 0.0), axis=0)
            row0 = (offs_t == 0)[:, None]
            m_re = tl.sum(tl.where(row0, mt_re, 0.0), axis=0)
            m_im = tl.sum(tl.where(row0, mt_im, 0.0), axis=0)
        _store_lane_grads(gpoles_ptr, pid, num_lanes, offs_l, lmask, weight, gp_re, gp_im)
        _store_lane_grads(gres_ptr, pid, num_lanes, offs_l, lmask, weight, gc_re, gc_im)
        if FIR_SIZE > 0:
            tl.store(gfir_ptr + pid * FIR_SIZE + offs_f, gfir, mask=offs_f < FIR_SIZE)


DEFAULT_BLOCK_T = 32
DEFAULT_NUM_WARPS = 4          # backward kernel
DEFAULT_FORWARD_NUM_WARPS = 1  # forward kernel


class TritonFastRecurrentFunction(torch.autograd.Function):
    """Inputs: projected (batch, time, rank); lane poles and residues as interleaved real views
    (rank, lanes, 2) of complex [pairs, real] tensors; FIR weights or None."""

    @staticmethod
    def forward(ctx, projected, poles_ri, residues_ri, fir_weights, num_pairs, block_t, num_warps, forward_warps, save):
        if projected.dtype != torch.float32:
            raise TypeError(f"triton_fast supports float32 only, got {projected.dtype}")
        u = projected.contiguous()
        batch, time_steps, rank = u.shape
        num_lanes = poles_ri.size(1)
        if num_lanes - num_pairs not in (0, 1):
            raise ValueError(f"unexpected lane layout: {num_lanes} lanes, {num_pairs} pairs")
        fir = fir_weights if fir_weights is not None else u.new_empty((0,))
        fir_size = int(fir.size(1)) if fir.numel() > 0 else 0
        num_tiles = triton.cdiv(time_steps, block_t)
        y = torch.empty_like(u)
        # The carried states are always written (a few MB), so training and evaluation use the
        # same compiled kernel and the warm-up step compiles everything before timing starts.
        carry = u.new_empty((2, batch * rank, num_tiles, num_lanes))
        _fast_forward_kernel[(batch * rank,)](
            u, poles_ri, residues_ri, fir, y, carry,
            time_steps, rank, num_lanes, num_pairs, num_tiles,
            BLOCK_T=block_t, BLOCK_L=_next_pow2(num_lanes), FIR_SIZE=fir_size, SAVE_CARRY=True,
            num_warps=forward_warps,
        )
        if save:
            ctx.save_for_backward(u, poles_ri, residues_ri, fir, carry)
            ctx.meta = (num_pairs, fir_size, num_tiles, block_t, num_warps, fir_weights is not None)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        u, poles_ri, residues_ri, fir, carry = ctx.saved_tensors
        num_pairs, fir_size, num_tiles, block_t, num_warps, has_fir = ctx.meta
        batch, time_steps, rank = u.shape
        num_lanes = poles_ri.size(1)
        gu = torch.empty_like(u)
        part = u.new_empty((2, batch, rank, num_lanes, 2))
        gfir = u.new_empty((batch, rank, max(fir_size, 1)))
        _fast_backward_kernel[(batch * rank,)](
            grad_y.contiguous(), u, poles_ri, residues_ri, fir, carry,
            gu, part[0], part[1], gfir,
            time_steps, rank, num_lanes, num_pairs, num_tiles,
            BLOCK_T=block_t, BLOCK_L=_next_pow2(num_lanes), FIR_SIZE=fir_size,
            BLOCK_F=_next_pow2(max(fir_size, 1)), num_warps=num_warps,
        )
        grad_poles, grad_res = part.sum(dim=1)
        grad_fir = gfir.sum(dim=0) if has_fir and fir_size > 0 else None
        return gu, grad_poles, grad_res, grad_fir, None, None, None, None, None


def triton_fast_recurrent_scan(
    projected: torch.Tensor,
    lane_poles: torch.Tensor,
    lane_residues: torch.Tensor,
    fir_weights: torch.Tensor | None,
    num_pairs: int,
    block_t: int | None = None,
    num_warps: int | None = None,
    forward_warps: int | None = None,
) -> torch.Tensor:
    """Recurrent branch from lane poles/residues (rank, lanes): ``num_pairs`` conjugate-pair
    lanes (one member of each pair) followed by the real pole, if any."""
    if not projected.is_cuda:
        raise RuntimeError("triton_fast_recurrent_scan requires CUDA tensors.")
    if not triton_fast_available():
        raise RuntimeError(f"Triton is not available: {TRITON_FAST_IMPORT_ERROR}")
    block_t = DEFAULT_BLOCK_T if block_t is None else block_t
    num_warps = DEFAULT_NUM_WARPS if num_warps is None else num_warps
    forward_warps = DEFAULT_FORWARD_NUM_WARPS if forward_warps is None else forward_warps
    poles_ri = torch.view_as_real(lane_poles.contiguous())
    residues_ri = torch.view_as_real(lane_residues.contiguous())
    save = torch.is_grad_enabled() and (
        projected.requires_grad or lane_poles.requires_grad or lane_residues.requires_grad
        or (fir_weights is not None and fir_weights.requires_grad)
    )
    return TritonFastRecurrentFunction.apply(
        projected, poles_ri, residues_ri, fir_weights, int(num_pairs), int(block_t), int(num_warps),
        int(forward_warps), bool(save)
    )
