#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kMaxThreads = 256;
constexpr int kMaxWarps = (kMaxThreads + 31) / 32;

inline int next_pow2_threads(int value) {
  int threads = 1;
  while (threads < value && threads < kMaxThreads) {
    threads <<= 1;
  }
  return threads;
}

template <typename scalar_t>
__device__ inline scalar_t warp_reduce_sum(scalar_t value) {
  const unsigned mask = __activemask();
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

template <typename scalar_t>
__device__ inline scalar_t block_reduce_sum(scalar_t value) {
  __shared__ scalar_t warp_sums[kMaxWarps];
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  scalar_t block_sum = static_cast<scalar_t>(0);
  if (warp == 0) {
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    block_sum = lane < num_warps ? warp_sums[lane] : static_cast<scalar_t>(0);
    block_sum = warp_reduce_sum(block_sum);
  }
  return block_sum;
}

template <typename scalar_t>
__global__ void fused_recurrent_forward_kernel(
    const scalar_t* __restrict__ projected,
    const scalar_t* __restrict__ poles_real,
    const scalar_t* __restrict__ poles_imag,
    const scalar_t* __restrict__ residues_real,
    const scalar_t* __restrict__ residues_imag,
    const scalar_t* __restrict__ fir_weights,
    scalar_t* __restrict__ filtered,
    scalar_t* __restrict__ state_real_hist,
    scalar_t* __restrict__ state_imag_hist,
    int batch_size,
    int time_steps,
    int rank,
    int num_poles,
    int fir_size) {
  const int sequence_index = blockIdx.x;
  const int batch_index = sequence_index / rank;
  const int rank_index = sequence_index % rank;
  const int pole_index = threadIdx.x;

  scalar_t state_real = static_cast<scalar_t>(0);
  scalar_t state_imag = static_cast<scalar_t>(0);
  scalar_t pole_real = static_cast<scalar_t>(0);
  scalar_t pole_imag = static_cast<scalar_t>(0);
  scalar_t residue_real = static_cast<scalar_t>(0);
  scalar_t residue_imag = static_cast<scalar_t>(0);

  if (pole_index < num_poles) {
    const int param_offset = rank_index * num_poles + pole_index;
    pole_real = poles_real[param_offset];
    pole_imag = poles_imag[param_offset];
    residue_real = residues_real[param_offset];
    residue_imag = residues_imag[param_offset];
  }

  for (int time_index = 0; time_index < time_steps; ++time_index) {
    const int projected_offset = ((batch_index * time_steps + time_index) * rank) + rank_index;
    const scalar_t projected_t = projected[projected_offset];

    scalar_t contribution = static_cast<scalar_t>(0);
    if (pole_index < num_poles) {
      const scalar_t next_state_real =
          state_real * pole_real - state_imag * pole_imag + projected_t;
      const scalar_t next_state_imag =
          state_real * pole_imag + state_imag * pole_real;
      state_real = next_state_real;
      state_imag = next_state_imag;

      const int history_offset =
          (((batch_index * time_steps + time_index) * rank + rank_index) * num_poles) + pole_index;
      if (state_real_hist != nullptr) {
        state_real_hist[history_offset] = state_real;
        state_imag_hist[history_offset] = state_imag;
      }
      contribution = state_real * residue_real - state_imag * residue_imag;
    }
    const scalar_t contribution_sum = block_reduce_sum(contribution);

    if (pole_index == 0) {
      scalar_t output_t = contribution_sum;
      if (fir_size > 0) {
        const int max_j = min(fir_size - 1, time_index);
        for (int fir_index = 0; fir_index <= max_j; ++fir_index) {
          const int source_offset =
              ((batch_index * time_steps + (time_index - fir_index)) * rank) + rank_index;
          output_t += projected[source_offset] * fir_weights[rank_index * fir_size + fir_index];
        }
      }
      filtered[projected_offset] = output_t;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void fused_recurrent_forward_generic_kernel(
    const scalar_t* __restrict__ projected,
    const scalar_t* __restrict__ poles_real,
    const scalar_t* __restrict__ poles_imag,
    const scalar_t* __restrict__ residues_real,
    const scalar_t* __restrict__ residues_imag,
    const scalar_t* __restrict__ fir_weights,
    scalar_t* __restrict__ filtered,
    scalar_t* __restrict__ state_real_hist,
    scalar_t* __restrict__ state_imag_hist,
    scalar_t* __restrict__ state_real_curr,
    scalar_t* __restrict__ state_imag_curr,
    int batch_size,
    int time_steps,
    int rank,
    int num_poles,
    int fir_size) {
  const int sequence_index = blockIdx.x;
  const int batch_index = sequence_index / rank;
  const int rank_index = sequence_index % rank;

  for (int time_index = 0; time_index < time_steps; ++time_index) {
    const int projected_offset = ((batch_index * time_steps + time_index) * rank) + rank_index;
    const scalar_t projected_t = projected[projected_offset];
    scalar_t contribution = static_cast<scalar_t>(0);

    for (int pole_index = threadIdx.x; pole_index < num_poles; pole_index += blockDim.x) {
      const int param_offset = rank_index * num_poles + pole_index;
      scalar_t prev_state_real = static_cast<scalar_t>(0);
      scalar_t prev_state_imag = static_cast<scalar_t>(0);
      if (state_real_hist != nullptr) {
        if (time_index > 0) {
          const int prev_history_offset =
              (((batch_index * time_steps + (time_index - 1)) * rank + rank_index) * num_poles) +
              pole_index;
          prev_state_real = state_real_hist[prev_history_offset];
          prev_state_imag = state_imag_hist[prev_history_offset];
        }
      } else {
        const int state_offset = sequence_index * num_poles + pole_index;
        prev_state_real = state_real_curr[state_offset];
        prev_state_imag = state_imag_curr[state_offset];
      }

      const scalar_t pole_real = poles_real[param_offset];
      const scalar_t pole_imag = poles_imag[param_offset];
      const scalar_t residue_real = residues_real[param_offset];
      const scalar_t residue_imag = residues_imag[param_offset];
      const scalar_t next_state_real =
          prev_state_real * pole_real - prev_state_imag * pole_imag + projected_t;
      const scalar_t next_state_imag =
          prev_state_real * pole_imag + prev_state_imag * pole_real;
      if (state_real_hist != nullptr) {
        const int history_offset =
            (((batch_index * time_steps + time_index) * rank + rank_index) * num_poles) + pole_index;
        state_real_hist[history_offset] = next_state_real;
        state_imag_hist[history_offset] = next_state_imag;
      } else {
        const int state_offset = sequence_index * num_poles + pole_index;
        state_real_curr[state_offset] = next_state_real;
        state_imag_curr[state_offset] = next_state_imag;
      }
      contribution += next_state_real * residue_real - next_state_imag * residue_imag;
    }

    const scalar_t contribution_sum = block_reduce_sum(contribution);
    if (threadIdx.x == 0) {
      scalar_t output_t = contribution_sum;
      if (fir_size > 0) {
        const int max_j = min(fir_size - 1, time_index);
        for (int fir_index = 0; fir_index <= max_j; ++fir_index) {
          const int source_offset =
              ((batch_index * time_steps + (time_index - fir_index)) * rank) + rank_index;
          output_t += projected[source_offset] * fir_weights[rank_index * fir_size + fir_index];
        }
      }
      filtered[projected_offset] = output_t;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void fused_recurrent_backward_kernel(
    const scalar_t* __restrict__ grad_filtered,
    const scalar_t* __restrict__ projected,
    const scalar_t* __restrict__ poles_real,
    const scalar_t* __restrict__ poles_imag,
    const scalar_t* __restrict__ residues_real,
    const scalar_t* __restrict__ residues_imag,
    const scalar_t* __restrict__ fir_weights,
    const scalar_t* __restrict__ state_real_hist,
    const scalar_t* __restrict__ state_imag_hist,
    scalar_t* __restrict__ grad_projected,
    scalar_t* __restrict__ partial_grad_poles_real,
    scalar_t* __restrict__ partial_grad_poles_imag,
    scalar_t* __restrict__ partial_grad_residues_real,
    scalar_t* __restrict__ partial_grad_residues_imag,
    scalar_t* __restrict__ partial_grad_fir_weights,
    int batch_size,
    int time_steps,
    int rank,
    int num_poles,
    int fir_size) {
  const int sequence_index = blockIdx.x;
  const int batch_index = sequence_index / rank;
  const int rank_index = sequence_index % rank;
  const int pole_index = threadIdx.x;

  scalar_t pole_real = static_cast<scalar_t>(0);
  scalar_t pole_imag = static_cast<scalar_t>(0);
  scalar_t residue_real = static_cast<scalar_t>(0);
  scalar_t residue_imag = static_cast<scalar_t>(0);
  if (pole_index < num_poles) {
    const int param_offset = rank_index * num_poles + pole_index;
    pole_real = poles_real[param_offset];
    pole_imag = poles_imag[param_offset];
    residue_real = residues_real[param_offset];
    residue_imag = residues_imag[param_offset];
  }

  scalar_t grad_state_real = static_cast<scalar_t>(0);
  scalar_t grad_state_imag = static_cast<scalar_t>(0);
  scalar_t grad_pole_real = static_cast<scalar_t>(0);
  scalar_t grad_pole_imag = static_cast<scalar_t>(0);
  scalar_t grad_residue_real = static_cast<scalar_t>(0);
  scalar_t grad_residue_imag = static_cast<scalar_t>(0);

  if (pole_index == 0 && fir_size > 0) {
    scalar_t* partial_fir_ptr =
        partial_grad_fir_weights + ((batch_index * rank + rank_index) * fir_size);
    for (int fir_index = 0; fir_index < fir_size; ++fir_index) {
      partial_fir_ptr[fir_index] = static_cast<scalar_t>(0);
    }
  }
  __syncthreads();

  for (int time_index = time_steps - 1; time_index >= 0; --time_index) {
    const int projected_offset = ((batch_index * time_steps + time_index) * rank) + rank_index;
    const scalar_t grad_output = grad_filtered[projected_offset];

    scalar_t grad_projected_from_poles = static_cast<scalar_t>(0);
    if (pole_index < num_poles) {
      const int history_offset =
          (((batch_index * time_steps + time_index) * rank + rank_index) * num_poles) + pole_index;
      const scalar_t state_real = state_real_hist[history_offset];
      const scalar_t state_imag = state_imag_hist[history_offset];

      scalar_t prev_state_real = static_cast<scalar_t>(0);
      scalar_t prev_state_imag = static_cast<scalar_t>(0);
      if (time_index > 0) {
        const int prev_history_offset =
            (((batch_index * time_steps + (time_index - 1)) * rank + rank_index) * num_poles) +
            pole_index;
        prev_state_real = state_real_hist[prev_history_offset];
        prev_state_imag = state_imag_hist[prev_history_offset];
      }

      grad_residue_real += grad_output * state_real;
      grad_residue_imag += -grad_output * state_imag;

      grad_state_real += grad_output * residue_real;
      grad_state_imag += -grad_output * residue_imag;

      grad_projected_from_poles = grad_state_real;
      grad_pole_real += grad_state_real * prev_state_real + grad_state_imag * prev_state_imag;
      grad_pole_imag += -grad_state_real * prev_state_imag + grad_state_imag * prev_state_real;

      const scalar_t next_grad_state_real =
          grad_state_real * pole_real + grad_state_imag * pole_imag;
      const scalar_t next_grad_state_imag =
          -grad_state_real * pole_imag + grad_state_imag * pole_real;
      grad_state_real = next_grad_state_real;
      grad_state_imag = next_grad_state_imag;
    }
    const scalar_t grad_projected_from_poles_sum = block_reduce_sum(grad_projected_from_poles);

    if (pole_index == 0) {
      scalar_t grad_projected_t = grad_projected_from_poles_sum;
      scalar_t* partial_fir_ptr =
          fir_size > 0 ? partial_grad_fir_weights + ((batch_index * rank + rank_index) * fir_size)
                       : nullptr;

      if (fir_size > 0) {
        const int max_future = min(fir_size - 1, time_steps - 1 - time_index);
        for (int fir_index = 0; fir_index <= max_future; ++fir_index) {
          const int future_offset =
              ((batch_index * time_steps + (time_index + fir_index)) * rank) + rank_index;
          grad_projected_t +=
              grad_filtered[future_offset] * fir_weights[rank_index * fir_size + fir_index];
        }

        const int max_past = min(fir_size - 1, time_index);
        for (int fir_index = 0; fir_index <= max_past; ++fir_index) {
          const int past_offset =
              ((batch_index * time_steps + (time_index - fir_index)) * rank) + rank_index;
          partial_fir_ptr[fir_index] += grad_output * projected[past_offset];
        }
      }

      grad_projected[projected_offset] = grad_projected_t;
    }
    __syncthreads();
  }

  if (pole_index < num_poles) {
    const int partial_offset = ((batch_index * rank + rank_index) * num_poles) + pole_index;
    partial_grad_poles_real[partial_offset] = grad_pole_real;
    partial_grad_poles_imag[partial_offset] = grad_pole_imag;
    partial_grad_residues_real[partial_offset] = grad_residue_real;
    partial_grad_residues_imag[partial_offset] = grad_residue_imag;
  }
}

template <typename scalar_t>
__global__ void fused_recurrent_backward_generic_kernel(
    const scalar_t* __restrict__ grad_filtered,
    const scalar_t* __restrict__ projected,
    const scalar_t* __restrict__ poles_real,
    const scalar_t* __restrict__ poles_imag,
    const scalar_t* __restrict__ residues_real,
    const scalar_t* __restrict__ residues_imag,
    const scalar_t* __restrict__ fir_weights,
    const scalar_t* __restrict__ state_real_hist,
    const scalar_t* __restrict__ state_imag_hist,
    scalar_t* __restrict__ grad_projected,
    scalar_t* __restrict__ partial_grad_poles_real,
    scalar_t* __restrict__ partial_grad_poles_imag,
    scalar_t* __restrict__ partial_grad_residues_real,
    scalar_t* __restrict__ partial_grad_residues_imag,
    scalar_t* __restrict__ partial_grad_fir_weights,
    scalar_t* __restrict__ grad_state_real_curr,
    scalar_t* __restrict__ grad_state_imag_curr,
    int batch_size,
    int time_steps,
    int rank,
    int num_poles,
    int fir_size) {
  const int sequence_index = blockIdx.x;
  const int batch_index = sequence_index / rank;
  const int rank_index = sequence_index % rank;
  const int partial_base = sequence_index * num_poles;
  const int grad_state_base = sequence_index * num_poles;

  if (threadIdx.x == 0 && fir_size > 0) {
    scalar_t* partial_fir_ptr =
        partial_grad_fir_weights + ((batch_index * rank + rank_index) * fir_size);
    for (int fir_index = 0; fir_index < fir_size; ++fir_index) {
      partial_fir_ptr[fir_index] = static_cast<scalar_t>(0);
    }
  }

  for (int pole_index = threadIdx.x; pole_index < num_poles; pole_index += blockDim.x) {
    const int offset = partial_base + pole_index;
    partial_grad_poles_real[offset] = static_cast<scalar_t>(0);
    partial_grad_poles_imag[offset] = static_cast<scalar_t>(0);
    partial_grad_residues_real[offset] = static_cast<scalar_t>(0);
    partial_grad_residues_imag[offset] = static_cast<scalar_t>(0);
    grad_state_real_curr[grad_state_base + pole_index] = static_cast<scalar_t>(0);
    grad_state_imag_curr[grad_state_base + pole_index] = static_cast<scalar_t>(0);
  }
  __syncthreads();

  for (int time_index = time_steps - 1; time_index >= 0; --time_index) {
    const int projected_offset = ((batch_index * time_steps + time_index) * rank) + rank_index;
    const scalar_t grad_output = grad_filtered[projected_offset];
    scalar_t grad_projected_from_poles = static_cast<scalar_t>(0);

    for (int pole_index = threadIdx.x; pole_index < num_poles; pole_index += blockDim.x) {
      const int param_offset = rank_index * num_poles + pole_index;
      const int partial_offset = partial_base + pole_index;
      const int grad_state_offset = grad_state_base + pole_index;
      const int history_offset =
          (((batch_index * time_steps + time_index) * rank + rank_index) * num_poles) + pole_index;

      const scalar_t state_real = state_real_hist[history_offset];
      const scalar_t state_imag = state_imag_hist[history_offset];
      scalar_t prev_state_real = static_cast<scalar_t>(0);
      scalar_t prev_state_imag = static_cast<scalar_t>(0);
      if (time_index > 0) {
        const int prev_history_offset =
            (((batch_index * time_steps + (time_index - 1)) * rank + rank_index) * num_poles) +
            pole_index;
        prev_state_real = state_real_hist[prev_history_offset];
        prev_state_imag = state_imag_hist[prev_history_offset];
      }

      scalar_t grad_state_real = grad_state_real_curr[grad_state_offset];
      scalar_t grad_state_imag = grad_state_imag_curr[grad_state_offset];
      partial_grad_residues_real[partial_offset] += grad_output * state_real;
      partial_grad_residues_imag[partial_offset] += -grad_output * state_imag;

      grad_state_real += grad_output * residues_real[param_offset];
      grad_state_imag += -grad_output * residues_imag[param_offset];
      grad_projected_from_poles += grad_state_real;

      partial_grad_poles_real[partial_offset] +=
          grad_state_real * prev_state_real + grad_state_imag * prev_state_imag;
      partial_grad_poles_imag[partial_offset] +=
          -grad_state_real * prev_state_imag + grad_state_imag * prev_state_real;

      const scalar_t next_grad_state_real =
          grad_state_real * poles_real[param_offset] + grad_state_imag * poles_imag[param_offset];
      const scalar_t next_grad_state_imag =
          -grad_state_real * poles_imag[param_offset] + grad_state_imag * poles_real[param_offset];
      grad_state_real_curr[grad_state_offset] = next_grad_state_real;
      grad_state_imag_curr[grad_state_offset] = next_grad_state_imag;
    }

    const scalar_t grad_projected_from_poles_sum = block_reduce_sum(grad_projected_from_poles);
    if (threadIdx.x == 0) {
      scalar_t grad_projected_t = grad_projected_from_poles_sum;
      scalar_t* partial_fir_ptr =
          fir_size > 0 ? partial_grad_fir_weights + ((batch_index * rank + rank_index) * fir_size)
                       : nullptr;

      if (fir_size > 0) {
        const int max_future = min(fir_size - 1, time_steps - 1 - time_index);
        for (int fir_index = 0; fir_index <= max_future; ++fir_index) {
          const int future_offset =
              ((batch_index * time_steps + (time_index + fir_index)) * rank) + rank_index;
          grad_projected_t +=
              grad_filtered[future_offset] * fir_weights[rank_index * fir_size + fir_index];
        }

        const int max_past = min(fir_size - 1, time_index);
        for (int fir_index = 0; fir_index <= max_past; ++fir_index) {
          const int past_offset =
              ((batch_index * time_steps + (time_index - fir_index)) * rank) + rank_index;
          partial_fir_ptr[fir_index] += grad_output * projected[past_offset];
        }
      }

      grad_projected[projected_offset] = grad_projected_t;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void reduce_partial_pole_grads_kernel(
    const scalar_t* __restrict__ partial_grad_poles_real,
    const scalar_t* __restrict__ partial_grad_poles_imag,
    const scalar_t* __restrict__ partial_grad_residues_real,
    const scalar_t* __restrict__ partial_grad_residues_imag,
    scalar_t* __restrict__ grad_poles_real,
    scalar_t* __restrict__ grad_poles_imag,
    scalar_t* __restrict__ grad_residues_real,
    scalar_t* __restrict__ grad_residues_imag,
    int batch_size,
    int rank,
    int num_poles) {
  const int rank_index = blockIdx.x;
  if (rank_index >= rank) {
    return;
  }
  for (int pole_index = threadIdx.x; pole_index < num_poles; pole_index += blockDim.x) {
    const int param_offset = rank_index * num_poles + pole_index;
    scalar_t grad_pole_real = static_cast<scalar_t>(0);
    scalar_t grad_pole_imag = static_cast<scalar_t>(0);
    scalar_t grad_residue_real = static_cast<scalar_t>(0);
    scalar_t grad_residue_imag = static_cast<scalar_t>(0);
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      const int partial_offset = ((batch_index * rank + rank_index) * num_poles) + pole_index;
      grad_pole_real += partial_grad_poles_real[partial_offset];
      grad_pole_imag += partial_grad_poles_imag[partial_offset];
      grad_residue_real += partial_grad_residues_real[partial_offset];
      grad_residue_imag += partial_grad_residues_imag[partial_offset];
    }
    grad_poles_real[param_offset] = grad_pole_real;
    grad_poles_imag[param_offset] = grad_pole_imag;
    grad_residues_real[param_offset] = grad_residue_real;
    grad_residues_imag[param_offset] = grad_residue_imag;
  }
}

template <typename scalar_t>
__global__ void reduce_partial_fir_grads_kernel(
    const scalar_t* __restrict__ partial_grad_fir_weights,
    scalar_t* __restrict__ grad_fir_weights,
    int batch_size,
    int rank,
    int fir_size) {
  const int rank_index = blockIdx.x;
  if (rank_index >= rank) {
    return;
  }
  for (int fir_index = threadIdx.x; fir_index < fir_size; fir_index += blockDim.x) {
    const int fir_offset = rank_index * fir_size + fir_index;
    scalar_t grad_fir = static_cast<scalar_t>(0);
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      const int partial_offset = ((batch_index * rank + rank_index) * fir_size) + fir_index;
      grad_fir += partial_grad_fir_weights[partial_offset];
    }
    grad_fir_weights[fir_offset] = grad_fir;
  }
}

}  // namespace

std::vector<torch::Tensor> fused_recurrent_forward_cuda(
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights) {
  const at::cuda::CUDAGuard device_guard(projected.device());

  const auto batch_size = static_cast<int>(projected.size(0));
  const auto time_steps = static_cast<int>(projected.size(1));
  const auto rank = static_cast<int>(projected.size(2));
  const auto num_poles = static_cast<int>(poles_real.size(1));
  const auto fir_size = static_cast<int>(fir_weights.numel() == 0 ? 0 : fir_weights.size(1));

  auto filtered = torch::zeros_like(projected);
  auto state_real_hist =
      torch::empty({batch_size, time_steps, rank, num_poles}, projected.options());
  auto state_imag_hist =
      torch::empty({batch_size, time_steps, rank, num_poles}, projected.options());

  const int blocks = batch_size * rank;
  const int threads = num_poles <= kMaxThreads ? next_pow2_threads(num_poles) : kMaxThreads;

  AT_DISPATCH_FLOATING_TYPES(projected.scalar_type(), "fused_recurrent_forward_cuda", [&] {
    if (num_poles <= kMaxThreads) {
      fused_recurrent_forward_kernel<scalar_t><<<blocks, threads>>>(
          projected.data_ptr<scalar_t>(),
          poles_real.data_ptr<scalar_t>(),
          poles_imag.data_ptr<scalar_t>(),
          residues_real.data_ptr<scalar_t>(),
          residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? fir_weights.data_ptr<scalar_t>() : nullptr,
          filtered.data_ptr<scalar_t>(),
          state_real_hist.data_ptr<scalar_t>(),
          state_imag_hist.data_ptr<scalar_t>(),
          batch_size,
          time_steps,
          rank,
          num_poles,
          fir_size);
    } else {
      fused_recurrent_forward_generic_kernel<scalar_t><<<blocks, threads>>>(
          projected.data_ptr<scalar_t>(),
          poles_real.data_ptr<scalar_t>(),
          poles_imag.data_ptr<scalar_t>(),
          residues_real.data_ptr<scalar_t>(),
          residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? fir_weights.data_ptr<scalar_t>() : nullptr,
          filtered.data_ptr<scalar_t>(),
          state_real_hist.data_ptr<scalar_t>(),
          state_imag_hist.data_ptr<scalar_t>(),
          nullptr,
          nullptr,
          batch_size,
          time_steps,
          rank,
          num_poles,
          fir_size);
    }
  });

  return {filtered, state_real_hist, state_imag_hist};
}

torch::Tensor fused_recurrent_forward_no_history_cuda(
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights) {
  const at::cuda::CUDAGuard device_guard(projected.device());

  const auto batch_size = static_cast<int>(projected.size(0));
  const auto time_steps = static_cast<int>(projected.size(1));
  const auto rank = static_cast<int>(projected.size(2));
  const auto num_poles = static_cast<int>(poles_real.size(1));
  const auto fir_size = static_cast<int>(fir_weights.numel() == 0 ? 0 : fir_weights.size(1));

  auto filtered = torch::zeros_like(projected);
  const int blocks = batch_size * rank;
  const int threads = num_poles <= kMaxThreads ? next_pow2_threads(num_poles) : kMaxThreads;

  AT_DISPATCH_FLOATING_TYPES(projected.scalar_type(), "fused_recurrent_forward_no_history_cuda", [&] {
    if (num_poles <= kMaxThreads) {
      fused_recurrent_forward_kernel<scalar_t><<<blocks, threads>>>(
          projected.data_ptr<scalar_t>(),
          poles_real.data_ptr<scalar_t>(),
          poles_imag.data_ptr<scalar_t>(),
          residues_real.data_ptr<scalar_t>(),
          residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? fir_weights.data_ptr<scalar_t>() : nullptr,
          filtered.data_ptr<scalar_t>(),
          nullptr,
          nullptr,
          batch_size,
          time_steps,
          rank,
          num_poles,
          fir_size);
    } else {
      auto state_real_curr = torch::zeros({batch_size, rank, num_poles}, projected.options());
      auto state_imag_curr = torch::zeros({batch_size, rank, num_poles}, projected.options());
      fused_recurrent_forward_generic_kernel<scalar_t><<<blocks, threads>>>(
          projected.data_ptr<scalar_t>(),
          poles_real.data_ptr<scalar_t>(),
          poles_imag.data_ptr<scalar_t>(),
          residues_real.data_ptr<scalar_t>(),
          residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? fir_weights.data_ptr<scalar_t>() : nullptr,
          filtered.data_ptr<scalar_t>(),
          nullptr,
          nullptr,
          state_real_curr.data_ptr<scalar_t>(),
          state_imag_curr.data_ptr<scalar_t>(),
          batch_size,
          time_steps,
          rank,
          num_poles,
          fir_size);
    }
  });

  return filtered;
}

std::vector<torch::Tensor> fused_recurrent_backward_cuda(
    const torch::Tensor& grad_filtered,
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights,
    const torch::Tensor& state_real_hist,
    const torch::Tensor& state_imag_hist) {
  const at::cuda::CUDAGuard device_guard(projected.device());

  const auto batch_size = static_cast<int>(projected.size(0));
  const auto time_steps = static_cast<int>(projected.size(1));
  const auto rank = static_cast<int>(projected.size(2));
  const auto num_poles = static_cast<int>(poles_real.size(1));
  const auto fir_size = static_cast<int>(fir_weights.numel() == 0 ? 0 : fir_weights.size(1));

  auto grad_projected = torch::empty_like(projected);
  auto partial_grad_poles_real =
      torch::zeros({batch_size, rank, num_poles}, projected.options());
  auto partial_grad_poles_imag =
      torch::zeros({batch_size, rank, num_poles}, projected.options());
  auto partial_grad_residues_real =
      torch::zeros({batch_size, rank, num_poles}, projected.options());
  auto partial_grad_residues_imag =
      torch::zeros({batch_size, rank, num_poles}, projected.options());
  auto partial_grad_fir_weights = fir_size > 0
                                      ? torch::zeros({batch_size, rank, fir_size}, projected.options())
                                      : torch::empty_like(fir_weights);
  auto grad_poles_real = torch::empty_like(poles_real);
  auto grad_poles_imag = torch::empty_like(poles_imag);
  auto grad_residues_real = torch::empty_like(residues_real);
  auto grad_residues_imag = torch::empty_like(residues_imag);
  auto grad_fir_weights =
      fir_size > 0 ? torch::empty_like(fir_weights) : torch::empty_like(fir_weights);

  const int blocks = batch_size * rank;
  const int threads = num_poles <= kMaxThreads ? next_pow2_threads(num_poles) : kMaxThreads;

  AT_DISPATCH_FLOATING_TYPES(projected.scalar_type(), "fused_recurrent_backward_cuda", [&] {
    if (num_poles <= kMaxThreads) {
      fused_recurrent_backward_kernel<scalar_t><<<blocks, threads>>>(
          grad_filtered.data_ptr<scalar_t>(),
          projected.data_ptr<scalar_t>(),
          poles_real.data_ptr<scalar_t>(),
          poles_imag.data_ptr<scalar_t>(),
          residues_real.data_ptr<scalar_t>(),
          residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? fir_weights.data_ptr<scalar_t>() : nullptr,
          state_real_hist.data_ptr<scalar_t>(),
          state_imag_hist.data_ptr<scalar_t>(),
          grad_projected.data_ptr<scalar_t>(),
          partial_grad_poles_real.data_ptr<scalar_t>(),
          partial_grad_poles_imag.data_ptr<scalar_t>(),
          partial_grad_residues_real.data_ptr<scalar_t>(),
          partial_grad_residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? partial_grad_fir_weights.data_ptr<scalar_t>() : nullptr,
          batch_size,
          time_steps,
          rank,
          num_poles,
          fir_size);
    } else {
      auto grad_state_real_curr = torch::zeros({batch_size, rank, num_poles}, projected.options());
      auto grad_state_imag_curr = torch::zeros({batch_size, rank, num_poles}, projected.options());
      fused_recurrent_backward_generic_kernel<scalar_t><<<blocks, threads>>>(
          grad_filtered.data_ptr<scalar_t>(),
          projected.data_ptr<scalar_t>(),
          poles_real.data_ptr<scalar_t>(),
          poles_imag.data_ptr<scalar_t>(),
          residues_real.data_ptr<scalar_t>(),
          residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? fir_weights.data_ptr<scalar_t>() : nullptr,
          state_real_hist.data_ptr<scalar_t>(),
          state_imag_hist.data_ptr<scalar_t>(),
          grad_projected.data_ptr<scalar_t>(),
          partial_grad_poles_real.data_ptr<scalar_t>(),
          partial_grad_poles_imag.data_ptr<scalar_t>(),
          partial_grad_residues_real.data_ptr<scalar_t>(),
          partial_grad_residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? partial_grad_fir_weights.data_ptr<scalar_t>() : nullptr,
          grad_state_real_curr.data_ptr<scalar_t>(),
          grad_state_imag_curr.data_ptr<scalar_t>(),
          batch_size,
          time_steps,
          rank,
          num_poles,
          fir_size);
    }

    reduce_partial_pole_grads_kernel<scalar_t><<<rank, threads>>>(
        partial_grad_poles_real.data_ptr<scalar_t>(),
        partial_grad_poles_imag.data_ptr<scalar_t>(),
        partial_grad_residues_real.data_ptr<scalar_t>(),
        partial_grad_residues_imag.data_ptr<scalar_t>(),
        grad_poles_real.data_ptr<scalar_t>(),
        grad_poles_imag.data_ptr<scalar_t>(),
        grad_residues_real.data_ptr<scalar_t>(),
        grad_residues_imag.data_ptr<scalar_t>(),
        batch_size,
        rank,
        num_poles);

    if (fir_size > 0) {
      const int fir_threads = next_pow2_threads(fir_size);
      reduce_partial_fir_grads_kernel<scalar_t><<<rank, fir_threads>>>(
          partial_grad_fir_weights.data_ptr<scalar_t>(),
          grad_fir_weights.data_ptr<scalar_t>(),
          batch_size,
          rank,
          fir_size);
    }
  });

  return {
      grad_projected,
      grad_poles_real,
      grad_poles_imag,
      grad_residues_real,
      grad_residues_imag,
      grad_fir_weights};
}

std::vector<torch::Tensor> fused_recurrent_backward_recompute_cuda(
    const torch::Tensor& grad_filtered,
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights) {
  const at::cuda::CUDAGuard device_guard(projected.device());

  const auto batch_size = static_cast<int>(projected.size(0));
  const auto time_steps = static_cast<int>(projected.size(1));
  const auto rank = static_cast<int>(projected.size(2));
  const auto num_poles = static_cast<int>(poles_real.size(1));
  const auto fir_size = static_cast<int>(fir_weights.numel() == 0 ? 0 : fir_weights.size(1));

  auto recomputed_filtered = torch::empty_like(projected);
  auto state_real_hist =
      torch::empty({batch_size, time_steps, rank, num_poles}, projected.options());
  auto state_imag_hist =
      torch::empty({batch_size, time_steps, rank, num_poles}, projected.options());

  const int blocks = batch_size * rank;
  const int threads = num_poles <= kMaxThreads ? next_pow2_threads(num_poles) : kMaxThreads;

  AT_DISPATCH_FLOATING_TYPES(projected.scalar_type(), "fused_recurrent_backward_recompute_cuda", [&] {
    if (num_poles <= kMaxThreads) {
      fused_recurrent_forward_kernel<scalar_t><<<blocks, threads>>>(
          projected.data_ptr<scalar_t>(),
          poles_real.data_ptr<scalar_t>(),
          poles_imag.data_ptr<scalar_t>(),
          residues_real.data_ptr<scalar_t>(),
          residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? fir_weights.data_ptr<scalar_t>() : nullptr,
          recomputed_filtered.data_ptr<scalar_t>(),
          state_real_hist.data_ptr<scalar_t>(),
          state_imag_hist.data_ptr<scalar_t>(),
          batch_size,
          time_steps,
          rank,
          num_poles,
          fir_size);
    } else {
      fused_recurrent_forward_generic_kernel<scalar_t><<<blocks, threads>>>(
          projected.data_ptr<scalar_t>(),
          poles_real.data_ptr<scalar_t>(),
          poles_imag.data_ptr<scalar_t>(),
          residues_real.data_ptr<scalar_t>(),
          residues_imag.data_ptr<scalar_t>(),
          fir_size > 0 ? fir_weights.data_ptr<scalar_t>() : nullptr,
          recomputed_filtered.data_ptr<scalar_t>(),
          state_real_hist.data_ptr<scalar_t>(),
          state_imag_hist.data_ptr<scalar_t>(),
          nullptr,
          nullptr,
          batch_size,
          time_steps,
          rank,
          num_poles,
          fir_size);
    }
  });

  return fused_recurrent_backward_cuda(
      grad_filtered,
      projected,
      poles_real,
      poles_imag,
      residues_real,
      residues_imag,
      fir_weights,
      state_real_hist,
      state_imag_hist);
}
