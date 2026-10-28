#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fused_recurrent_forward_cuda(
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights);

torch::Tensor fused_recurrent_forward_no_history_cuda(
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights);

std::vector<torch::Tensor> fused_recurrent_backward_cuda(
    const torch::Tensor& grad_filtered,
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights,
    const torch::Tensor& state_real_hist,
    const torch::Tensor& state_imag_hist);

std::vector<torch::Tensor> fused_recurrent_backward_recompute_cuda(
    const torch::Tensor& grad_filtered,
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights);

std::vector<torch::Tensor> fused_recurrent_forward(
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights) {
  CHECK_INPUT(projected);
  CHECK_INPUT(poles_real);
  CHECK_INPUT(poles_imag);
  CHECK_INPUT(residues_real);
  CHECK_INPUT(residues_imag);
  CHECK_INPUT(fir_weights);
  TORCH_CHECK(projected.dim() == 3, "projected must have shape [batch, time, rank]");
  TORCH_CHECK(poles_real.dim() == 2, "poles_real must have shape [rank, num_poles]");
  TORCH_CHECK(poles_imag.sizes() == poles_real.sizes(), "pole tensors must match");
  TORCH_CHECK(residues_real.sizes() == poles_real.sizes(), "residue tensors must match poles");
  TORCH_CHECK(residues_imag.sizes() == poles_real.sizes(), "residue tensors must match poles");
  TORCH_CHECK(projected.size(2) == poles_real.size(0), "rank mismatch");
  if (fir_weights.numel() > 0) {
    TORCH_CHECK(fir_weights.dim() == 2, "fir_weights must have shape [rank, fir_size]");
    TORCH_CHECK(fir_weights.size(0) == poles_real.size(0), "fir rank mismatch");
  }
  return fused_recurrent_forward_cuda(
      projected,
      poles_real,
      poles_imag,
      residues_real,
      residues_imag,
      fir_weights);
}

torch::Tensor fused_recurrent_forward_no_history(
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights) {
  CHECK_INPUT(projected);
  CHECK_INPUT(poles_real);
  CHECK_INPUT(poles_imag);
  CHECK_INPUT(residues_real);
  CHECK_INPUT(residues_imag);
  CHECK_INPUT(fir_weights);
  TORCH_CHECK(projected.dim() == 3, "projected must have shape [batch, time, rank]");
  TORCH_CHECK(poles_real.dim() == 2, "poles_real must have shape [rank, num_poles]");
  TORCH_CHECK(poles_imag.sizes() == poles_real.sizes(), "pole tensors must match");
  TORCH_CHECK(residues_real.sizes() == poles_real.sizes(), "residue tensors must match poles");
  TORCH_CHECK(residues_imag.sizes() == poles_real.sizes(), "residue tensors must match poles");
  TORCH_CHECK(projected.size(2) == poles_real.size(0), "rank mismatch");
  if (fir_weights.numel() > 0) {
    TORCH_CHECK(fir_weights.dim() == 2, "fir_weights must have shape [rank, fir_size]");
    TORCH_CHECK(fir_weights.size(0) == poles_real.size(0), "fir rank mismatch");
  }
  return fused_recurrent_forward_no_history_cuda(
      projected,
      poles_real,
      poles_imag,
      residues_real,
      residues_imag,
      fir_weights);
}

std::vector<torch::Tensor> fused_recurrent_backward(
    const torch::Tensor& grad_filtered,
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights,
    const torch::Tensor& state_real_hist,
    const torch::Tensor& state_imag_hist) {
  CHECK_INPUT(grad_filtered);
  CHECK_INPUT(projected);
  CHECK_INPUT(poles_real);
  CHECK_INPUT(poles_imag);
  CHECK_INPUT(residues_real);
  CHECK_INPUT(residues_imag);
  CHECK_INPUT(fir_weights);
  CHECK_INPUT(state_real_hist);
  CHECK_INPUT(state_imag_hist);
  TORCH_CHECK(grad_filtered.sizes() == projected.sizes(), "grad_filtered shape mismatch");
  TORCH_CHECK(state_real_hist.dim() == 4, "state_real_hist must have shape [batch, time, rank, num_poles]");
  TORCH_CHECK(state_imag_hist.sizes() == state_real_hist.sizes(), "state history tensors must match");
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

std::vector<torch::Tensor> fused_recurrent_backward_recompute(
    const torch::Tensor& grad_filtered,
    const torch::Tensor& projected,
    const torch::Tensor& poles_real,
    const torch::Tensor& poles_imag,
    const torch::Tensor& residues_real,
    const torch::Tensor& residues_imag,
    const torch::Tensor& fir_weights) {
  CHECK_INPUT(grad_filtered);
  CHECK_INPUT(projected);
  CHECK_INPUT(poles_real);
  CHECK_INPUT(poles_imag);
  CHECK_INPUT(residues_real);
  CHECK_INPUT(residues_imag);
  CHECK_INPUT(fir_weights);
  TORCH_CHECK(grad_filtered.sizes() == projected.sizes(), "grad_filtered shape mismatch");
  return fused_recurrent_backward_recompute_cuda(
      grad_filtered,
      projected,
      poles_real,
      poles_imag,
      residues_real,
      residues_imag,
      fir_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_recurrent_forward, "Fused recurrent forward (CUDA)");
  m.def("forward_no_history", &fused_recurrent_forward_no_history, "Fused recurrent forward without saving history (CUDA)");
  m.def("backward", &fused_recurrent_backward, "Fused recurrent backward (CUDA)");
  m.def("backward_recompute", &fused_recurrent_backward_recompute, "Fused recurrent backward with exact recomputation (CUDA)");
}
