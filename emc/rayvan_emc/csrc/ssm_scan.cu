#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// One lane per independent (request, channel). The entire time recurrence
// stays inside one launch; adjacent lanes access adjacent channels. No host
// token loop, atomics, global state or division by cumulative decay products.
__global__ void forward_kernel(const float* a, const float* b, float* h,
                               int64_t B, int64_t T, int64_t D) {
  int64_t lane = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lane >= B * D) return;
  int64_t offset = (lane / D) * T * D + lane % D;
  float state = 0.f;
  for (int64_t t = 0; t < T; ++t) {
    int64_t i = offset + t * D;
    state = a[i] * state + b[i];
    h[i] = state;
  }
}
__global__ void backward_kernel(const float* a, const float* h, const float* g,
                                float* da, float* db, int64_t B, int64_t T, int64_t D) {
  int64_t lane = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lane >= B * D) return;
  int64_t offset = (lane / D) * T * D + lane % D;
  float q = 0.f;
  for (int64_t t = T; t-- > 0;) {
    int64_t i = offset + t * D;
    q = g[i] + (t + 1 < T ? a[i + D] * q : 0.f);
    db[i] = q;
    da[i] = t > 0 ? q * h[i - D] : 0.f;
  }
}
void check(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device() == b.device(), "same CUDA device required");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "FP32 required");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(a.dim() == 3 && a.sizes() == b.sizes() && a.size(1) > 0, "matching BTD shapes required");
}
torch::Tensor ssm_forward(torch::Tensor a, torch::Tensor b) {
  check(a, b);
  const c10::cuda::CUDAGuard guard(a.device());
  auto h = torch::empty_like(b);
  auto lanes = a.size(0) * a.size(2);
  if (lanes) forward_kernel<<<(lanes + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
    a.data_ptr<float>(), b.data_ptr<float>(), h.data_ptr<float>(), a.size(0), a.size(1), a.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}
std::vector<torch::Tensor> ssm_backward(torch::Tensor a, torch::Tensor h, torch::Tensor g) {
  check(a, h); check(a, g);
  const c10::cuda::CUDAGuard guard(a.device());
  auto da = torch::empty_like(a), db = torch::empty_like(a);
  auto lanes = a.size(0) * a.size(2);
  if (lanes) backward_kernel<<<(lanes + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
    a.data_ptr<float>(), h.data_ptr<float>(), g.data_ptr<float>(), da.data_ptr<float>(), db.data_ptr<float>(),
    a.size(0), a.size(1), a.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {da, db};
}
