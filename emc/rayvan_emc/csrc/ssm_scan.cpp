#include <torch/extension.h>

torch::Tensor ssm_forward(torch::Tensor a, torch::Tensor b);
std::vector<torch::Tensor> ssm_backward(torch::Tensor a, torch::Tensor h, torch::Tensor g);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ssm_forward);
  m.def("backward", &ssm_backward);
}
