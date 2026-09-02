#pragma once

#include <torch/optim/adamw.h>

namespace rayvan::emc {

// AdamW with the stock LibTorch state/checkpoint format and an ATen foreach
// update path. LibTorch's C++ AdamW frontend still updates one tensor at a
// time, even when all tensors are compatible with CUDA multi-tensor kernels.
class ForeachAdamW final : public torch::optim::AdamW {
public:
    using torch::optim::AdamW::AdamW;

    torch::Tensor step(LossClosure closure = nullptr) override;
};

}  // namespace rayvan::emc
