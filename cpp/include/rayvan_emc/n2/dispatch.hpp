#pragma once

#include "rayvan_emc/tensor.hpp"

#include <cstdint>

namespace rayvan::emc {

struct DispatchPlan {
    Tensor expert_ids;             // [B*K]
    Tensor source_indices;         // [B*K]
    Tensor slot_indices;           // [B*K]
    Tensor permutation;            // [B*K]
    Tensor inverse_permutation;    // [B*K]
    Tensor sorted_expert_ids;      // [B*K]
    Tensor sorted_source_indices;  // [B*K]
    Tensor sorted_slot_indices;    // [B*K]
    Tensor expert_counts;          // [N]
    Tensor expert_offsets;         // [N+1]

    static DispatchPlan from_routing(const Tensor& selected_indices, std::int64_t num_experts);
    Tensor restore(const Tensor& grouped_proposals, std::int64_t batch, std::int64_t sequence, std::int64_t latent) const;
};

}  // namespace rayvan::emc
