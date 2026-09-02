#pragma once

#include "rayvan_emc/model.hpp"

namespace rayvan::emc {

inline CausalIntervention disable_n1(const Tensor& availability_mask) {
    CausalIntervention intervention;
    intervention.availability_mask = availability_mask;
    return intervention;
}

inline CausalIntervention force_alternate_n1(const Tensor& forced_nodes) {
    CausalIntervention intervention;
    intervention.forced_nodes = forced_nodes;
    return intervention;
}

inline CausalIntervention zero_n1_proposal(const Tensor& zero_mask) {
    CausalIntervention intervention;
    intervention.zero_proposal_mask = zero_mask;
    return intervention;
}

}  // namespace rayvan::emc
