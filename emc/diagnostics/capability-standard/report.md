# Executive Summary

- Learning: `learning_signal_present`
- Overall loss: `3.0409`
- Language perplexity: `4.3944`
- Strongest / weakest: `language` / `arithmetic`
- Routing: slot_monopoly; ssm[1] selected in 97.4% of requests, recurrent[2] selected in 100.0% of requests; near-dead: delta[3] at 0.09% of slots
- Specialization (`moderate_specialization_signal`): moderate_specialization_signal: descriptive causal differences (language: recurrent; associative_recall: recurrent; fuzzy_recall: ssm; selective_copying: recurrent); sampled data do not establish statistical significance.
- Causal diagnostics: 648 active interventions evaluated; 533 measurably worsened loss or token accuracy
- Surface vs computation: operation correlates more strongly than surface; the larger normalized association is very_weak, which is descriptive rather than evidence of causal specialization.
- Cycles: n1_chunked exposes chunk recurrence and lease telemetry, but does not expose independently repeatable EMC cycles or a cycle-limit intervention

# Diagnostic Integrity Warnings

- `exact_accuracy_too_low_for_ablation` (warning): Overall exact accuracy is at or below 5%; exact-match deltas are unlikely to resolve causal effects. Use loss increase and token-accuracy degradation.
- `near_dead_experts` (warning): Experts [3] are below the configured selection-frequency threshold.
- `near_universal_experts` (warning): Experts [1, 2] exceed the configured request-presence threshold; inspect slot monopolization.
- `gradients_unavailable` (info): No live or training-instrumented gradient tensors are available for this diagnostic run.
- `update_norms_unavailable` (info): Parameter update norms were not captured during training.
- `cycle_telemetry_unavailable` (info): n1_chunked exposes chunk recurrence and lease telemetry, but does not expose independently repeatable EMC cycles or a cycle-limit intervention
- `small_ablation_sample` (warning): At least one family/capability cell has fewer than three active ablations; specialization evidence is descriptive.
- `ablation_target_not_active` (info): 474 interventions targeted a module absent from the baseline active path and were excluded from causal aggregation.
- `intervention_path_unchanged` (warning): 78 interventions did not change the validated active path.

# Overall Metrics

| row | examples | language_perplexity | overall_exact_accuracy | overall_loss | overall_token_accuracy | skipped |
|---|---|---|---|---|---|---|
| overall | 1000 | 4.3944 | 0.0260 | 3.0409 | 0.2141 | 0 |

# Capability Results

| row | exact_accuracy | token_accuracy | cross_entropy | perplexity |
|---|---|---|---|---|
| arithmetic | 0.0100 | 0.0628 | 3.3642 | 28.9097 |
| associative_recall | 0.0000 | 0.1814 | 3.4673 | 32.0486 |
| compression | 0.0100 | 0.1800 | 2.2944 | 9.9184 |
| fuzzy_recall | 0.0000 | 0.4829 | 1.8553 | 6.3936 |
| language | 0.0000 | 0.5707 | 1.4803 | 4.3944 |
| program_execution | 0.0300 | 0.1600 | 2.7940 | 16.3470 |
| selective_copying | 0.0000 | 0.1302 | 3.8317 | 46.1407 |
| stateful_action | 0.0200 | 0.0833 | 4.0774 | 58.9934 |
| symbolic | 0.1100 | 0.1800 | 3.4234 | 30.6737 |
| working_memory | 0.0800 | 0.1100 | 3.8211 | 45.6564 |

# Generalization and Difficulty Curves

| row | cross_entropy | exact_accuracy | route_delta | route_gpt | route_recurrent | route_ssm | routing_entropy | token_accuracy |
|---|---|---|---|---|---|---|---|---|
| arithmetic/difficulty-1 | 3.4781 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6528 | 0.0280 |
| arithmetic/difficulty-2 | 3.1467 | 0.0400 | 0.0000 | 0.0104 | 0.5000 | 0.4896 | 0.6521 | 0.1240 |
| arithmetic/difficulty-3 | 3.4673 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6536 | 0.0380 |
| arithmetic/difficulty-4 | 3.3645 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6526 | 0.0613 |
| associative_recall/difficulty-1 | 3.2502 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6581 | 0.2467 |
| associative_recall/difficulty-2 | 3.3706 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6581 | 0.1899 |
| associative_recall/difficulty-3 | 3.5312 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6558 | 0.1684 |
| associative_recall/difficulty-4 | 3.7170 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6528 | 0.1206 |
| compression/difficulty-1 | 2.3397 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6632 | 0.0667 |
| compression/difficulty-2 | 2.2052 | 0.0400 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6633 | 0.2933 |
| compression/difficulty-3 | 2.3091 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6628 | 0.1867 |
| compression/difficulty-4 | 2.3236 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6628 | 0.1733 |
| fuzzy_recall/difficulty-1 | 1.8145 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6621 | 0.4883 |
| fuzzy_recall/difficulty-2 | 1.8198 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6616 | 0.5272 |
| fuzzy_recall/difficulty-3 | 1.8324 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6604 | 0.4856 |
| fuzzy_recall/difficulty-4 | 1.9545 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6589 | 0.4306 |
| language/difficulty-1 | 1.5050 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6478 | 0.5575 |
| language/difficulty-2 | 1.5040 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6467 | 0.5663 |
| language/difficulty-3 | 1.4484 | 0.0000 | 0.0021 | 0.0000 | 0.5000 | 0.4979 | 0.6482 | 0.5872 |
| language/difficulty-4 | 1.4639 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6493 | 0.5717 |
| program_execution/difficulty-1 | 2.8173 | 0.1200 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6617 | 0.2600 |
| program_execution/difficulty-2 | 2.6131 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6615 | 0.2600 |
| program_execution/difficulty-3 | 2.8525 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6622 | 0.0800 |
| program_execution/difficulty-4 | 2.8933 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6623 | 0.0400 |
| selective_copying/difficulty-1 | 4.0189 | 0.0000 | 0.0000 | 0.0903 | 0.5000 | 0.4097 | 0.6614 | 0.1440 |
| selective_copying/difficulty-2 | 3.9753 | 0.0000 | 0.0000 | 0.0524 | 0.5000 | 0.4476 | 0.6588 | 0.1200 |
| selective_copying/difficulty-3 | 3.6678 | 0.0000 | 0.0019 | 0.0604 | 0.5000 | 0.4377 | 0.6598 | 0.1309 |
| selective_copying/difficulty-4 | 3.6649 | 0.0000 | 0.0000 | 0.0212 | 0.5000 | 0.4788 | 0.6572 | 0.1257 |
| stateful_action/difficulty-1 | 2.6840 | 0.0800 | 0.0000 | 0.1600 | 0.5000 | 0.3400 | 0.6645 | 0.0800 |
| stateful_action/difficulty-2 | 4.3864 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6664 | 0.0133 |
| stateful_action/difficulty-3 | 5.8675 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6515 | 0.0000 |
| stateful_action/difficulty-4 | 3.3718 | 0.0000 | 0.0400 | 0.0000 | 0.5000 | 0.4600 | 0.6603 | 0.2400 |
| symbolic/difficulty-1 | 2.4482 | 0.2400 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6602 | 0.2400 |
| symbolic/difficulty-2 | 2.6980 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6505 | 0.2800 |
| symbolic/difficulty-3 | 2.4053 | 0.2000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6629 | 0.2000 |
| symbolic/difficulty-4 | 6.1421 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6643 | 0.0000 |
| working_memory/difficulty-1 | 3.8878 | 0.1600 | 0.0000 | 0.1800 | 0.5000 | 0.3200 | 0.6681 | 0.1800 |
| working_memory/difficulty-2 | 3.8711 | 0.0800 | 0.0000 | 0.1436 | 0.5000 | 0.3564 | 0.6674 | 0.1200 |
| working_memory/difficulty-3 | 3.7751 | 0.0400 | 0.0000 | 0.1510 | 0.5000 | 0.3490 | 0.6662 | 0.0800 |
| working_memory/difficulty-4 | 3.7507 | 0.0400 | 0.0000 | 0.1201 | 0.5000 | 0.3799 | 0.6652 | 0.0600 |

# Nexus Analysis

Selection frequency is a slot count; router probability, normalized selected weight, and Integrator acceptance are distinct metrics.

## Per-Family Routing Metrics

| row | chunk_selection_fraction | effectively_never_used_capability_fraction | mean_integrator_acceptance | mean_integrator_token_contribution | mean_normalized_selected_weight | mean_router_probability_before_top_k | request_selection_fraction | routing_unit_selection_fraction | selection_frequency | selection_slot_distribution |
|---|---|---|---|---|---|---|---|---|---|---|
| delta | 0.0019 | 0.9000 | 0.3114 | 4.6662 | 0.4016 | 0.1658 | 0.0080 | 0.0019 | 0.0009 | {'1': 1.0} |
| gpt | 0.0388 | 0.7000 | 0.3314 | 3.5330 | 0.4167 | 0.1693 | 0.0840 | 0.0388 | 0.0194 | {'1': 1.0} |
| recurrent | 1.0000 | 0.0000 | 0.6039 | 8.2279 | 0.6278 | 0.4195 | 1.0000 | 1.0000 | 0.5000 | {'0': 1.0} |
| ssm | 0.9593 | 0.0000 | 0.3993 | 6.1433 | 0.3693 | 0.2454 | 0.9740 | 0.9593 | 0.4797 | {'1': 1.0} |

## Routing Frequency by Capability

| row | delta | gpt | recurrent | ssm |
|---|---|---|---|---|
| arithmetic | 0.0000 | 0.0022 | 0.5000 | 0.4978 |
| associative_recall | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| compression | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| fuzzy_recall | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| language | 0.0006 | 0.0000 | 0.5000 | 0.4994 |
| program_execution | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| selective_copying | 0.0005 | 0.0500 | 0.5000 | 0.4495 |
| stateful_action | 0.0115 | 0.0308 | 0.5000 | 0.4577 |
| symbolic | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| working_memory | 0.0000 | 0.1399 | 0.5000 | 0.3601 |

## Mean Router Probability Before Top-K by Capability

| row | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| arithmetic | 0.1576 | 0.2467 | 0.4395 | 0.1562 |
| associative_recall | 0.1519 | 0.2560 | 0.4434 | 0.1487 |
| compression | 0.1719 | 0.2491 | 0.4084 | 0.1706 |
| fuzzy_recall | 0.1595 | 0.2556 | 0.4267 | 0.1582 |
| language | 0.1686 | 0.2347 | 0.4332 | 0.1635 |
| program_execution | 0.1709 | 0.2466 | 0.4094 | 0.1731 |
| selective_copying | 0.1691 | 0.2466 | 0.4215 | 0.1628 |
| stateful_action | 0.1797 | 0.2400 | 0.4056 | 0.1747 |
| symbolic | 0.1696 | 0.2457 | 0.4155 | 0.1692 |
| working_memory | 0.1937 | 0.2329 | 0.3919 | 0.1814 |

## Mean Normalized Selected Weight by Capability

| row | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| arithmetic | 0.4210 | 0.3613 | 0.6386 | unavailable |
| associative_recall | unavailable | 0.3664 | 0.6336 | unavailable |
| compression | unavailable | 0.3794 | 0.6206 | unavailable |
| fuzzy_recall | unavailable | 0.3751 | 0.6249 | unavailable |
| language | unavailable | 0.3518 | 0.6481 | 0.3941 |
| program_execution | unavailable | 0.3762 | 0.6238 | unavailable |
| selective_copying | 0.3937 | 0.3698 | 0.6274 | 0.4100 |
| stateful_action | 0.4178 | 0.3695 | 0.6239 | 0.4015 |
| symbolic | unavailable | 0.3719 | 0.6281 | unavailable |
| working_memory | 0.4322 | 0.3725 | 0.6088 | unavailable |

## Router Collapse

Status: `slot_monopoly`

| row | average_unique_modules_per_request | chunk_routing_concentration | distribution | effective_module_count | mean_post_top_k_routing_entropy_nats | mean_pre_top_k_routing_entropy_nats | minimum_utilization | most_common_request_module_set | most_common_request_module_set_fraction | near_dead_experts | near_universal_experts | normalized_entropy | number_near_dead_experts | number_near_universal_experts | observations | post_top_k_utilization_entropy_nats | request_pool_concentration | request_selection_fraction | requests | routing_unit_selection_fraction | routing_units | selection_frequency | top_1_concentration | top_2_concentration | top_k_slot_occupancy | top_module | utilization_gini |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 2.0660 | 0.5000 | [0.01941294196130754, 0.47965310206804534, 0.5, 0.0009339559706470981] | 2.1859 | 0.6589 | 1.3013 | 0.0009 | [1, 2] | 0.9090 | [3] | [1, 2] | 0.5641 | 1 | 2 | 14990 | 0.7820 | 0.2500 | [0.084, 0.974, 1.0, 0.008] | 1000 | [0.03882588392261508, 0.9593062041360907, 1.0, 0.0018679119412941961] | 7495 | [0.01941294196130754, 0.47965310206804534, 0.5, 0.0009339559706470981] | 0.5000 | 0.9797 | {'0': [0.0, 0.0, 1.0, 0.0], '1': [0.03882588392261508, 0.9593062041360907, 0.0, 0.0018679119412941961]} | 2 | 0.4894 |

Configured thresholds:

- `global_fixed_set_request_fraction`: `0.9500`
- `low_utilization_entropy`: `0.3500`
- `mild_imbalance_gini`: `0.2000`
- `near_dead_selection_frequency`: `0.0100`
- `near_universal_request_fraction`: `0.9500`
- `partial_collapse_gini`: `0.4500`

Status definitions:

- `global_collapse`: at least top-k experts are near-universal, the same request set dominates, and effective utilization is close to top-k
- `healthy`: no configured imbalance or liveness threshold crossed
- `mild_imbalance`: utilization Gini crosses the mild threshold or at least one expert is near-dead
- `partial_collapse`: strong utilization imbalance, low entropy, or at least half the experts are near-dead
- `slot_monopoly`: at least one expert appears in the configured near-universal fraction of requests while other top-k slots may still vary

# Integrator Analysis

## Acceptance by Capability

| row | delta | gpt | recurrent | ssm |
|---|---|---|---|---|
| arithmetic | unavailable | 0.4026 | 0.6321 | 0.3679 |
| associative_recall | unavailable | unavailable | 0.6047 | 0.3953 |
| compression | unavailable | unavailable | 0.5406 | 0.4594 |
| fuzzy_recall | unavailable | unavailable | 0.5607 | 0.4393 |
| language | 0.3750 | unavailable | 0.5862 | 0.4139 |
| program_execution | unavailable | unavailable | 0.6152 | 0.3848 |
| selective_copying | 0.3101 | 0.2908 | 0.6198 | 0.3894 |
| stateful_action | 0.3011 | 0.3782 | 0.5982 | 0.4070 |
| symbolic | unavailable | unavailable | 0.6650 | 0.3350 |
| working_memory | unavailable | 0.3405 | 0.6165 | 0.4018 |

## Nexus Selection versus Integrator Acceptance

Flagged rows: 0

# Causal Ablations

Sign conventions:

- `exact_accuracy_delta`: intervened minus baseline; negative means ablation hurt
- `loss_increase`: intervened cross-entropy minus baseline; positive means ablation hurt
- `perplexity_increase`: intervened minus baseline; positive means ablation hurt
- `token_accuracy_degradation`: baseline minus intervened; positive means ablation hurt
- `token_accuracy_delta`: intervened minus baseline; negative means ablation hurt

## Family Removal: Mean Loss Increase

| row | delta | gpt | recurrent | ssm |
|---|---|---|---|---|
| arithmetic | unavailable | unavailable | 0.2997 | 0.2972 |
| associative_recall | unavailable | unavailable | 0.9718 | 0.2100 |
| compression | unavailable | unavailable | 0.7346 | 0.4296 |
| fuzzy_recall | unavailable | unavailable | 0.2800 | 0.6741 |
| language | unavailable | unavailable | 0.3308 | 0.2848 |
| program_execution | unavailable | unavailable | 1.0055 | 0.0987 |
| selective_copying | unavailable | -0.0056 | 0.9511 | 0.0192 |
| stateful_action | unavailable | 0.8825 | 1.4199 | 0.0665 |
| symbolic | unavailable | unavailable | 0.8660 | 0.0686 |
| working_memory | unavailable | -0.0320 | 0.2629 | -0.2578 |

## Family Removal: Mean Token-Accuracy Degradation

| row | delta | gpt | recurrent | ssm |
|---|---|---|---|---|
| arithmetic | unavailable | unavailable | 0.0104 | -0.0312 |
| associative_recall | unavailable | unavailable | 0.0952 | 0.1190 |
| compression | unavailable | unavailable | 0.2083 | 0.1250 |
| fuzzy_recall | unavailable | unavailable | 0.1354 | 0.2101 |
| language | unavailable | unavailable | 0.1166 | 0.1064 |
| program_execution | unavailable | unavailable | 0.1875 | -0.0000 |
| selective_copying | unavailable | -0.0000 | 0.0043 | 0.0071 |
| stateful_action | unavailable | -0.0000 | 0.0625 | 0.0312 |
| symbolic | unavailable | unavailable | -0.0000 | -0.0000 |
| working_memory | unavailable | -0.0000 | -0.1875 | -0.0000 |

## Family Removal: Mean Exact-Accuracy Delta

| row | delta | gpt | recurrent | ssm |
|---|---|---|---|---|
| arithmetic | unavailable | unavailable | 0.0000 | 0.0000 |
| associative_recall | unavailable | unavailable | 0.0000 | 0.0000 |
| compression | unavailable | unavailable | 0.0000 | 0.0000 |
| fuzzy_recall | unavailable | unavailable | 0.0000 | 0.0000 |
| language | unavailable | unavailable | 0.0000 | 0.0000 |
| program_execution | unavailable | unavailable | 0.0000 | 0.0000 |
| selective_copying | unavailable | 0.0000 | 0.0000 | 0.0000 |
| stateful_action | unavailable | 0.0000 | 0.0000 | 0.0000 |
| symbolic | unavailable | unavailable | 0.0000 | 0.0000 |
| working_memory | unavailable | 0.0000 | 0.1250 | 0.0000 |

## Specialization Criteria

- `insufficient_evidence`: fewer than 8 active family-capability cells or fewer than 2 capabilities with active family comparisons
- `meaningful_family_capability_effect`: at least 2 active interventions, at least 50% measurably worsened, and mean loss increase >=0.02 nats or mean token-accuracy degradation >=0.02
- `moderate`: at least 2 capabilities have meaningful, margin-separated effects with at least 2 different dominant families and >=3 interventions per dominant cell
- `strong`: at least 3 capabilities have margin-separated effects with at least 2 dominant families, >=5 interventions per dominant cell, and >=75% measurable worsening
- `weak`: a meaningful capability-specific contrast or differential effect exists, but family diversity/sample consistency is limited

# Surface-vs-Computation Analysis

operation correlates more strongly than surface; the larger normalized association is very_weak, which is descriptive rather than evidence of causal specialization.

Samples: `14990`; routing variable: `selected family per top-k slot`.
Operation MI/NMI: `0.036807` / `0.027176`; surface MI/NMI: `0.004855` / `0.004426`.

# Temporal/Lease Analysis

| row | average_lease_length | lease_state_norm | median_lease_length | same_module_continuation_probability | state_change_per_chunk | state_reset_count | switches_per_chunk | switches_per_request |
|---|---|---|---|---|---|---|---|---|
| overall | 7.4140 | 6.8479 | 5.0000 | 0.8592 | 7.0436 | 2110 | 0.1408 | 1.0550 |

# Sparse Execution

| row | active_flops | actual_modules_executed_per_chunk | approximate_active_parameter_fraction_per_forward | approximate_active_parameter_uses_per_forward | average_module_population_touched_per_request | chunk_module_computations_executed | chunk_module_computations_requested | discarded_module_computations | estimated_module_compute_relative_to_executing_all_modules | large_discrepancy | logical_top_k | mean_actively_executed_module_parameters_per_routing_unit | mean_fraction_total_parameters_actively_executed_in_modules | mean_fraction_total_parameters_in_selected_module_population | mean_selected_module_parameter_count_per_request | requested_executed_discrepancy_fraction | route_weighted_module_parameters_per_selected_slot | total_model_parameter_count | total_routable_module_parameter_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | unavailable | 2.0000 | 0.7437 | 311322 | 0.5165 | 14990 | 14990 | 0 | 0.5087 | False | 2 | 109193.3993 | 0.2608 | 0.2688 | 112511.1040 | 0.0000 | 54596.6997 | 418638 | 214632 |

# Cycle Analysis

Supported: `False`

Reason: n1_chunked exposes chunk recurrence and lease telemetry, but does not expose independently repeatable EMC cycles or a cycle-limit intervention

# Module Diagnostics

| row | execution_context | gradients_captured | note | optimizer_state_available | update_norms_available |
|---|---|---|---|---|---|
| availability | checkpoint_or_evaluation_only | False | Evaluation runs under torch.inference_mode; absent tensors are reported as unavailable, never as numeric zero. | False | False |

| row | family | gradient_norm | gradient_source | module | parameter_norm | update_norm |
|---|---|---|---|---|---|---|
| 0 | gpt | unavailable | unavailable | 0 | 20.9660 | unavailable |
| 1 | ssm | unavailable | unavailable | 1 | 19.9790 | unavailable |
| 2 | recurrent | unavailable | unavailable | 2 | 19.5018 | unavailable |
| 3 | delta | unavailable | unavailable | 3 | 20.2449 | unavailable |

# Notable Examples

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `watch the stars from the hill.`
- Prediction: `wateh ohe mcars foow the ritl `
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.41511499881744385, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -19.999998807907104, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.1717360019683838, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -3.3333301544189453, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.41511499881744385, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -19.999998807907104, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.1717360019683838, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -3.3333301544189453, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.6672040224075317, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -23.33333194255829, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.7461738586425781, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -26.666665077209473, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.1717360019683838, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -3.3333301544189453, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.3573344945907593, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": -16.66666269302368, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
the moon is bright tonight. we
```

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `from the hill. a quiet river flows under the old bridge.`
- Prediction: `boor the sitl  ttbunt  tiner ttows onier the swe bligeh `
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.2537088394165039, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -5.357140302658081, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.2984170913696289, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -10.7142835855484, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.2537088394165039, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -5.357140302658081, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.2984170913696289, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -10.7142835855484, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.4723637104034424, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -10.7142835855484, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.8316164016723633, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -14.28571343421936, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.2984170913696289, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -10.7142835855484, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.2998971939086914, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": -1.7857134342193604, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
the little boat returns before sunset. we watch the stars
```

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `the small fox runs through the garden. the red bird sings in the morning.`
- Prediction: `the rtall ooo rens theounh the maldhm  the mit brte mcns  tn the mooeen  `
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.2863771915435791, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -8.219176530838013, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.4454437494277954, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -12.328764796257019, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.2863771915435791, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -8.219176530838013, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.4454437494277954, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -12.328764796257019, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.4412198066711426, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -6.849312782287598, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.6785839796066284, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -13.698628544807434, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.4454437494277954, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -12.328764796257019, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.41905665397644043, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": -13.698628544807434, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
we watch the stars from the hill. we watch the stars from the hill.
```

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `runs through the garden. the small fox runs through the garden. the moon is bright tonight.`
- Prediction: `tens theirehethe rots r  the rtatl too rens theoug  the matdems the moowstt wreg   thneg.t `
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.31771934032440186, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -7.692307233810425, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.3677266836166382, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -17.582419514656067, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.31771934032440186, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -7.692307233810425, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.3677266836166382, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -17.582419514656067, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.49123919010162354, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -12.08791434764862, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.6760392189025879, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -13.18681538105011, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.3677266836166382, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -17.582419514656067, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.3955957889556885, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": -16.483518481254578, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
the small fox runs through the garden. the little boat returns before sunset. the small fox
```

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `the red bird sings in the morning.`
- Prediction: `Ohe mir brtd titss tn the rooesng `
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.13747799396514893, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -11.764705181121826, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.4134030342102051, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -17.64705777168274, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.13747799396514893, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -11.764705181121826, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.4134030342102051, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -17.64705777168274, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.40920305252075195, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -8.82352888584137, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.7425755262374878, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -23.529410362243652, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.4134030342102051, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -17.64705777168274, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.40671396255493164, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": -11.764705181121826, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
the small fox runs through the garden.
```

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `before sunset. a quiet river flows under the old bridge.`
- Prediction: `tirnre tcns   
ttxunnt titer too s onser the mwi rrig t `
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.251051664352417, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -5.357140302658081, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.289569616317749, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -7.14285671710968, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.251051664352417, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -5.357140302658081, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.289569616317749, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -7.14285671710968, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.5187810659408569, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -8.92857015132904, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.7521780729293823, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -16.07142686843872, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.289569616317749, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -7.14285671710968, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.2356576919555664, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": 1.7857164144515991, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
the little boat returns before sunset. the little boat returns
```

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `bridge. we watch the stars from the hill. the small fox runs through the garden.`
- Prediction: `brin h  tn wnt h the mcals toow the retl  the rcall too rens rheoueh the mrtders`
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.325710654258728, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -11.250001192092896, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.3589397668838501, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -16.25000238418579, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.325710654258728, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -11.250001192092896, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.3589397668838501, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -16.25000238418579, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.5342166423797607, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -12.5, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.7588953971862793, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -17.500001192092896, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.3589397668838501, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -16.25000238418579, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.4523388147354126, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": -20.000001788139343, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
the small fox runs through the garden. a quiet river flows under the old
```

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `flows under the old bridge. a quiet river flows under the old bridge. the moon is bright tonight.`
- Prediction: `toows bnier the mbe biog h  ttbuntt tiner toows rnser the rwi brig    the mornsrt wrenh  thneght `
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.2908698320388794, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -15.463915467262268, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.30103111267089844, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -8.247420191764832, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.2908698320388794, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -15.463915467262268, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.30103111267089844, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -8.247420191764832, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.5813080072402954, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -19.587627053260803, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.8566874265670776, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -20.618554949760437, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.30103111267089844, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -8.247420191764832, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.30688416957855225, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": -13.402059674263, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
we watch the stars from the hill. the red bird sings in the morning. a quiet river
```

## causally important module intervention

- Capability: `associative_recall`
- Surface: `english`
- Target: `V0G`
- Prediction: `V"
`
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.37096428871154785, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -33.33333432674408, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.5047881603240967, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": 0.0, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.37096428871154785, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -33.33333432674408, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.5047881603240967, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.4207723140716553, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -33.33333432674408, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.6427590847015381, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -33.33333432674408, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.5047881603240967, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": -0.07703590393066406, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
K59 maps to VS8P.
K09 maps to V0G.
K46 maps to VF.
K26 maps to V6.
Value for K09?
```

## causally important module intervention

- Capability: `associative_recall`
- Surface: `structured`
- Target: `VA0-7Q4`
- Prediction: `VV--R--`
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.41596198081970215, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -14.28571492433548, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.4642646312713623, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": 0.0, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.41596198081970215, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -14.28571492433548, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.4642646312713623, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.4708824157714844, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -28.57142984867096, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.735342264175415, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -14.28571492433548, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.4642646312713623, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.4749643802642822, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
K56:VAJ
K57:VE-R6
K19:V5
K19:VA0-7Q4
K14:V1YM
K12:V0PJ-R
K68:VO13-OP8
K44:V5-3K
K59:VH62-JP
query K19=
```

## causally important module intervention

- Capability: `associative_recall`
- Surface: `json`
- Target: `VG8-6-U`
- Prediction: `V7--K-R`
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.06608176231384277, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -14.28571343421936, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 1.1323318481445312, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -42.85714328289032, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.06608176231384277, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -14.28571343421936, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 1.1323318481445312, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -42.85714328289032, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.17299723625183105, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": 0.0, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 1.0104789733886719, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -42.85714328289032, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 1.317002773284912, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -42.85714328289032, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 1.1323318481445312, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": -42.85714328289032, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
{"K66":"VR-7IC","K65":"VON-I7-C4","K53":"VB-1NA","K61":"VG8-6-U","K57":"VBK-P","K85":"VUO-IDG","K36":"V8YM-2","K32":"V87E-QPD-2S","K12":"V47-JB","K24":"V7O0","K63":"VN2-Z2-7LY","K19":"V7F-RJS-3","K99":"VZ","K54":"VB-IZ","K44":"VAQ-FH-9","K03":"V2FM-G"}
lookup(K61)=
```

## causally important module intervention

- Capability: `associative_recall`
- Surface: `symbolic`
- Target: `VRZ`
- Prediction: `V53`
- Causal effects: `[{"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "0", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.43367862701416016, "status": "active_intervention", "target": "1", "token_accuracy_delta_points": -33.33333432674408, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.7585101127624512, "status": "active_intervention", "target": "2", "token_accuracy_delta_points": -33.33333432674408, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_module", "loss_increase": 0.0, "status": "target_not_selected", "target": "3", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.43367862701416016, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -33.33333432674408, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.7585101127624512, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -33.33333432674408, "validation": "target was active at baseline and absent from intervened routing"}, {"exact_delta_points": 0.0, "intervention": "disable_family", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "gpt", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.5478188991546631, "status": "active_intervention", "target": "ssm", "token_accuracy_delta_points": -33.33333432674408, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 1.4970703125, "status": "active_intervention", "target": "recurrent", "token_accuracy_delta_points": -33.33333432674408, "validation": "selected target proposal was zero at the Integrator input"}, {"exact_delta_points": 0.0, "intervention": "zero_family_proposal", "loss_increase": 0.0, "status": "target_not_selected", "target": "delta", "token_accuracy_delta_points": 0.0, "validation": "target was absent from the baseline active forward path"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.7585101127624512, "status": "active_intervention", "target": "gpt", "token_accuracy_delta_points": -33.33333432674408, "validation": "forced family appeared and routing counts or slots changed"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.0, "status": "intervention_no_effect", "target": "recurrent", "token_accuracy_delta_points": 0.0, "validation": "forced routing matched baseline counts and slot occupancy"}, {"exact_delta_points": 0.0, "intervention": "force_family_alternative", "loss_increase": 0.8404655456542969, "status": "active_intervention", "target": "delta", "token_accuracy_delta_points": -33.33333432674408, "validation": "forced family appeared and routing counts or slots changed"}]`

```text
K37→VDJ K74→VRZ K06→V7 K52→VKU
get(K74)=
```
