# Executive Summary

- Learning: `no_clear_learning_signal`
- Overall loss: `4.2790`
- Language perplexity: `42.5100`
- Strongest / weakest: `selective_copying` / `associative_recall`
- Router collapse: `no_global_collapse`
- Specialization: No measurable causal family specialization in sampled ablations.
- Surface vs computation: Operation and surface format have similar association with module selection.
- Cycles: checkpoint does not expose multiple EMC cycles

# Overall Metrics

| row | examples | language_perplexity | overall_exact_accuracy | overall_loss | overall_token_accuracy | skipped |
|---|---|---|---|---|---|---|
| overall | 200 | 42.5100 | 0.0000 | 4.2790 | 0.0489 | 0 |

# Capability Results

| row | exact_accuracy | token_accuracy | cross_entropy | perplexity |
|---|---|---|---|---|
| language | 0.0000 | 0.1637 | 3.7497 | 42.5100 |
| associative_recall | 0.0000 | 0.0000 | 4.7599 | 116.7339 |
| fuzzy_recall | 0.0000 | 0.1090 | 4.0087 | 55.0751 |
| selective_copying | 0.0000 | 0.2162 | 4.2725 | 71.7009 |
| working_memory | 0.0000 | 0.0000 | 4.1938 | 66.2769 |
| compression | 0.0000 | 0.0000 | 4.3047 | 74.0482 |
| arithmetic | 0.0000 | 0.0000 | 4.1865 | 65.7929 |
| symbolic | 0.0000 | 0.0000 | 4.4477 | 85.4292 |
| program_execution | 0.0000 | 0.0000 | 4.2913 | 73.0640 |
| stateful_action | 0.0000 | 0.0000 | 4.7189 | 112.0440 |

# Generalization and Difficulty Curves

| row | cross_entropy | exact_accuracy | route_delta | route_gpt | route_recurrent | route_ssm | routing_entropy | token_accuracy |
|---|---|---|---|---|---|---|---|---|
| language/difficulty-1 | 3.7771 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.1570 |
| language/difficulty-2 | 3.7311 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6930 | 0.1663 |
| language/difficulty-3 | 3.7506 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.1698 |
| language/difficulty-4 | 3.7401 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6930 | 0.1619 |
| associative_recall/difficulty-1 | 4.8932 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| associative_recall/difficulty-2 | 4.7459 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6930 | 0.0000 |
| associative_recall/difficulty-3 | 4.6108 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6930 | 0.0000 |
| associative_recall/difficulty-4 | — | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6929 | 0.0000 |
| fuzzy_recall/difficulty-1 | 3.9413 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6930 | 0.1194 |
| fuzzy_recall/difficulty-2 | 4.0209 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6930 | 0.1194 |
| fuzzy_recall/difficulty-3 | 3.9393 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6930 | 0.1250 |
| fuzzy_recall/difficulty-4 | 4.1333 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6929 | 0.0722 |
| selective_copying/difficulty-1 | 4.4747 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.2000 |
| selective_copying/difficulty-2 | 4.2424 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.2000 |
| selective_copying/difficulty-3 | 4.1488 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.2364 |
| selective_copying/difficulty-4 | 4.2121 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6930 | 0.2286 |
| working_memory/difficulty-1 | 4.2137 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| working_memory/difficulty-2 | 4.1030 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| working_memory/difficulty-3 | 4.1577 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| working_memory/difficulty-4 | 4.3009 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| compression/difficulty-1 | 4.3391 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6930 | 0.0000 |
| compression/difficulty-2 | 4.2531 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6929 | 0.0000 |
| compression/difficulty-3 | 4.3224 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6929 | 0.0000 |
| compression/difficulty-4 | 4.3043 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6929 | 0.0000 |
| arithmetic/difficulty-1 | 4.0590 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| arithmetic/difficulty-2 | 4.1788 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| arithmetic/difficulty-3 | 4.5899 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| arithmetic/difficulty-4 | 3.9183 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| symbolic/difficulty-1 | 4.3697 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| symbolic/difficulty-2 | 4.2272 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| symbolic/difficulty-3 | 4.4820 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| symbolic/difficulty-4 | 4.7119 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| program_execution/difficulty-1 | 4.1661 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| program_execution/difficulty-2 | 4.2176 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| program_execution/difficulty-3 | 4.4397 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| program_execution/difficulty-4 | 4.3420 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| stateful_action/difficulty-1 | 4.5520 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| stateful_action/difficulty-2 | 4.7567 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| stateful_action/difficulty-3 | 5.7076 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |
| stateful_action/difficulty-4 | 3.8592 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | 0.0000 |

# Nexus Analysis

## Routing Frequency by Capability

| row | delta | gpt | recurrent | ssm |
|---|---|---|---|---|
| language | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| associative_recall | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| fuzzy_recall | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| selective_copying | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| working_memory | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| compression | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| arithmetic | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| symbolic | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| program_execution | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| stateful_action | 0.0000 | 0.5000 | 0.0000 | 0.5000 |

## Routing Frequency by Surface

| row | delta | gpt | recurrent | ssm |
|---|---|---|---|---|
| english | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| structured | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| json | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| code | 0.0000 | 0.5000 | 0.0000 | 0.5000 |
| symbolic | 0.0000 | 0.5000 | 0.0000 | 0.5000 |

# Integrator Analysis

## Acceptance by Capability

| row | gpt | ssm |
|---|---|---|
| language | 0.4343 | 0.5657 |
| associative_recall | 0.4123 | 0.5877 |
| fuzzy_recall | 0.4081 | 0.5919 |
| selective_copying | 0.4259 | 0.5741 |
| working_memory | 0.4403 | 0.5597 |
| compression | 0.4144 | 0.5856 |
| arithmetic | 0.4498 | 0.5502 |
| symbolic | 0.4635 | 0.5365 |
| program_execution | 0.4466 | 0.5534 |
| stateful_action | 0.4310 | 0.5690 |

## Nexus Selection versus Integrator Acceptance

Flagged rows: 0

# Causal Ablations

## Performance Drop When Family Removed

| row | delta | gpt | recurrent | ssm |
|---|---|---|---|---|
| language | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| associative_recall | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fuzzy_recall | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| selective_copying | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| working_memory | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compression | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| arithmetic | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| symbolic | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| program_execution | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stateful_action | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

# Surface-vs-Computation Analysis

Operation and surface format have similar association with module selection.

Operation MI: `0.000000`; surface MI: `0.000000`.

# Temporal/Lease Analysis

| row | average_lease_length | lease_state_norm | median_lease_length | same_module_continuation_probability | state_change_per_chunk | state_reset_count | switches_per_chunk | switches_per_request |
|---|---|---|---|---|---|---|---|---|
| overall | 7.4360 | 6.4099 | 5.0000 | 0.8660 | 2.5397 | 400 | 0.1340 | 1.0000 |

# Sparse Execution

| row | active_flops | actual_modules_executed_per_chunk | approximate_active_parameter_uses_per_forward | average_module_population_touched_per_request | chunk_module_computations_executed | chunk_module_computations_requested | discarded_module_computations | large_discrepancy | logical_top_k | requested_executed_discrepancy_fraction | route_weighted_module_parameters_per_selected_slot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | — | 2.0000 | 311322 | 0.5000 | 2986 | 2986 | 0 | False | 2 | 0.0000 | 52880.0000 |

# Cycle Analysis

Supported: `False`

# Module Diagnostics

| row | family | gradient_norm | module | parameter_norm | update_norm |
|---|---|---|---|---|---|
| 0 | gpt | 0.0000 | 0 | 20.4397 | — |
| 1 | ssm | 0.0000 | 1 | 18.2231 | — |
| 2 | recurrent | 0.0000 | 2 | 17.4401 | — |
| 3 | delta | 0.0000 | 3 | 19.7385 | — |

# Notable Examples

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `watch the stars from the hill.`
- Prediction: `te  e  e       e o            `
- Causal effects: `[{"intervention": "disable_module", "target": "0", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_module", "target": "1", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_module", "target": "2", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_module", "target": "3", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "gpt", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "ssm", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "recurrent", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "delta", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "zero_family_proposal", "target": "gpt", "exact_delta_points": 0.0, "token_accuracy_delta_points": -3.333333134651184}, {"intervention": "zero_family_proposal", "target": "ssm", "exact_delta_points": 0.0, "token_accuracy_delta_points": -3.333333134651184}, {"intervention": "zero_family_proposal", "target": "recurrent", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "zero_family_proposal", "target": "delta", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "force_family_alternative", "target": "ssm", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "force_family_alternative", "target": "recurrent", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "force_family_alternative", "target": "delta", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}]`

```text
the moon is bright tonight. we
```

## causally important module intervention

- Capability: `language`
- Surface: `english`
- Target: `from the hill. a quiet river flows under the old bridge.`
- Prediction: `                                                        `
- Causal effects: `[{"intervention": "disable_module", "target": "0", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_module", "target": "1", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_module", "target": "2", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_module", "target": "3", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "gpt", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "ssm", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "recurrent", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "delta", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "zero_family_proposal", "target": "gpt", "exact_delta_points": 0.0, "token_accuracy_delta_points": -3.57142835855484}, {"intervention": "zero_family_proposal", "target": "ssm", "exact_delta_points": 0.0, "token_accuracy_delta_points": -1.7857134342193604}, {"intervention": "zero_family_proposal", "target": "recurrent", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "zero_family_proposal", "target": "delta", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "force_family_alternative", "target": "ssm", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "force_family_alternative", "target": "recurrent", "exact_delta_points": 0.0, "token_accuracy_delta_points": -1.7857134342193604}, {"intervention": "force_family_alternative", "target": "delta", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}]`

```text
the little boat returns before sunset. we watch the stars
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `structured`
- Target: `T2 U8 K9`
- Prediction: `        `
- Causal effects: `[{"intervention": "disable_module", "target": "0", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_module", "target": "1", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_module", "target": "2", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_module", "target": "3", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "gpt", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "ssm", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "recurrent", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "disable_family", "target": "delta", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "zero_family_proposal", "target": "gpt", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "zero_family_proposal", "target": "ssm", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "zero_family_proposal", "target": "recurrent", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "zero_family_proposal", "target": "delta", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "force_family_alternative", "target": "ssm", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "force_family_alternative", "target": "recurrent", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}, {"intervention": "force_family_alternative", "target": "delta", "exact_delta_points": 0.0, "token_accuracy_delta_points": 0.0}]`

```text
0:E0 0:L4 0:G9 0:D7 1:T2 0:S8 1:U8 0:N2 0:F5 1:K9
ones=
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `code`
- Target: `U1 D1 P5 J3 A2`
- Prediction: `              `
- Causal effects: `[]`

```text
pass(T1);pass(Q1);pass(B4);pass(U6);pass(H5);emit(U1);pass(T0);pass(S1);emit(D1);emit(P5);emit(J3);pass(P9);pass(S8);pass(M7);emit(A2);pass(D7)
output=
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `structured`
- Target: `L4 T2 P5 E4`
- Prediction: `           `
- Causal effects: `[]`

```text
0:C2 0:X6 0:M6 1:L4 0:V1 0:V2 1:T2 0:G4 1:P5 0:P1 1:E4 0:J4 0:L9
ones=
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `symbolic`
- Target: `S6 Z9 X3`
- Prediction: `        `
- Causal effects: `[]`

```text
-Z3 -L2 -I1 +S6 +Z9 -K4 -F5 +X3 -P9 -Y2
+sequence=
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `english`
- Target: `U9 U9 X8 S2`
- Prediction: `           `
- Causal effects: `[]`

```text
skip X9.
remember U9.
remember U9.
skip W9.
remember X8.
skip D8.
skip E0.
skip D8.
skip J1.
skip X2.
skip M2.
skip A0.
remember S2.
List remembered items:
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `structured`
- Target: `E5 D0 Z5 Y0 F4`
- Prediction: `              `
- Causal effects: `[]`

```text
0:Q3 0:J1 1:E5 1:D0 1:Z5 0:L8 0:H5 0:A0 1:Y0 0:T0 0:V6 0:M4 0:X8 0:O1 1:F4 0:S9
ones=
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `code`
- Target: `L1 S6 C8`
- Prediction: `        `
- Causal effects: `[]`

```text
pass(S8);pass(M9);pass(J3);pass(G9);pass(S2);pass(M8);pass(X0);emit(L1);emit(S6);emit(C8)
output=
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `symbolic`
- Target: `Y7 Z9 V2 Z4`
- Prediction: `           `
- Causal effects: `[]`

```text
-N9 -W7 +Y7 -O7 -B4 -V5 -D2 +Z9 -C8 -F2 -V3 +V2 +Z4
+sequence=
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `english`
- Target: `C6 I3 J1 G4 O8`
- Prediction: `              `
- Causal effects: `[]`

```text
skip A6.
skip E6.
remember C6.
skip X6.
skip A7.
remember I3.
remember J1.
skip W0.
skip B1.
remember G4.
skip W7.
skip Y2.
remember O8.
skip J0.
skip A1.
skip N5.
List remembered items:
```

## Nexus and Integrator disagree

- Capability: `selective_copying`
- Surface: `structured`
- Target: `C9 W3`
- Prediction: `     `
- Causal effects: `[]`

```text
0:E9 0:C3 0:R0 1:C9 1:W3 0:U2 0:M0
ones=
```
