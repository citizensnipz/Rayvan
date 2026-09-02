# Three-Way Architecture Comparison

No architecture is preferred by the harness. Quality and causal usefulness take precedence over balanced utilization.

| architecture | validation loss mean±std | perplexity mean±std | token accuracy mean±std | tokens/s mean±std |
|---|---:|---:|---:|---:|
| homogeneous_serial | 3.05667±0 | 21.2566±0 | 0±0 | 233.991±0 |
| old_emc | 3.41773±0 | 30.5±0 | 0±0 | 205.441±0 |
| n2_mixed | 3.07008±0 | 21.5437±0 | 0±0 | 125.338±0 |
| n2_gpt4 | 3.8489±0 | 46.9412±0 | 0±0 | 143.963±0 |
| n2_ssm4 | 3.30828±0 | 27.3381±0 | 0±0 | 116.396±0 |
| n2_recurrent4 | 3.34939±0 | 28.4855±0 | 0±0 | 144.876±0 |
| n2_delta4 | 3.31866±0 | 27.6233±0 | 0±0 | 82.5016±0 |

## Fairness mismatches

Exact parameter, active-parameter, and approximate-compute ratios are in `comparison.json` under `fairness_mismatches`. Approximate FLOPs are documented lower bounds, not hidden profiler claims.

## Developmental diagnostics

EMC milestone routing/update telemetry is stored under `emc/seed-*/telemetry.json`; optional causal reports are under `milestone-diagnostics/<tokens>/`.
