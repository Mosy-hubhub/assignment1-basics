# Checkpoint parameter-direction summary

FP64; each maximum covers ALL checkpoint pairs within its range. All = all available checkpoints; half = latter half by checkpoint count; last5 = last five. Angles are in degrees; distances compare unit vectors.

## ADAMW

| Run (weight decay) | LR schedule | Learning rate | all max_angle_deg | all max_direction_distance | half max_angle_deg | half max_direction_distance | last5 max_angle_deg | last5 max_direction_distance |
|---|---|---|---:|---:|---:|---:|---:|---:|
| tinystories_adamw_constant_no_wd_20261023_234410 (wd=0) | constant | 0.001 | 10.152694 | 0.1769662 | 6.7447191 | 0.1176496 | 3.992921 | 0.069675517 |
| tinystories_adamw_no_wd_20261023_191427 (wd=0) | warmup(1000)+cosine | 0.001 -> 0.0001 | 4.0248594 | 0.070232607 | 2.6707937 | 0.046609924 | 0.74734137 | 0.013043475 |
| tinystories_baseline_adamw_wd01_norm_20261023_232836 (wd=0.1) | warmup(1000)+cosine | 0.001 -> 0.0001 | 21.754374 | 0.3774089 | 14.936623 | 0.25995567 | 5.0812344 | 0.088655212 |
| tinystories_silu_ffn_direction_20261023_165419 (wd=0.1) | warmup(1000)+cosine | 0.001 -> 0.0001 | 20.878759 | 0.36239023 | 14.115706 | 0.24574296 | 4.9565782 | 0.086481637 |

## SGD

| Run (weight decay) | LR schedule | Learning rate | all max_angle_deg | all max_direction_distance | half max_angle_deg | half max_direction_distance | last5 max_angle_deg | last5 max_direction_distance |
|---|---|---|---:|---:|---:|---:|---:|---:|
| tinystories_sgd_constant_no_wd_20261023_192958 (wd=0) | constant | 0.001 | 0.073137016 | 0.0012764816 | 0.034418574 | 0.00060071744 | 0.0067142488 | 0.00011718575 |
| tinystories_sgd_cosine_lr100x_no_wd_20261023_232437 (wd=0) | warmup(1000)+cosine | 0.1 -> 0.01 | 0.21208745 | 0.0037016222 | 0.056087418 | 0.00097891007 | 0.010046919 | 0.00017535182 |
| tinystories_sgd_cosine_lr10x_no_wd_20261023_215703 (wd=0) | warmup(1000)+cosine | 0.01 -> 0.001 | 0.080852492 | 0.0014111421 | 0.021359402 | 0.00037279188 | 0.002967644 | 5.1795159e-05 |
| tinystories_sgd_cosine_lr4x_no_wd_20261023_215703 (wd=0) | warmup(1000)+cosine | 0.004 -> 0.0004 | 0.052487989 | 0.0009160882 | 0.013793744 | 0.00024074625 | 0.0018771838 | 3.2763038e-05 |
| tinystories_sgd_cosine_lr50x_no_wd_20261023_231924 (wd=0) | warmup(1000)+cosine | 0.05 -> 0.005 | 0.15057315 | 0.0026279964 | 0.039380445 | 0.00068731841 | 0.0062037379 | 0.00010827565 |
| tinystories_sgd_cosine_no_wd_20261023_202955 (wd=0) | warmup(1000)+cosine | 0.001 -> 0.0001 | 0.025011283 | 0.00043652924 | 0.0064399548 | 0.00011239841 | 0.00088108241 | 1.5377789e-05 |

Computed runs: 10; pair comparisons: 8200; skipped empty runs: 5.
