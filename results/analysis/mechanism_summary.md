# Mechanism analysis summary (desc w0.1, ETTh1/ETTm1, 96->96)

Raw JSONs: probe_*.json / gradflow_*.json / infoflow_*.json / gain_*.json(+ .md) in this directory.
Probe protocol: frozen backbone, test split, pooled hidden; probe train/test = time-blocked 60/40 (primary, leakage-free) and seeded shuffle (robustness).

## Task 1: last-layer probe deltas (desc - base)

| setting | split | cp_auc | cp_pos_r2 | spec_r2 | drift_acc | vol_acc | slope_acc |
|---|---|---|---|---|---|---|---|
| ETTh1_itr_s2021 | blocked | +0.0125 | -0.0234 | -0.4504 | -0.0036 | +0.0117 | -0.0395 |
| ETTh1_itr_s2021_shuffle | shuffle | +0.0337 | -0.0053 | -0.0073 | +0.0135 | +0.0341 | -0.0045 |
| ETTh1_patch_s2021 | blocked | -0.0676 | -0.0917 | -0.1559 | -0.0117 | -0.0305 | -0.0206 |
| ETTh1_patch_s2021_shuffle | shuffle | +0.0128 | -0.0370 | -0.0164 | +0.0027 | -0.0431 | -0.0224 |
| ETTh1_patch_s2022 | blocked | +0.0856 | +0.0565 | -0.0314 | +0.0081 | +0.0180 | +0.0081 |
| ETTh1_patch_s2022_shuffle | shuffle | +0.0050 | +0.0493 | -0.0300 | +0.0278 | +0.0305 | +0.0144 |
| ETTh1_patch_s2023 | blocked | +0.0382 | -0.1644 | -0.0387 | +0.0117 | -0.0215 | +0.0260 |
| ETTh1_patch_s2023_shuffle | shuffle | -0.0056 | -0.0072 | +0.0416 | +0.0108 | -0.0054 | -0.0045 |
| ETTm1_itr_s2021 | blocked | +0.0598 | -0.0057 | -0.0129 | -0.0206 | +0.0011 | -0.0142 |
| ETTm1_itr_s2021_shuffle | shuffle | +0.0191 | -0.0018 | +0.0233 | +0.0201 | +0.0298 | +0.0195 |
| ETTm1_patch_s2021 | blocked | +0.0174 | +0.0735 | -0.0181 | +0.0245 | +0.0160 | -0.0013 |
| ETTm1_patch_s2021_shuffle | shuffle | +0.0291 | +0.0476 | -0.0135 | +0.0265 | +0.0182 | +0.0333 |
| ETTm1_patch_s2022 | blocked | +0.0246 | +0.0491 | -0.1384 | -0.0092 | +0.0055 | -0.0116 |
| ETTm1_patch_s2022_shuffle | shuffle | +0.0351 | +0.0619 | -0.0317 | +0.0129 | -0.0033 | +0.0079 |
| ETTm1_patch_s2023 | blocked | -0.0116 | -0.0415 | -0.0074 | -0.0120 | +0.0232 | +0.0007 |
| ETTm1_patch_s2023_shuffle | shuffle | +0.0128 | +0.0417 | +0.0388 | +0.0361 | +0.0420 | +0.0481 |
| **MEAN** | blocked | +0.0188 | +0.0001 | -0.0530 | +0.0086 | +0.0079 | +0.0024 |
| **MEAN** | _shuffle | +0.0178 | +0.0186 | +0.0006 | +0.0188 | +0.0129 | +0.0115 |

## Task 2: per-layer cp_auc (blocked) — information vs depth

| setting | layer | base | desc | delta |
|---|---|---|---|---|
| ETTh1_itr_s2021 | last | 0.620 | 0.632 | +0.012 |
| ETTh1_itr_s2021 | emb | 0.619 | 0.619 | +0.000 |
| ETTh1_itr_s2021 | enc0 | 0.611 | 0.614 | +0.003 |
| ETTh1_itr_s2021 | enc1 | 0.620 | 0.632 | +0.013 |
| ETTh1_patch_s2021 | last | 0.589 | 0.522 | -0.068 |
| ETTh1_patch_s2021 | emb | 0.591 | 0.591 | -0.000 |
| ETTh1_patch_s2021 | enc0 | 0.590 | 0.522 | -0.068 |
| ETTh1_patch_s2022 | last | 0.494 | 0.579 | +0.086 |
| ETTh1_patch_s2022 | emb | 0.591 | 0.591 | -0.000 |
| ETTh1_patch_s2022 | enc0 | 0.496 | 0.581 | +0.086 |
| ETTh1_patch_s2023 | last | 0.546 | 0.585 | +0.038 |
| ETTh1_patch_s2023 | emb | 0.591 | 0.591 | +0.000 |
| ETTh1_patch_s2023 | enc0 | 0.545 | 0.584 | +0.040 |
| ETTm1_itr_s2021 | last | 0.614 | 0.674 | +0.060 |
| ETTm1_itr_s2021 | emb | 0.549 | 0.549 | +0.000 |
| ETTm1_itr_s2021 | enc0 | 0.645 | 0.667 | +0.023 |
| ETTm1_itr_s2021 | enc1 | 0.614 | 0.674 | +0.060 |
| ETTm1_patch_s2021 | last | 0.679 | 0.696 | +0.017 |
| ETTm1_patch_s2021 | emb | 0.519 | 0.519 | +0.000 |
| ETTm1_patch_s2021 | enc0 | 0.679 | 0.697 | +0.018 |
| ETTm1_patch_s2022 | last | 0.647 | 0.672 | +0.025 |
| ETTm1_patch_s2022 | emb | 0.519 | 0.519 | +0.000 |
| ETTm1_patch_s2022 | enc0 | 0.647 | 0.671 | +0.024 |
| ETTm1_patch_s2023 | last | 0.680 | 0.669 | -0.012 |
| ETTm1_patch_s2023 | emb | 0.519 | 0.519 | +0.000 |
| ETTm1_patch_s2023 | enc0 | 0.680 | 0.669 | -0.011 |

## Task 3: gradient norms per layer (aux/task ratio; ckpt trained at w=0.1, raw ratio at w=1)

| setting | layer | |g_task| | |g_aux| | ratio |
|---|---|---|---|---|
| ETTh1_itr_descw01 | embedding | 1.253 | 15.654 | 12.49 |
| ETTh1_itr_descw01 | aux_head | 0.000 | 32.186 | n/a |
| ETTh1_itr_descw01 | encoder.layer0 | 0.840 | 12.134 | 14.45 |
| ETTh1_itr_descw01 | encoder.layer1 | 0.979 | 12.281 | 12.54 |
| ETTh1_itr_descw01 | encoder.norm | 0.148 | 1.859 | 12.57 |
| ETTh1_itr_descw01 | forecast_head | 1.592 | 0.000 | 0.00 |
| ETTh1_itr_descw01 | **TOTAL** | 2.406 | 39.781 | **16.53** |
| ETTh1_patch_descw01 | embedding | 0.328 | 3.188 | 9.70 |
| ETTh1_patch_descw01 | aux_head | 0.000 | 27.467 | n/a |
| ETTh1_patch_descw01 | encoder.layer0 | 1.796 | 16.837 | 9.37 |
| ETTh1_patch_descw01 | encoder.norm | 0.125 | 1.986 | 15.94 |
| ETTh1_patch_descw01 | forecast_head | 10.262 | 0.000 | 0.00 |
| ETTh1_patch_descw01 | **TOTAL** | 10.424 | 32.435 | **3.11** |
| ETTm1_itr_descw01 | embedding | 1.807 | 40.653 | 22.50 |
| ETTm1_itr_descw01 | aux_head | 0.000 | 46.649 | n/a |
| ETTm1_itr_descw01 | encoder.layer0 | 1.757 | 43.672 | 24.86 |
| ETTm1_itr_descw01 | encoder.layer1 | 2.090 | 53.184 | 25.45 |
| ETTm1_itr_descw01 | encoder.norm | 0.146 | 3.709 | 25.32 |
| ETTm1_itr_descw01 | forecast_head | 1.174 | 0.000 | 0.00 |
| ETTm1_itr_descw01 | **TOTAL** | 3.482 | 92.619 | **26.60** |
| ETTm1_patch_descw01 | embedding | 0.482 | 8.937 | 18.56 |
| ETTm1_patch_descw01 | aux_head | 0.000 | 34.362 | n/a |
| ETTm1_patch_descw01 | encoder.layer0 | 2.698 | 50.682 | 18.79 |
| ETTm1_patch_descw01 | encoder.norm | 0.146 | 5.233 | 35.86 |
| ETTm1_patch_descw01 | forecast_head | 7.506 | 0.000 | 0.00 |
| ETTm1_patch_descw01 | **TOTAL** | 7.992 | 62.102 | **7.77** |

## Task 4: z interventions on cond models (test MSE)

| setting | clean | zero-z | perm-z | d_zero | d_perm |
|---|---|---|---|---|---|
| ETTh1_itr_condc | 0.3968 | 0.4039 | 0.3975 | +0.00710 | +0.00079 |
| ETTh1_itr_condc1 | 0.4072 | 0.4162 | 0.4113 | +0.00899 | +0.00407 |
| ETTh1_itr_condf | 0.4021 | 0.4276 | 0.4043 | +0.02548 | +0.00217 |
| ETTh1_patch_condc | 0.3784 | 0.3882 | 0.3793 | +0.00980 | +0.00087 |
| ETTh1_patch_condc1 | 0.3826 | 0.3913 | 0.3852 | +0.00870 | +0.00261 |
| ETTh1_patch_condf | 0.3823 | 0.4610 | 0.3839 | +0.07874 | +0.00162 |
| ETTh1_patch_parallel_ctrl | 0.3747 | 0.3747 | 0.3747 | +0.00000 | +0.00000 |
| ETTm1_itr_condc | 0.3429 | 0.3448 | 0.3517 | +0.00185 | +0.00875 |
| ETTm1_itr_condc1 | 0.3501 | 0.3563 | 0.3593 | +0.00615 | +0.00920 |
| ETTm1_itr_condf | 0.3501 | 0.4017 | 0.3880 | +0.05158 | +0.03788 |
| ETTm1_patch_condc | 0.3298 | 0.3248 | 0.3339 | -0.00498 | +0.00415 |
| ETTm1_patch_condc1 | 0.3287 | 0.3391 | 0.3365 | +0.01040 | +0.00783 |
| ETTm1_patch_condf | 0.3319 | 0.4583 | 0.3806 | +0.12644 | +0.04868 |

## Task 5: gain localization

See gain_ETTh1_patch.md / gain_ETTm1_patch.md / gain_ETTh1_itr.md / gain_ETTm1_itr.md in this directory.
Headline: PatchSTDesc ETTh1 +1.21% overall (CUSUM groups +0.94~+1.84%, flat), ETTm1 +1.22% (CUSUM g4 +2.84%, drift bin4 +3.54%, concentrated);
iTransformer descw01 is negative on both datasets (ETTh1 -1.50%, ETTm1 -1.32%), worst in structure-rich groups.
