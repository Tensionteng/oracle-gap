# Gain localization: ETTm1_patch

base: `results/long_term_forecast_ETTm1_96_96_PatchTST_ETTm1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0`  
desc: `pred_dumps/long_term_forecast_ETTm1_96_96_descw01_PatchSTDesc_ETTm1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_descw01_0`


## by CUSUM score group (train-quantile edges)

| cusum_score group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 2335 | 0.29053 | 0.28626 | +1.47% |
| 1 | 2418 | 0.31699 | 0.31422 | +0.87% |
| 2 | 2217 | 0.30926 | 0.30673 | +0.82% |
| 3 | 2331 | 0.32510 | 0.32548 | -0.11% |
| 4 | 2124 | 0.39781 | 0.38653 | +2.84% |
| all | 11425 | 0.32676 | 0.32279 | +1.22% |

## by drift bin (train-quantile edges)

| drift_bin group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 1579 | 0.41637 | 0.42562 | -2.22% |
| 1 | 2395 | 0.29152 | 0.28949 | +0.70% |
| 2 | 3573 | 0.28733 | 0.28094 | +2.22% |
| 3 | 2115 | 0.31000 | 0.30635 | +1.18% |
| 4 | 1763 | 0.39442 | 0.38048 | +3.54% |
| all | 11425 | 0.32676 | 0.32279 | +1.22% |
