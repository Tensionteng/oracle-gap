# Gain localization: smoke_patchtst

base: `results/long_term_forecast_smoke_base_patchtst_PatchTST_ETTh1_ftM_sl96_ll48_pl96_dm16_nh2_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_smoke_0`  
desc: `pred_dumps/long_term_forecast_smoke_desc_patchtst_PatchSTDesc_ETTh1_ftM_sl96_ll48_pl96_dm16_nh2_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_smoke_0`


## by CUSUM score group (train-quantile edges)

| cusum_score group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 578 | 0.87048 | 0.87024 | +0.03% |
| 1 | 639 | 0.89640 | 0.89620 | +0.02% |
| 2 | 685 | 0.93480 | 0.93465 | +0.02% |
| 3 | 528 | 0.90684 | 0.90663 | +0.02% |
| 4 | 355 | 0.83576 | 0.83583 | -0.01% |
| all | 2785 | 0.89471 | 0.89455 | +0.02% |

## by drift bin (train-quantile edges)

| drift_bin group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 222 | 0.99080 | 0.99048 | +0.03% |
| 1 | 545 | 0.91950 | 0.91926 | +0.03% |
| 2 | 1210 | 0.88693 | 0.88676 | +0.02% |
| 3 | 579 | 0.87556 | 0.87544 | +0.01% |
| 4 | 229 | 0.83213 | 0.83228 | -0.02% |
| all | 2785 | 0.89471 | 0.89455 | +0.02% |
