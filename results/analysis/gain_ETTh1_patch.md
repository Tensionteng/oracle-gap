# Gain localization: ETTh1_patch

base: `results/long_term_forecast_ETTh1_96_96_PatchTST_ETTh1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0`  
desc: `pred_dumps/long_term_forecast_ETTh1_96_96_descw01_PatchSTDesc_ETTh1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_descw01_0`


## by CUSUM score group (train-quantile edges)

| cusum_score group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 578 | 0.36287 | 0.35877 | +1.13% |
| 1 | 639 | 0.36619 | 0.35945 | +1.84% |
| 2 | 685 | 0.38874 | 0.38493 | +0.98% |
| 3 | 528 | 0.37999 | 0.37603 | +1.04% |
| 4 | 355 | 0.41029 | 0.40645 | +0.94% |
| all | 2785 | 0.37928 | 0.37471 | +1.21% |

## by drift bin (train-quantile edges)

| drift_bin group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 222 | 0.38866 | 0.38029 | +2.15% |
| 1 | 545 | 0.39206 | 0.38700 | +1.29% |
| 2 | 1210 | 0.37402 | 0.37119 | +0.76% |
| 3 | 579 | 0.39353 | 0.38651 | +1.78% |
| 4 | 229 | 0.33160 | 0.32882 | +0.84% |
| all | 2785 | 0.37928 | 0.37471 | +1.21% |
