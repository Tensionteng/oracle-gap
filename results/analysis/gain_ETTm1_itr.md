# Gain localization: ETTm1_itr

base: `results/long_term_forecast_ETTm1_96_96_iTransformer_ETTm1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0`  
desc: `pred_dumps/long_term_forecast_ETTm1_96_96_descw01_iTransformerDesc_ETTm1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_descw01_0`


## by CUSUM score group (train-quantile edges)

| cusum_score group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 2335 | 0.29697 | 0.29946 | -0.84% |
| 1 | 2418 | 0.32944 | 0.33252 | -0.93% |
| 2 | 2217 | 0.32803 | 0.33337 | -1.63% |
| 3 | 2331 | 0.34652 | 0.35269 | -1.78% |
| 4 | 2124 | 0.41158 | 0.41717 | -1.36% |
| all | 11425 | 0.34128 | 0.34578 | -1.32% |

## by drift bin (train-quantile edges)

| drift_bin group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 1579 | 0.44094 | 0.45858 | -4.00% |
| 1 | 2395 | 0.31586 | 0.31725 | -0.44% |
| 2 | 3573 | 0.30159 | 0.30459 | -0.99% |
| 3 | 2115 | 0.32612 | 0.32821 | -0.64% |
| 4 | 1763 | 0.38519 | 0.38805 | -0.74% |
| all | 11425 | 0.34128 | 0.34578 | -1.32% |
