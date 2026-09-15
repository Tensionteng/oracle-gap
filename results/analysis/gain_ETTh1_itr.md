# Gain localization: ETTh1_itr

base: `results/long_term_forecast_ETTh1_96_96_iTransformer_ETTh1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0`  
desc: `pred_dumps/long_term_forecast_ETTh1_96_96_descw01_iTransformerDesc_ETTh1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_descw01_0`


## by CUSUM score group (train-quantile edges)

| cusum_score group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 578 | 0.37903 | 0.38383 | -1.26% |
| 1 | 639 | 0.37840 | 0.38386 | -1.44% |
| 2 | 685 | 0.40099 | 0.40720 | -1.55% |
| 3 | 528 | 0.39643 | 0.40285 | -1.62% |
| 4 | 355 | 0.43698 | 0.44428 | -1.67% |
| all | 2785 | 0.39497 | 0.40090 | -1.50% |

## by drift bin (train-quantile edges)

| drift_bin group | n | mse_base | mse_desc | rel_gain |
|---|---|---|---|---|
| 0 | 222 | 0.39832 | 0.40433 | -1.51% |
| 1 | 545 | 0.39830 | 0.40316 | -1.22% |
| 2 | 1210 | 0.39284 | 0.39907 | -1.59% |
| 3 | 579 | 0.41259 | 0.42016 | -1.84% |
| 4 | 229 | 0.35056 | 0.35310 | -0.73% |
| all | 2785 | 0.39497 | 0.40090 | -1.50% |
