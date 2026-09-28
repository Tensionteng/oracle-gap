# Not All Smoothness Is Fixable — Code & Artifacts

**Diagnosing and Conditionally Repairing Over-Smoothing in Time Series Forecasting**

Point forecasts trained with MSE are systematically smoother than the series they predict. This repository shows that such over-smoothing has two components: **rational shrinkage** (the optimal response to uncertainty, which no method should remove) and **unexploited structure** (which is repairable). We provide:

- **The oracle gap** — a training-free diagnostic that measures how much repairable structure a dataset-model pair leaves behind;
- **The law** — across 15 distinct intervention cells (nine datasets, six backbones), the sign of the oracle gap predicts the direction of the repair's effect in 14 cells (the single exception is diagnosable in advance from prediction geometry), corroborated by a 665-cell audit of thirteen foundation models and a trained PatchTST on 50 GIFT-Eval variants. A cross-intervention check with FreDF sharpens the law: the negative direction transfers across intervention families (where no repairable structure remains, dispersion-targeting objectives hurt regardless of mechanism), while the positive direction is mechanism-specific (a positive gap certifies headroom; bounded counter-bias repairs exploit it);
- **RegionFocal** — a region-weighted loss that repairs over-smoothing where the gap is positive (Electricity: −2.3% MSE across 3 seeds and 4 horizons) and is withheld where the gap is non-positive, avoiding the harm it would otherwise cause;
- **Descriptor-driven selective prediction** — a light auxiliary head whose predicted change-point probability ranks forecast risk better than oracle change-point labels and static baselines (up to −5.3% remaining MSE when abstaining on the top 20%; −9.6% on a fine-tuned Chronos-Bolt). The deploy-where-risk-is-forward-looking rule holds across backbone families (PatchTST and fine-tuned Chronos-Bolt, Table `tab:decision`).

## Repository structure

```
code/
  Time-Series-Library/ experiment workbench (TSLib fork) with:
                         utils/descriptor_labels.py, utils/region_focal.py,
                         utils/compute_descriptor_stats.py,
                         models/{PatchSTDesc,iTransformerDesc,DLinearDesc,DescriptorHead,...}.py,
                         exp/exp_descriptor_forecast.py,
                         analysis/{oracle_diagnostic,behavior_metrics,probe_descriptor,
                                   gain_localization,error_correlation,...}.py
  tsfm_stage2/         Chronos-Bolt fine-tuning harness (finetune_chronos_bolt.py,
                         errcorr_bolt.py: selective-prediction analysis incl. volatility channel)
  gapbench/            GIFT-Eval adapter + TSFM zero-shot dump + oracle-gap scanner
experiments/
  queue/               all experiment job scripts (~230 runs, bash, single-GPU each),
                         incl. f0_fredf_* (cross-intervention check) and x*_tsfm_* (bolt
                         decision-side transfer)
  staged_x/            pre-registered predictions and outcome of the x-series
  tools/               scheduler.py (GPU queue scheduler), gpu_monitor.sh
  results.csv          master results table of all task-model runs
results/
  matrix.csv           the 665-cell oracle-gap audit matrix
  protocol_windows/    oracle gap under sliding vs tail window protocols
  analysis/            probe / behavior / gain-localization / error-correlation JSONs,
                         incl. errcorr_bolt_* (fine-tuned Chronos-Bolt selective prediction)
```

The LaTeX source of the paper lives in the companion paper-archive repository.

## Reproduce

1. **Environment**: Python ≥3.10 with [uv](https://github.com/astral-sh/uv). In `code/Time-Series-Library/`: `uv sync`. The tsfm_stage2 and gapbench arms use their own uv environments (see their READMEs).
2. **Data**: forecasting benchmarks (ETT, Electricity, Traffic, Weather, SWaT, etc.) via the TSLib dataset release; GIFT-Eval datasets via Hugging Face `Salesforce/GiftEval`. Set `MTP4TS_ROOT` (repo root) and `GIFT_ROOT` (GIFT-Eval cache) environment variables.
3. **Oracle gap on your own model**: produce `pred.npy`/`true.npy` ([N, horizon, C]) for a test split, then run `analysis/oracle_diagnostic.py` (see its docstring).
4. **RegionFocal**: see `code/Time-Series-Library/utils/region_focal.py`; training entry `run.py --task_loss regionfocal --rf_au 1.0`.
5. **Figures**: `paper/figures/make_figs.py` (in the companion paper repo) regenerates both figures from `results/`.

## Key experiment artifacts

- `results/matrix.csv` — 665-cell oracle-gap matrix (13 foundation models × 50 GIFT-Eval variants + a trained PatchTST), with bootstrap confidence intervals.
- `results/protocol_windows/` — oracle gap under sliding vs tail window protocols (incl. ETTh1/ETTm1 sliding, the protocol FreDF was published in).
- `results/analysis/errcorr_bolt_{ETTh1,ETTm1,ECL,traffic}.json` — selective-prediction probe on fine-tuned Chronos-Bolt (risk-coverage by descriptor change-point / volatility / context-volatility / oracle rankings).
- `experiments/results.csv` — 200+ task-model runs: baselines, descriptor-head variants, RegionFocal dose-response, backbones × datasets, and the FreDF cross-intervention check (`f0_fredf_*`, `32_fredf_*` jobs).
- `experiments/staged_x/README.md` — pre-registered predictions and outcome of the decision-side TSFM transfer runs (`x0`–`x3`).

## Notes

- We release the audit matrix and diagnostic code so the oracle gap can become a standard column in forecasting evaluations.
- Model checkpoints and datasets are not included (size); all numbers in `results/` were produced by the scripts in this repository.
