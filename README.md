# Not All Smoothness Is Fixable

**Diagnosing and Conditionally Repairing Over-Smoothing in Time Series Forecasting**

Point forecasts trained with MSE are systematically smoother than the series they predict. This repository shows that such over-smoothing has two components: **rational shrinkage** (the optimal response to uncertainty, which no method should remove) and **unexploited structure** (which is repairable). We provide:

- **The oracle gap** — a training-free diagnostic that measures how much repairable structure a dataset-model pair leaves behind;
- **The law** — across 678 dataset-model cells (13 task-model cells + a 665-cell audit of 13 foundation models on 50 GIFT-Eval variants), the sign of the oracle gap predicts the sign of every anti-smoothing intervention we tested, without contradiction;
- **RegionFocal** — a region-weighted loss that repairs over-smoothing where the gap is positive (Electricity: −2.3% MSE across 3 seeds and 4 horizons) and degrades gracefully to MSE where it is not;
- **Descriptor-driven selective prediction** — a light auxiliary head whose predicted change-point probability ranks forecast risk better than oracle change-point labels and static baselines (up to −5.3% remaining MSE when abstaining on the top 20%).

## Repository structure

```
paper/                 LaTeX source of the paper (ICLR 2027 style)
  main.tex             entry point
  sections/            per-section sources
  figures/             figure PDFs + make_figs.py (regenerates from results/)
  refs.bib             bibliography (all entries verified against DBLP/arXiv)
code/
  Time-Series-Library/ experiment workbench (TSLib fork) with:
                         utils/descriptor_labels.py, utils/region_focal.py,
                         utils/compute_descriptor_stats.py,
                         models/{PatchSTDesc,iTransformerDesc,DLinearDesc,DescriptorHead,...}.py,
                         exp/exp_descriptor_forecast.py,
                         analysis/{oracle_diagnostic,behavior_metrics,probe_descriptor,
                                   gain_localization,error_correlation,...}.py
  tsfm_stage2/         Chronos-Bolt fine-tuning harness (finetune_chronos_bolt.py)
  gapbench/            GIFT-Eval adapter + TSFM zero-shot dump + oracle-gap scanner
experiments/
  queue/               all experiment job scripts (~230 runs, bash, single-GPU each)
  tools/               scheduler.py (GPU queue scheduler), gpu_monitor.sh
  results.csv          master results table of all task-model runs
results/
  matrix.csv           the 665-cell oracle-gap audit matrix
  analysis/            probe / behavior / gain-localization / error-correlation JSONs
```

## Reproduce

1. **Environment**: Python ≥3.10 with [uv](https://github.com/astral-sh/uv). In `code/Time-Series-Library/`: `uv sync`. The tsfm_stage2 and gapbench arms use their own uv environments (see their READMEs).
2. **Data**: forecasting benchmarks (ETT, Electricity, Traffic, Weather, SWaT, etc.) via the TSLib dataset release; GIFT-Eval datasets via Hugging Face `Salesforce/GiftEval`. Set `MTP4TS_ROOT` (repo root) and `GIFT_ROOT` (GIFT-Eval cache) environment variables.
3. **Oracle gap on your own model**: produce `pred.npy`/`true.npy` ([N, horizon, C]) for a test split, then run `analysis/oracle_diagnostic.py` (see its docstring).
4. **RegionFocal**: see `code/Time-Series-Library/utils/region_focal.py`; training entry `run.py --task_loss regionfocal --rf_au 1.0`.
5. **Figures**: `paper/figures/make_figs.py` regenerates both figures from `results/`.

## Key experiment artifacts

- `results/matrix.csv` — 665-cell oracle-gap matrix (13 foundation models × 50 GIFT-Eval variants), with bootstrap confidence intervals.
- `experiments/results.csv` — 200+ task-model runs: baselines, descriptor-head variants, RegionFocal dose-response, backbones × datasets.
- `ccfa-review-reports/` (if present) — internal review and integrity-audit records.

## Notes

- We release the audit matrix and diagnostic code so the oracle gap can become a standard column in forecasting evaluations.
- Model checkpoints and datasets are not included (size); all numbers in `results/` and `paper/` were produced by the scripts in this repository.
