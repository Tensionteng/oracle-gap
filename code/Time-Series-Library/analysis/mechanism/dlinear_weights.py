#!/usr/bin/env python
"""Mechanism experiment 2: DLinear weight decay curves ("linear auto-gating" hypothesis).

For each (dataset, seq_len, seed) LBv2 arm, load Linear_Seasonal / Linear_Trend
weights [pred_len=96, seq_len] and compute the |w| mass profile over lag distance
(d=1 -> most recent input step, d=seq_len -> oldest; input index 0 is the earliest lag).

Outputs:
  logs/mechanism/exp2/dlinear_decay_{dataset}.npz  (curves)
  logs/mechanism/figs/exp2_dlinear_decay_{dataset}.png
  logs/mechanism/exp2/summary.md
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIGS = os.path.join(ROOT, 'logs', 'mechanism', 'figs')
OUT = os.path.join(ROOT, 'logs', 'mechanism', 'exp2')
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SEQ_LENS = [96, 336, 720, 1440, 2880]
SEEDS = [2021, 2022, 2023]
DATASETS = ['exchange_rate', 'electricity']


def load_weight(dataset, sl, seed, which):
    pat = os.path.join(ROOT, 'checkpoints',
                       f'long_term_forecast_LBv2_{dataset}_DLinear_{sl}_s{seed}_*/checkpoint.pth')
    hits = sorted(glob.glob(pat))
    if not hits:
        return None
    sd = torch.load(hits[0], map_location='cpu')
    return sd[f'{which}.weight'].numpy()  # [pred_len, seq_len]


def mass_curve(W):
    """W: [pred, seq]. Returns mass per position j (0=oldest), averaged over output rows."""
    return np.abs(W).mean(axis=0)  # [seq]


def main():
    lines = ['# Exp2: DLinear weight-mass decay\n']
    for ds in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        results = {}
        for which, ax in zip(['Linear_Seasonal', 'Linear_Trend'], axes):
            for sl in SEQ_LENS:
                curves = []
                for seed in SEEDS:
                    W = load_weight(ds, sl, seed, which)
                    if W is not None:
                        curves.append(mass_curve(W))
                if not curves:
                    print(f'[warn] no checkpoint for {ds} sl={sl} {which}')
                    continue
                m = np.mean(np.stack(curves), axis=0)  # [sl], index 0 = oldest
                m_norm = m / m.sum()
                results[(which, sl)] = m_norm
                d = np.arange(sl, 0, -1)  # lag distance: sl .. 1 (1 = most recent)
                ax.plot(d, m_norm, label=f'sl={sl}', lw=1.2)
                # mass within the most recent 96 lags
                k = min(96, sl)
                frac96 = m[-k:].sum() / m.sum()
                # mass beyond lag 720
                if sl > 720:
                    frac_far = m[:sl - 720].sum() / m.sum()
                else:
                    frac_far = 0.0
                # daily-harmonic contrast: |w| mass at d = 24k (+-1) vs off-phase d = 24k+12 (+-1)
                d_all = np.arange(sl, 0, -1)  # d_all[j] = lag of position j
                on = np.zeros(sl, bool)
                off = np.zeros(sl, bool)
                for kk in range(1, sl // 24 + 1):
                    on |= (np.abs(d_all - 24 * kk) <= 1)
                    off |= (np.abs(d_all - (24 * kk + 12)) <= 1)
                contrast = m[on].mean() / (m[off].mean() + 1e-12)
                lines.append(f'- {ds} {which} sl={sl}: mass(last 96 lags)={frac96:.3f}, '
                             f'mass(lag>720)={frac_far:.3f}, daily-harmonic contrast={contrast:.2f} '
                             f'(seeds n={len(curves)})')
            ax.set_xscale('log')
            ax.invert_xaxis()
            ax.set_xlabel('lag distance d (1 = most recent)')
            ax.set_ylabel('normalized |w| mass')
            ax.set_title(f'{ds} {which}')
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f'exp2_dlinear_decay_{ds}.png'), dpi=150)
        plt.close(fig)
        np.savez(os.path.join(OUT, f'dlinear_decay_{ds}.npz'),
                 **{f'{which}_sl{sl}': v for (which, sl), v in results.items()})
    with open(os.path.join(OUT, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
