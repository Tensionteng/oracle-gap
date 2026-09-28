"""Generate paper figures: fig_map (665-cell oracle-gap heatmap) and fig_riskcov (risk-coverage curves).
Run: uv run python ../paper/figures/make_figs.py  (from Time-Series-Library venv)
Outputs: ../paper/figures/fig_map.pdf, fig_riskcov.pdf (+ .png previews)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import os

ROOT = os.environ.get('MTP4TS_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'paper', 'figures')

def _matrix_path():
    for cand in (os.path.join(ROOT, 'results', 'matrix.csv'),
                 os.path.join(ROOT, 'gapbench', 'results', 'matrix.csv')):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError('matrix.csv not found under results/ or gapbench/results/')

# ---------- fig_map ----------
rows = list(csv.DictReader(open(_matrix_path())))
models_order = ['bolt_tiny', 'bolt_mini', 'bolt_small', 'bolt_base', 'chronos2',
                'sundial', 'timemoe_50m', 'timemoe_200m', 'timer_84m', 'tirex',
                'moirai11_small', 'moirai11_large', 'moirai2_small', 'patchtst']
pretty_model = {'bolt_tiny': 'Chronos-Bolt 8M', 'bolt_mini': 'Chronos-Bolt 21M',
                'bolt_small': 'Chronos-Bolt 48M', 'bolt_base': 'Chronos-Bolt 205M',
                'chronos2': 'Chronos-2', 'sundial': 'Sundial', 'timemoe_50m': 'TimeMoE 50M',
                'timemoe_200m': 'TimeMoE 200M', 'timer_84m': 'Timer 84M', 'tirex': 'TiRex',
                'moirai11_small': 'Moirai-1.1 14M', 'moirai11_large': 'Moirai-1.1 311M',
                'moirai2_small': 'Moirai-2.0', 'patchtst': 'PatchTST (trained)'}
gap = {(r['model'], r['dataset']): float(r['gap']) for r in rows if r.get('status', 'ok') == 'ok' and r['gap']}
datasets = sorted({d for (m, d) in gap}, key=lambda d: np.nanmean([v for (m, dd), v in gap.items() if dd == d]))
M = np.array([[gap.get((m, d), np.nan) for d in datasets] for m in models_order], dtype=float)

fig, ax = plt.subplots(figsize=(9.2, 3.4))
norm = TwoSlopeNorm(vmin=-0.08, vcenter=0.0, vmax=0.08)
im = ax.imshow(np.clip(M, -0.08, 0.08), cmap='PuOr_r', norm=norm, aspect='auto', interpolation='nearest')
ax.set_yticks(range(len(models_order)))
ax.set_yticklabels([pretty_model[m] for m in models_order], fontsize=7)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels([d.replace('_', ' ') for d in datasets], rotation=90, fontsize=4.5)
ax.set_xlabel('GIFT-Eval dataset variant (sorted by mean gap)', fontsize=8)
cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03, extend='both')
cbar.set_label('Oracle gap (clipped at ±0.08)', fontsize=7)
cbar.ax.tick_params(labelsize=6)
ax.set_title('Repairable structure by dataset and model', fontsize=9)
plt.tight_layout()
fig.savefig(f'{OUT}/fig_map.pdf')
fig.savefig(f'{OUT}/fig_map.png', dpi=200)
plt.close(fig)

# ---------- fig_riskcov ----------
signals = [('pred_cp', 'Predicted change-point prob. (ours)', '#0072B2', '-'),
           ('oracle_cp', 'Oracle change-point ranking', '#666666', '--'),
           ('ctx_vol', 'Context volatility (static)', '#D55E00', '-.'),
           ('random', 'Random ranking', '#999999', ':'),
           ('oracle_vol', 'Oracle volatility (upper bound)', '#000000', '-')]
def _resolve(*cands):
    for cand in cands:
        p = os.path.join(ROOT, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(cands[0])

panels = [('ETTh1', _resolve('Time-Series-Library/analysis/results/errcorr_ETTh1_patch_descw01_curves.npz',
                            'results/analysis/errcorr_ETTh1_patch_descw01_curves.npz')),
          ('ETTm1', _resolve('Time-Series-Library/analysis/results/errcorr_ETTm1_patch_descw01_curves.npz',
                            'results/analysis/errcorr_ETTm1_patch_descw01_curves.npz'))]
bolt_json = [('Chronos-Bolt ft ETTh1', _resolve('tsfm_stage2/results/errcorr_bolt_ETTh1.json',
                                                'results/analysis/errcorr_bolt_ETTh1.json')),
             ('Chronos-Bolt ft ETTm1', _resolve('tsfm_stage2/results/errcorr_bolt_ETTm1.json',
                                                'results/analysis/errcorr_bolt_ETTm1.json'))]

import json as _json
fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.2), sharey=True)
for ax, (name, path) in zip(axes[:2], panels):
    z = np.load(path)
    cov = z['coverage']
    for key, label, color, ls in signals:
        k = f'curve_{key}'
        if k not in z.files:
            continue
        mse = z[k]
        ax.plot(cov, mse / mse[0], label=label, color=color, ls=ls, lw=1.6)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('Coverage', fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel('Remaining MSE (relative)', fontsize=8)
for ax, (name, path) in zip([axes[2]], bolt_json[:1] + bolt_json[1:]):
    pass
axb = axes[2]
for name, path in bolt_json:
    d = _json.load(open(path))
    cov = np.array(d['coverage_grid'], dtype=float)
    for key, lab, col in [('pred_cp', 'Predicted change-point prob. (ours)', '#0072B2'),
                          ('ctx_vol', 'Context volatility (static)', '#D55E00'),
                          ('oracle_vol', 'Oracle volatility (upper bound)', '#000000')]:
        c = np.array(d['curves'][key], dtype=float)
        axb.plot(cov, c / c[0], label=f"{lab.split(' (')[0]} ({name.split(' ')[-1]})", color=col,
                 ls='-' if 'ETTh1' in name else '--', lw=1.5)
axb.set_title('Chronos-Bolt fine-tuned', fontsize=9)
axb.set_xlabel('Coverage', fontsize=8)
axb.tick_params(labelsize=7)
axb.legend(fontsize=6, loc='upper right', frameon=False)
for ax in axes: ax.invert_xaxis()
axes[1].legend(fontsize=6.5, loc='upper right', frameon=False)
plt.tight_layout()
fig.savefig(f'{OUT}/fig_riskcov.pdf')
fig.savefig(f'{OUT}/fig_riskcov.png', dpi=200)
print('figures written')

# ---------- fig_map_domain (main-text aggregated version) ----------
DOMAIN = {
    'SZ_TAXI': 'Transit', 'LOOP_SEATTLE': 'Transit', 'bizitobs_application': 'Cloud/IoT',
    'bizitobs_l2c': 'Cloud/IoT', 'bizitobs_service': 'Cloud/IoT', 'bitbrains_rnd': 'Cloud/IoT',
    'bitbrains_fast_storage': 'Cloud/IoT', 'M_DENSE': 'Cloud/IoT', 'kdd_cup_2018_with_missing': 'Cloud/IoT',
    'solar': 'Energy', 'electricity': 'Energy',
    'jena_weather': 'Weather', 'temperature_rain_with_missing': 'Weather', 'saugeenday': 'Weather',
    'hierarchical_sales': 'Sales', 'restaurant': 'Sales', 'car_parts_with_missing': 'Sales',
    'covid_deaths': 'Health', 'hospital': 'Health', 'us_births': 'Health',
    'm4_hourly': 'M4', 'm4_daily': 'M4', 'm4_weekly': 'M4', 'm4_monthly': 'M4',
    'm4_quarterly': 'M4', 'm4_yearly': 'M4',
    'ett1': 'ETT', 'ett2': 'ETT',
}
doms = ['Transit', 'Energy', 'Cloud/IoT', 'Weather', 'Sales', 'Health', 'M4', 'ETT']
Md = np.full((len(models_order), len(doms)), np.nan)
for i, m in enumerate(models_order):
    for j, dm in enumerate(doms):
        vals = [v for (mm, d), v in gap.items() if mm == m and DOMAIN.get(d.split('/')[0]) == dm]
        if vals: Md[i, j] = np.nanmean(vals)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
im = ax.imshow(Md, cmap='PuOr_r', norm=TwoSlopeNorm(vmin=-0.08, vcenter=0.0, vmax=0.08), aspect='auto')
ax.set_yticks(range(len(models_order))); ax.set_yticklabels([pretty_model[m] for m in models_order], fontsize=7)
ax.set_xticks(range(len(doms))); ax.set_xticklabels(doms, fontsize=8, rotation=20, ha='right')
for i in range(len(models_order)):
    for j in range(len(doms)):
        if np.isfinite(Md[i, j]):
            ax.text(j, i, f'{Md[i,j]:+.2f}', ha='center', va='center', fontsize=5.2,
                    color='white' if abs(Md[i, j]) > 0.05 else '#333333')
cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04, extend='both')
cbar.set_label('Mean oracle gap by domain', fontsize=7); cbar.ax.tick_params(labelsize=6)
plt.tight_layout()
fig.savefig(f'{OUT}/fig_map_domain.pdf')
fig.savefig(f'{OUT}/fig_map_domain.png', dpi=200)
print('domain map written')
