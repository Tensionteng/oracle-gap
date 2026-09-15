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

OUT = '" + os.environ.get("MTP4TS_ROOT", ".") + "/paper/figures'

# ---------- fig_map ----------
rows = list(csv.DictReader.open if 0 else csv.DictReader(open('" + os.environ.get("MTP4TS_ROOT", ".") + "/gapbench/results/matrix.csv')))
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
panels = [('ETTh1', '" + os.environ.get("MTP4TS_ROOT", ".") + "/Time-Series-Library/analysis/results/errcorr_ETTh1_patch_descw01_curves.npz'),
          ('ETTm1', '" + os.environ.get("MTP4TS_ROOT", ".") + "/Time-Series-Library/analysis/results/errcorr_ETTm1_patch_descw01_curves.npz')]

fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
for ax, (name, path) in zip(axes, panels):
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
axes[0].invert_xaxis()
axes[1].invert_xaxis()
axes[1].legend(fontsize=6.5, loc='upper right', frameon=False)
plt.tight_layout()
fig.savefig(f'{OUT}/fig_riskcov.pdf')
fig.savefig(f'{OUT}/fig_riskcov.png', dpi=200)
print('figures written')
