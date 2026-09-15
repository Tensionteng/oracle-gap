#!/usr/bin/env python
"""Collect mechanism experiment 3 results (PatchTSTGated repair ablation).

Parses logs/mechanism/exp3/*.log for final test metrics, joins with the ehs_v2
main-matrix PatchTST reference (capped windows, original einsum attention),
dumps logs/mechanism/exp3/summary.md and exp3_results.npz.

Also reports the learned denbias null-logit b per layer/head and the resulting
null-mass (fraction of softmax weight on the "attend-to-nothing" column),
measured on the first test batches.
"""
import argparse
import glob
import os
import re

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGDIR = os.path.join(ROOT, 'logs', 'mechanism', 'exp3')
OUT = os.path.join(ROOT, 'logs', 'mechanism', 'exp3')
FIGS = os.path.join(ROOT, 'logs', 'mechanism', 'figs')

SEQ_LENS = [96, 1440, 2880]
VARIANTS = ['none', 'denbias', 'recmask']
SEEDS = [2021, 2022]
DATASETS = ['exchange_rate', 'electricity']

# ehs_v2 main-matrix PatchTST test MSE (mean over seeds), from logs/ehs_v2/SUMMARY.md
EHS_V2 = {
    'exchange_rate': {96: (0.0871, 0.0020), 336: (0.1048, 0.0125), 720: (0.0924, 0.0030),
                      1440: (0.1484, 0.0239), 2880: (0.4053, 0.0169)},
    'electricity': {96: (0.1816, 0.0001), 336: (0.1380, 0.0000)},
}


def parse_log(path):
    try:
        with open(path, 'rb') as f:
            f.seek(max(0, os.path.getsize(path) - 4000))
            tail = f.read().decode(errors='ignore')
    except OSError:
        return None
    m = re.findall(r'mse:([\d.]+), mae:([\d.]+)', tail)
    if not m:
        return None
    return float(m[-1][0]), float(m[-1][1])


def collect():
    rows = {}
    for ds in DATASETS:
        for sl in SEQ_LENS:
            for variant in VARIANTS:
                vals = []
                for seed in SEEDS:
                    r = parse_log(os.path.join(LOGDIR, f'{ds}_{variant}_sl{sl}_s{seed}.log'))
                    if r is not None:
                        vals.append(r)
                rows[(ds, sl, variant)] = vals
    return rows


@torch.no_grad()
def null_mass(ds, sl, seed):
    """Learned null-logit and its softmax mass for a denbias arm."""
    import sys
    sys.path.insert(0, ROOT)
    from models import PatchTSTGated
    from data_provider.data_factory import data_provider
    pat = os.path.join(ROOT, 'checkpoints',
                       f'long_term_forecast_MECH_{ds}_PTG_denbias_{sl}_s{seed}_*/checkpoint.pth')
    hits = sorted(glob.glob(pat))
    if not hits:
        return None
    enc = 8 if ds == 'exchange_rate' else 321
    root = f'./dataset/{ds}/'
    cfg = argparse.Namespace(
        task_name='long_term_forecast', seq_len=sl, pred_len=96,
        d_model=512, n_heads=8, e_layers=2, d_ff=2048, dropout=0.1,
        activation='gelu', enc_in=enc, factor=3,
        attn_variant='denbias', recmask_window=336)
    model = PatchTSTGated.Model(cfg).float().cuda().eval()
    model.load_state_dict(torch.load(hits[0], map_location='cuda'))
    for layer in model.encoder.attn_layers:
        layer.attention.inner_attention.output_attention = True
    args = argparse.Namespace(
        task_name='long_term_forecast', is_training=0, model_id='nm', model='PatchTSTGated',
        data='custom', root_path=root, data_path=f'{ds}.csv', features='M', target='OT',
        freq='h', checkpoints='./checkpoints/', seq_len=sl, label_len=48, pred_len=96,
        seasonal_patterns='Monthly', inverse=False, embed='timeF',
        batch_size=2 if enc > 8 else 8, num_workers=0)
    _, loader = data_provider(args, 'test')
    bvals, nmass = [], [[] for _ in range(cfg.e_layers)]
    for layer in model.encoder.attn_layers:
        bvals.append(layer.attention.inner_attention.null_logit.detach().cpu().numpy())
    # proper forward to get per-layer inputs
    for i, (bx, by, bxm, bym) in enumerate(loader):
        if i >= 4:
            break
        bx = bx.float().cuda()
        means = bx.mean(1, keepdim=True).detach()
        x = (bx - means) / torch.sqrt(torch.var(bx, dim=1, keepdim=True, unbiased=False) + 1e-5)
        enc_out, _ = model.patch_embedding(x.permute(0, 2, 1))
        for l, layer in enumerate(model.encoder.attn_layers):
            ia = layer.attention.inner_attention
            B, L, _ = enc_out.shape
            q = layer.attention.query_projection(enc_out).view(B, L, cfg.n_heads, -1).transpose(1, 2)
            k = layer.attention.key_projection(enc_out).view(B, L, cfg.n_heads, -1).transpose(1, 2)
            scores = q @ k.transpose(-2, -1) / np.sqrt(q.shape[-1])
            nb = ia.null_logit.view(1, cfg.n_heads, 1, 1).expand(B, cfg.n_heads, L, 1)
            scores = torch.cat([scores, nb], dim=-1)
            A = torch.softmax(scores, dim=-1)
            nmass[l].append(A[..., -1].mean().item())
            enc_out, _ = layer(enc_out)
    return dict(b=bvals, null_mass=[float(np.mean(v)) for v in nmass])


def main():
    rows = collect()
    lines = ['# Exp3: PatchTSTGated repair ablation (uncapped windows; pred_len=96)', '',
             'MSE mean±std over seeds {2021,2022}. `none` = PatchTSTGated vanilla (SDPA).', '']
    for ds in DATASETS:
        lines.append(f'## {ds}')
        lines.append('')
        lines.append('| seq_len | none (ctrl) | denbias | recmask | ehs_v2 PatchTST (capped) |')
        lines.append('|---|---|---|---|---|')
        for sl in SEQ_LENS:
            cells = []
            for variant in VARIANTS:
                vals = rows[(ds, sl, variant)]
                if vals:
                    mses = [v[0] for v in vals]
                    m, s = np.mean(mses), np.std(mses)
                    cells.append(f'{m:.4f}±{s:.4f}[n={len(vals)}]')
                else:
                    cells.append('-')
            ref = EHS_V2.get(ds, {}).get(sl)
            cells.append(f'{ref[0]:.4f}±{ref[1]:.4f}' if ref else '-')
            lines.append(f'| {sl} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    np.savez(os.path.join(OUT, 'exp3_results.npz'),
             **{f'{ds}_{sl}_{v}': np.array([x[0] for x in rows[(ds, sl, v)]])
                for ds in DATASETS for sl in SEQ_LENS for v in VARIANTS})
    with open(os.path.join(OUT, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # figure: MSE vs seq_len per variant, one panel per dataset
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    markers = {'none': 'o-', 'denbias': 's--', 'recmask': '^-.'}
    labels = {'none': 'vanilla (ctrl)', 'denbias': 'denbias', 'recmask': 'recmask'}
    for ax, ds in zip(axes, DATASETS):
        for variant in VARIANTS:
            xs, ys, es = [], [], []
            for sl in SEQ_LENS:
                vals = [v[0] for v in rows[(ds, sl, variant)]]
                if vals:
                    xs.append(sl)
                    ys.append(np.mean(vals))
                    es.append(np.std(vals))
            if xs:
                ax.errorbar(xs, ys, yerr=es, fmt=markers[variant], label=labels[variant],
                            capsize=3, lw=1.6)
        ref = EHS_V2.get(ds, {})
        xs = [sl for sl in SEQ_LENS if sl in ref]
        if xs:
            ax.errorbar(xs, [ref[sl][0] for sl in xs], yerr=[ref[sl][1] for sl in xs],
                        fmt='x:', color='gray', label='ehs_v2 PatchTST (capped)', capsize=3)
        ax.set_xscale('log')
        ax.set_xticks(SEQ_LENS)
        ax.set_xticklabels([str(s) for s in SEQ_LENS])
        ax.set_xlabel('seq_len')
        ax.set_ylabel('test MSE')
        ax.set_title(ds)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8)
    fig.suptitle('Exp3: adding selectivity to PatchTST attention (uncapped windows, seeds 2021/2022)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'exp3_repair_ablation.png'), dpi=150)
    plt.close(fig)

    # denbias null-logit diagnostics (only if those runs finished)
    dl = ['\n## denbias learned null-logit b and null mass\n']
    for ds in DATASETS:
        for sl in SEQ_LENS:
            for seed in SEEDS:
                try:
                    r = null_mass(ds, sl, seed)
                except Exception as e:
                    r = None
                    print(f'[warn] null_mass {ds} sl{sl} s{seed}: {e}')
                if r is None:
                    continue
                bs = ', '.join(f'L{l}: mean b={b.mean():.2f}' for l, b in enumerate(r['b']))
                nm = ', '.join(f'L{l}: {m:.3f}' for l, m in enumerate(r['null_mass']))
                dl.append(f'- {ds} sl={sl} s{seed}: {bs}; null mass: {nm}')
                print(dl[-1])
    with open(os.path.join(OUT, 'summary.md'), 'a') as f:
        f.write('\n'.join(dl) + '\n')


if __name__ == '__main__':
    main()
