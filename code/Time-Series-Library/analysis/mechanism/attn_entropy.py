#!/usr/bin/env python
"""Mechanism experiment 1: attention entropy / sink / recency-mass analysis.

Loads trained PatchTST (LBv2 checkpoints, original einsum FullAttention) or
PatchTSTGated 'none' (MECH checkpoints) models, flips output_attention=True,
forwards the first test batches, and computes per-layer:
  - mean per-query attention entropy (nats; also normalized by log S)
  - sink mass: attention mass on the first key (oldest patch)
  - near-10% mass: attention mass on the most recent ceil(10%) patches

Inputs are patch tokens (patch_len=16, stride=8, S=(sl-16)/8+2).

Outputs:
  logs/mechanism/exp1/attn_metrics.npz
  logs/mechanism/figs/exp1_heatmap_{dataset}_sl{sl}.png   (one example per cell)
  logs/mechanism/figs/exp1_entropy_vs_lookback.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_provider.data_factory import data_provider

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIGS = os.path.join(ROOT, 'logs', 'mechanism', 'figs')
OUT = os.path.join(ROOT, 'logs', 'mechanism', 'exp1')
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

DS_CFG = {
    'exchange_rate': dict(root='./dataset/exchange_rate/', csv='exchange_rate.csv', enc=8,
                          n_batches=8, bs=8),
    'electricity': dict(root='./dataset/electricity/', csv='electricity.csv', enc=321,
                        n_batches=8, bs=2),
}

PATCH_LEN, STRIDE = 16, 8


def make_args(ds, sl, bs):
    return argparse.Namespace(
        task_name='long_term_forecast', is_training=0, model_id='attnprobe',
        model='PatchTST', data='custom', root_path=DS_CFG[ds]['root'],
        data_path=DS_CFG[ds]['csv'], features='M', target='OT', freq='h',
        checkpoints='./checkpoints/', seq_len=sl, label_len=48, pred_len=96,
        seasonal_patterns='Monthly', inverse=False, embed='timeF',
        batch_size=bs, num_workers=0)


def find_ckpt(dataset, sl, seed, source):
    if source == 'lbv2':
        pat = f'long_term_forecast_LBv2_{dataset}_PatchTST_{sl}_s{seed}_*/checkpoint.pth'
    else:
        pat = f'long_term_forecast_MECH_{dataset}_PTG_none_{sl}_s{seed}_*/checkpoint.pth'
    hits = sorted(glob.glob(os.path.join(ROOT, 'checkpoints', pat)))
    return hits[0] if hits else None


def build_model(dataset, sl, source, device):
    if source == 'lbv2':
        from models import PatchTST as M
    else:
        from models import PatchTSTGated as M
    cfg = argparse.Namespace(
        task_name='long_term_forecast', seq_len=sl, pred_len=96,
        d_model=512, n_heads=8, e_layers=2, d_ff=2048, dropout=0.1,
        activation='gelu', enc_in=DS_CFG[dataset]['enc'], factor=3,
        attn_variant='none', recmask_window=336)
    model = M.Model(cfg).float().to(device).eval()
    for layer in model.encoder.attn_layers:
        layer.attention.inner_attention.output_attention = True
    return model


@torch.no_grad()
def probe(dataset, sl, seed, source, device='cuda:0', save_heatmap=False):
    ckpt = find_ckpt(dataset, sl, seed, source)
    if ckpt is None:
        return None
    model = build_model(dataset, sl, source, device)
    sd = torch.load(ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    args = make_args(dataset, sl, DS_CFG[dataset]['bs'])
    _, loader = data_provider(args, 'test')

    n_layers = len(model.encoder.attn_layers)
    ent = [[] for _ in range(n_layers)]      # mean per-query entropy per batch
    sink = [[] for _ in range(n_layers)]
    near = [[] for _ in range(n_layers)]
    heat = None
    n_batches = DS_CFG[dataset]['n_batches']
    for i, (bx, by, bxm, bym) in enumerate(loader):
        if i >= n_batches:
            break
        bx = bx.float().to(device)
        out, attns = None, None
        # forward through forecast path manually to grab attns
        means = bx.mean(1, keepdim=True).detach()
        x = bx - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        x = x.permute(0, 2, 1)
        enc_out, _ = model.patch_embedding(x)
        _, attns = model.encoder(enc_out)
        for l, A in enumerate(attns):
            # A: [B*nvars, H, Lq, S]
            A_ = A.clamp_min(1e-12)
            H = -(A_ * A_.log()).sum(-1)  # [B*nvars, H, Lq]
            ent[l].append(H.mean().item())
            sink[l].append(A[..., 0].mean().item())
            k = max(1, int(round(0.1 * A.shape[-1])))
            near[l].append(A[..., -k:].sum(-1).mean().item())
        if i == 0 and save_heatmap:
            heat = [A[0].mean(0).float().cpu().numpy() for A in attns]  # layer -> [L,S] head-mean
        del attns
    S = int((sl - PATCH_LEN) / STRIDE + 2)
    res = dict(
        entropy=[float(np.mean(e)) for e in ent],
        entropy_norm=[float(np.mean(e)) / np.log(S) for e in ent],
        sink=[float(np.mean(s)) for s in sink],
        near10=[float(np.mean(n)) for n in near],
        S=S, ckpt=os.path.relpath(ckpt, ROOT), heat=heat)
    return res


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    plan = [
        # (dataset, sl, seeds, source, note)
        ('exchange_rate', 96, [2021, 2022], 'lbv2', ''),
        ('exchange_rate', 720, [2021, 2022], 'lbv2', ''),
        ('exchange_rate', 2880, [2021, 2022], 'lbv2', ''),
        ('electricity', 96, [2021, 2022], 'lbv2', ''),
        ('electricity', 720, [2021], 'lbv2', 'partial-4epoch ckpt (ehs_v2 OOM-cut)'),
        # electricity 1440/2880: no completed LBv2 ckpt; filled from MECH vanilla runs
        ('electricity', 96, [2021, 2022], 'mech', 'uncapped vanilla control'),
        ('electricity', 1440, [2021, 2022], 'mech', 'uncapped vanilla control'),
        ('electricity', 2880, [2021, 2022], 'mech', 'uncapped vanilla control'),
    ]
    all_res = {}
    for ds, sl, seeds, source, note in plan:
        for seed in seeds:
            key = f'{ds}_{sl}_s{seed}_{source}'
            save_hm = (seed == seeds[0])
            try:
                r = probe(ds, sl, seed, source, device, save_heatmap=save_hm)
            except AssertionError as e:
                print(f'[skip] {key}: {e}')
                continue
            if r is None:
                print(f'[skip] {key}: no checkpoint')
                continue
            r['note'] = note
            all_res[key] = r
            print(f'[ok] {key}: S={r["S"]} ent={["%.3f" % e for e in r["entropy"]]} '
                  f'sink={["%.3f" % s for s in r["sink"]]} near10={["%.3f" % n for n in r["near10"]]}',
                  flush=True)
            if save_hm and r['heat'] is not None:
                nL = len(r['heat'])
                fig, axes = plt.subplots(1, nL, figsize=(5.2 * nL, 4.4))
                if nL == 1:
                    axes = [axes]
                for l, ax in enumerate(axes):
                    im = ax.imshow(r['heat'][l], origin='lower', aspect='auto',
                                   cmap='viridis')
                    ax.set_title(f'layer {l}')
                    ax.set_xlabel('key patch (0=oldest)')
                    ax.set_ylabel('query patch')
                    fig.colorbar(im, ax=ax, fraction=0.046)
                fig.suptitle(f'{ds} sl={sl} ({source}) attention map, head-mean, 1st test window ch0')
                fig.tight_layout()
                fig.savefig(os.path.join(FIGS, f'exp1_heatmap_{ds}_sl{sl}_{source}.png'), dpi=140)
                plt.close(fig)

    np.savez(os.path.join(OUT, 'attn_metrics.npz'),
             **{k + '_entropy': v['entropy'] for k, v in all_res.items()},
             **{k + '_entropy_norm': v['entropy_norm'] for k, v in all_res.items()},
             **{k + '_sink': v['sink'] for k, v in all_res.items()},
             **{k + '_near10': v['near10'] for k, v in all_res.items()},
             **{k + '_meta': [v['S'], v['ckpt'], v['note']] for k, v in all_res.items()})

    # comparison figure: entropy / near10 / sink vs lookback, per dataset & source
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    import collections
    by = collections.defaultdict(list)  # (ds, source, sl) -> list of res
    for k, v in all_res.items():
        ds, sl, seed, source = k.rsplit('_', 3)
        by[(ds, source, int(sl))].append(v)
    colors = {'lbv2': 'tab:blue', 'mech': 'tab:green'}
    for (ds, source, sl), rs in sorted(by.items()):
        ent = np.array([r['entropy'] for r in rs])      # [seeds, layers]
        entn = np.array([r['entropy_norm'] for r in rs])
        sink = np.array([r['sink'] for r in rs])
        near = np.array([r['near10'] for r in rs])
        for l in range(ent.shape[1]):
            c = colors.get(source, 'tab:red')
            linestyle = '-' if ds == 'exchange_rate' else '--'
            alpha = 0.45 if l == 0 else 0.9
            axes[0].errorbar(sl, entn[:, l].mean(),
                             yerr=entn[:, l].std() if len(rs) > 1 else None,
                             fmt='o' + linestyle, color=c, alpha=alpha, capsize=2)
            axes[1].errorbar(sl, near[:, l].mean(),
                             yerr=near[:, l].std() if len(rs) > 1 else None,
                             fmt='o' + linestyle, color=c, alpha=alpha, capsize=2)
            axes[2].errorbar(sl, sink[:, l].mean(),
                             yerr=sink[:, l].std() if len(rs) > 1 else None,
                             fmt='o' + linestyle, color=c, alpha=alpha, capsize=2)
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    axes[2].set_xscale('log')
    axes[0].set_ylabel('normalized entropy H/log(S)')
    axes[1].set_ylabel('mass on most recent 10% patches')
    axes[2].set_ylabel('mass on first (oldest) patch')
    for ax in axes:
        ax.set_xlabel('seq_len')
        ax.grid(alpha=0.3)
    axes[0].set_title('attention entropy vs lookback')
    axes[1].set_title('recency mass vs lookback')
    axes[2].set_title('sink (first-patch) mass vs lookback')
    fig.suptitle('solid=exchange_rate (lbv2), dashed=electricity; '
                 'blue=LBv2 ckpt, green=MECH uncapped vanilla; darker=deeper layer')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'exp1_entropy_vs_lookback.png'), dpi=150)
    plt.close(fig)
    print('[done]')


if __name__ == '__main__':
    main()
