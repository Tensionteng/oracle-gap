"""
Linear-probe diagnostic for mechanism M3 (sufficient-statistics shifting,
idea.md §5.1): how much future-window descriptor information is linearly
readable from the pooled hidden state of a frozen backbone, comparing a base
checkpoint against a descriptor-trained checkpoint of the same setting.

Both checkpoints are loaded into the hidden-exposing Desc variant of --model
with aux_head='none' (state dicts of the aux head, if present, are skipped),
and hidden is read via the forward(..., return_hidden=True) path on the
chosen split with the backbone frozen.

Probe protocol: time-ordered blocked split of the extracted windows (first
--probe_train_frac trains the probe, remainder evaluates), hidden
standardized with statistics from the probe-train part. Probes:
  cp_prob (binarized at 0.5) -> logistic regression (AUC, accuracy)
  cp_pos, spectral (5-dim)   -> ridge regression (R^2)
  drift / vol / slope (5-bin)-> multinomial logistic (accuracy)

Usage from the TSLib root (CPU):
  uv run python analysis/probe_descriptor.py \
    --base_ckpt checkpoints/long_term_forecast_ETTh1_96_96_PatchTST_ETTh1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --desc_ckpt checkpoints/long_term_forecast_ETTh1_96_96_desc_PatchSTDesc_ETTh1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_desc_0/checkpoint.pth \
    --setting ETTh1_96_96_patchtst --model PatchTST --data ETTh1 \
    --e_layers 1 --n_heads 2 --d_model 512 --d_ff 2048
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_dict
from utils.compute_descriptor_stats import ensure_descriptor_stats
from utils.descriptor_labels import compute_descriptor_targets

# stock backbone -> hidden-exposing variant (same weights, pooling is
# parameter-free for PatchTST/iTransformer; DLinearDesc.hidden_proj is a
# fixed random projection when probing a base DLinear checkpoint)
DESC_BACKBONE = {
    'PatchTST': 'PatchSTDesc',
    'iTransformer': 'iTransformerDesc',
    'DLinear': 'DLinearDesc',
}


def build_model_args(args):
    return SimpleNamespace(
        task_name='long_term_forecast', is_training=1, model_id='probe', model=args.model,
        data=args.data, root_path=args.root_path, data_path=args.data_path, features=args.features,
        target=args.target, freq=args.freq, checkpoints='./checkpoints/', max_train_windows=-1,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        seasonal_patterns='Monthly', inverse=False,
        enc_in=args.enc_in, dec_in=args.dec_in, c_out=args.c_out,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, moving_avg=args.moving_avg, factor=args.factor, dropout=args.dropout,
        embed=args.embed, activation=args.activation,
        batch_size=args.batch_size, num_workers=args.num_workers,
        augmentation_ratio=0, desc_k=args.desc_k, aux_head='none',
    )


def load_backbone(args, margs, ckpt_path):
    import importlib
    module = importlib.import_module('models.' + DESC_BACKBONE[args.model])
    model = module.Model(margs).float()
    state = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    print('{}: loaded {} (missing={}, unexpected={})'.format(
        os.path.basename(os.path.dirname(ckpt_path)), ckpt_path,
        len(missing), len(unexpected)))
    model.eval()
    return model


def extract(model, loader, margs, stats, device, probe_layer='last'):
    """Extract pooled hidden + descriptor targets. probe_layer='all'
    additionally registers hooks on the embedding and every encoder layer and
    pools each layer's output the same way as the final hidden, returning a
    dict {layer_name: [N, D]} (key 'last' = the stock pooled hidden, which for
    these encoder-only backbones is also the main head's input)."""
    per_layer = probe_layer == 'all'
    captured, handles = {}, []
    if per_layer:
        emb_mod = model.patch_embedding if hasattr(model, 'patch_embedding') else model.enc_embedding
        is_patch = hasattr(model, 'patch_embedding')

        def pool(out):
            o = out[0] if isinstance(out, tuple) else out
            if is_patch:  # [B*C, patch_num, d_model] -> [B, d_model]
                o = o.reshape(-1, margs.enc_in, o.shape[-2], o.shape[-1]).mean(dim=(1, 2))
            else:         # [B, N(+marks), d_model] -> [B, d_model]
                o = o[:, :margs.enc_in, :].mean(dim=1)
            return o

        def make_hook(name):
            def hook(module, inp, out):
                captured.setdefault(name, []).append(pool(out).detach().cpu().numpy())
            return hook
        handles.append(emb_mod.register_forward_hook(make_hook('emb')))
        for li, layer in enumerate(model.encoder.attn_layers):
            handles.append(layer.register_forward_hook(make_hook('enc{}'.format(li))))

    hiddens, targets = [], {k: [] for k in ['cp_prob', 'cp_pos', 'drift_cls', 'vol_cls', 'slope_cls', 'spectral']}
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -margs.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :margs.label_len, :], dec_inp], dim=1).float().to(device)
            _, hidden = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_hidden=True)
            f_dim = -1 if margs.features == 'MS' else 0
            future = batch_y[:, -margs.pred_len:, f_dim:]
            t = compute_descriptor_targets(future, margs.desc_k, stats)
            hiddens.append(hidden.cpu().numpy())
            for k in targets:
                targets[k].append(t[k].cpu().numpy())
    for h in handles:
        h.remove()
    T = {k: np.concatenate(v) for k, v in targets.items()}
    if not per_layer:
        return np.concatenate(hiddens), T
    H = {'last': np.concatenate(hiddens)}
    for name, chunks in captured.items():
        H[name] = np.concatenate(chunks)
    return H, T


def run_probes(H, T, train_frac, split_mode='blocked', seed=0):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, r2_score, accuracy_score

    n = H.shape[0]
    if split_mode == 'shuffle':
        # seeded permutation; adjacent test windows overlap, which inflates
        # absolute scores equally for base and desc - used only as a
        # robustness check alongside the default blocked split
        rng = np.random.RandomState(seed)
        order = rng.permutation(n)
    else:
        order = np.arange(n)
    n_tr = int(n * train_frac)
    tr, te = order[:n_tr], order[n_tr:]
    scaler = StandardScaler().fit(H[tr])
    X_tr, X_te = scaler.transform(H[tr]), scaler.transform(H[te])
    out = {}

    def split(v):
        return v[tr], v[te]

    # change-point probability (binarized) -> logistic
    y_tr, y_te = split((T['cp_prob'] > 0.5).astype(int))
    if len(np.unique(y_tr)) > 1 and len(np.unique(y_te)) > 1:
        clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, 1]
        out['cp_prob'] = {'auc': float(roc_auc_score(y_te, p)),
                          'acc': float(accuracy_score(y_te, p > 0.5)),
                          'pos_rate_test': float(y_te.mean())}
    else:
        out['cp_prob'] = {'auc': float('nan'), 'acc': float('nan'),
                          'pos_rate_test': float(y_te.mean())}

    # change-point position -> ridge
    y_tr, y_te = split(T['cp_pos'])
    reg = Ridge().fit(X_tr, y_tr)
    out['cp_pos'] = {'r2': float(r2_score(y_te, reg.predict(X_te)))}

    # spectral features (5-dim) -> ridge per dim
    y_tr, y_te = split(T['spectral'])
    reg = Ridge().fit(X_tr, y_tr)
    r2 = r2_score(y_te, reg.predict(X_te), multioutput='raw_values')
    out['spectral'] = {'r2_mean': float(r2.mean()), 'r2_per_dim': [float(v) for v in r2]}

    # ordinal classes -> multinomial logistic
    for key in ['drift_cls', 'vol_cls', 'slope_cls']:
        y_tr, y_te = split(T[key])
        clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
        majority = float(np.bincount(y_te.astype(int), minlength=5).max() / len(y_te))
        out[key] = {'acc': float(accuracy_score(y_te, clf.predict(X_te))),
                    'majority_acc': majority}
    return out


def diff_metrics(base, desc):
    d = {}
    for k in desc:
        d[k] = {}
        for m, v in desc[k].items():
            if isinstance(v, list):
                d[k][m] = [dv - bv for dv, bv in zip(v, base[k][m])]
            else:
                d[k][m] = v - base[k][m]
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_ckpt', type=str, required=True)
    parser.add_argument('--desc_ckpt', type=str, required=True)
    parser.add_argument('--setting', type=str, required=True, help='label used in the output filename')
    parser.add_argument('--model', type=str, default='PatchTST', choices=list(DESC_BACKBONE))
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--desc_k', type=int, default=24)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--max_windows', type=int, default=-1,
                        help='cap extracted windows by evenly-spaced subsampling; -1 = all')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--probe_train_frac', type=float, default=0.6)
    parser.add_argument('--probe_split', type=str, default='blocked', choices=['blocked', 'shuffle'],
                        help='blocked: time-ordered probe train/test; shuffle: seeded permutation (robustness check)')
    parser.add_argument('--probe_layer', type=str, default='last', choices=['last', 'all'],
                        help="all: probe the embedding and every encoder layer's pooled output as well")
    parser.add_argument('--device', type=str, default='cpu', help='e.g. cpu or cuda:0')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path or (args.data + '.csv')
    args.data_path = data_path
    out_path = args.out or os.path.join('analysis', 'results', 'probe_{}.json'.format(args.setting))

    margs = build_model_args(args)
    stats = ensure_descriptor_stats(margs)
    stats = {k: torch.from_numpy(np.asarray(v)).float() for k, v in stats.items() if v.dtype != object}

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    data_set = Data(args=margs, root_path=args.root_path, data_path=data_path, flag=args.split,
                    size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                    target=args.target, timeenc=timeenc, freq=args.freq,
                    seasonal_patterns=margs.seasonal_patterns)
    if 0 < args.max_windows < len(data_set):
        idx = np.unique(np.linspace(0, len(data_set) - 1, args.max_windows).round().astype(np.int64))
        data_set = Subset(data_set, idx.tolist())
    loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    print('probe split={}, windows={}'.format(args.split, len(data_set)))

    device = torch.device(args.device)
    stats = {k: v.to(device) for k, v in stats.items()}
    per_layer = args.probe_layer == 'all'
    scores = {}
    for name, ckpt in [('base', args.base_ckpt), ('desc', args.desc_ckpt)]:
        model = load_backbone(args, margs, ckpt).to(device)
        H, T = extract(model, loader, margs, stats, device, probe_layer=args.probe_layer)
        if not per_layer:
            scores[name] = run_probes(H, T, args.probe_train_frac, args.probe_split)
            print('{} hidden={} probe: {}'.format(name, H.shape, json.dumps(scores[name], indent=None)))
        else:
            scores[name] = {}
            for layer, Hl in H.items():
                scores[name][layer] = run_probes(Hl, T, args.probe_train_frac, args.probe_split)
                s = scores[name][layer]
                print('{} [{}] cp_auc={:.4f} cp_pos_r2={:.4f} spec_r2={:.4f} drift_acc={:.4f}'.format(
                    name, layer, s['cp_prob']['auc'], s['cp_pos']['r2'],
                    s['spectral']['r2_mean'], s['drift_cls']['acc']))

    if not per_layer:
        result = {
            'setting': args.setting, 'model': args.model, 'data': args.data, 'split': args.split,
            'pred_len': args.pred_len, 'desc_k': args.desc_k, 'n_windows': len(data_set),
            'd_hidden': int(H.shape[1]), 'probe_train_frac': args.probe_train_frac, 'probe_split': args.probe_split,
            'base_ckpt': args.base_ckpt, 'desc_ckpt': args.desc_ckpt,
            'base': scores['base'], 'desc': scores['desc'],
            'diff_desc_minus_base': diff_metrics(scores['base'], scores['desc']),
        }
    else:
        layers = list(scores['desc'].keys())
        result = {
            'setting': args.setting, 'model': args.model, 'data': args.data, 'split': args.split,
            'pred_len': args.pred_len, 'desc_k': args.desc_k, 'n_windows': len(data_set),
            'probe_train_frac': args.probe_train_frac, 'probe_split': args.probe_split, 'probe_layer': 'all', 'layers': layers,
            'base_ckpt': args.base_ckpt, 'desc_ckpt': args.desc_ckpt,
            'base': scores['base'], 'desc': scores['desc'],
            'diff_desc_minus_base': {l: diff_metrics(scores['base'][l], scores['desc'][l]) for l in layers},
        }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print('probe results written to {}'.format(out_path))


if __name__ == '__main__':
    main()
