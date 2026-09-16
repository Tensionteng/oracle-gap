"""
Descriptor-conditioned split-conformal calibration (decision-side piece 2):
can the descriptor head's per-window signals serve as conditioning variables
for conformal interval widths?

Two variants at nominal level alpha=0.9:
  (a) unconditional split-conformal: interval = pred +/- q, q = 90% quantile
      of validation absolute residuals (all points pooled);
  (b) conditional: width(window) = q_s * m_bin(window), where bins are formed
      by a predicted signal (pred vol expected bin or cp_prob) on validation
      windows, m_bin = mean absolute residual in the bin, and q_s = 90%
      quantile of the normalized scores |res|/m_bin on validation.

Metrics on the test split: marginal coverage, mean interval width, and
conditional coverage / width within cp_prob tertiles. A positive result: the
conditional variant achieves narrower widths at equal marginal coverage and
more uniform coverage across groups.

Usage from the TSLib root:
  uv run python analysis/conformal_desc.py \
    --desc_ckpt checkpoints/<desc_setting>/checkpoint.pth --model PatchTST \
    --data ETTh1 --e_layers 1 --n_heads 2 --d_model 512 --d_ff 2048 [--device cuda:0]
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_dict

DESC_BACKBONE = {'PatchTST': 'PatchSTDesc', 'iTransformer': 'iTransformerDesc'}
ALPHA = 0.9
N_BINS = 10


def collect(model, loader, args, device):
    preds, trues, sig_cp, sig_vol = [], [], [], []
    classes = torch.arange(5, device=device, dtype=torch.float)
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            outputs, aux = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_hidden=True)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            future = batch_y[:, -args.pred_len:, f_dim:]
            preds.append(outputs.cpu().numpy())
            trues.append(future.cpu().numpy())
            sig_cp.append(torch.sigmoid(aux['cp_prob']).cpu().numpy())
            sig_vol.append((torch.softmax(aux['vol'], dim=-1) * classes).sum(-1).cpu().numpy())
    return (np.concatenate(preds), np.concatenate(trues),
            np.concatenate(sig_cp), np.concatenate(sig_vol))


def fit_bins(sig, err_win, n_bins=N_BINS):
    """Equal-count bins over sig; returns (edges, m per bin, q_s) fitted on
    validation windows. err_win: per-window mean absolute residual."""
    edges = np.quantile(sig, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    assign = np.clip(np.digitize(sig, edges[1:-1]), 0, n_bins - 1)
    m = np.zeros(n_bins)
    for b in range(n_bins):
        m[b] = err_win[assign == b].mean() if (assign == b).any() else err_win.mean()
    scores = err_win / (m[assign] + 1e-12)
    q_s = float(np.quantile(scores, ALPHA))
    return edges, m, q_s


def evaluate(pred, true, widths, name):
    """widths: per-window half-width [N]. Returns marginal coverage/width."""
    half = widths.reshape(-1, 1, 1)
    inside = np.abs(pred - true) <= half + 1e-12
    return {'method': name, 'marginal_cov': float(inside.mean()),
            'mean_width': float((2 * widths).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc_ckpt', type=str, required=True)
    parser.add_argument('--setting', type=str, default=None)
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
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--desc_k', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path or (args.data + '.csv')
    setting = args.setting or os.path.basename(os.path.dirname(args.desc_ckpt))
    out_path = args.out or os.path.join('analysis', 'results', 'conformal_{}.json'.format(setting))

    margs = SimpleNamespace(
        task_name='long_term_forecast', is_training=1, model_id='conformal', model=args.model,
        data=args.data, root_path=args.root_path, data_path=data_path, features=args.features,
        target=args.target, freq=args.freq, checkpoints='./checkpoints/', max_train_windows=-1,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        seasonal_patterns='Monthly', inverse=False,
        enc_in=args.enc_in, dec_in=args.enc_in, c_out=args.enc_in,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, moving_avg=25, factor=args.factor, dropout=args.dropout,
        embed=args.embed, activation=args.activation,
        batch_size=args.batch_size, num_workers=args.num_workers,
        augmentation_ratio=0, desc_k=args.desc_k, aux_head='desc', desc_mode='pooled',
    )
    device = torch.device(args.device)

    import importlib
    module = importlib.import_module('models.' + DESC_BACKBONE[args.model])
    model = module.Model(margs).float().to(device)
    missing, unexpected = model.load_state_dict(torch.load(args.desc_ckpt, map_location='cpu'), strict=False)
    print('loaded {} (missing={}, unexpected={})'.format(args.desc_ckpt, len(missing), len(unexpected)))
    model.eval()

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    def make_loader(flag):
        ds = Data(args=margs, root_path=args.root_path, data_path=data_path, flag=flag,
                  size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                  target=args.target, timeenc=timeenc, freq=args.freq,
                  seasonal_patterns=margs.seasonal_patterns)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, drop_last=False)

    pv, yv, cp_v, vol_v = collect(model, make_loader('val'), args, device)
    pt, yt, cp_t, vol_t = collect(model, make_loader('test'), args, device)
    print('val={} test={}'.format(len(pv), len(pt)))

    err_v = np.abs(pv - yv)
    err_win_v = err_v.mean(axis=(1, 2))

    methods = {}
    # (a) unconditional
    q = float(np.quantile(err_v, ALPHA))
    methods['uncond'] = {'widths_t': np.full(len(pt), q), 'fit': {'q': q}}
    # (b) conditional on predicted vol bin expectation
    edges, m, q_s = fit_bins(vol_v, err_win_v)
    assign_t = np.clip(np.digitize(vol_t, edges[1:-1]), 0, N_BINS - 1)
    methods['cond_vol'] = {'widths_t': q_s * m[assign_t],
                           'fit': {'edges': edges.tolist(), 'm': m.tolist(), 'q_s': q_s}}
    # (c) conditional on cp_prob
    edges_c, m_c, q_sc = fit_bins(cp_v, err_win_v)
    assign_ct = np.clip(np.digitize(cp_t, edges_c[1:-1]), 0, N_BINS - 1)
    methods['cond_cp'] = {'widths_t': q_sc * m_c[assign_ct],
                          'fit': {'edges': edges_c.tolist(), 'm': m_c.tolist(), 'q_s': q_sc}}

    results = {}
    for name, mm in methods.items():
        results[name] = evaluate(pt, yt, mm['widths_t'], name)
        results[name]['fit'] = mm['fit']

    # width-at-equal-coverage frontier: scale each method's widths by the
    # constant that lands test coverage exactly at ALPHA, then compare widths
    # (the "at equal marginal coverage" comparison)
    for name, mm in methods.items():
        w = mm['widths_t']

        def cov(c):
            return float((np.abs(pt - yt) <= (c * w).reshape(-1, 1, 1) + 1e-12).mean())
        lo, hi = 0.0, 10.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if cov(mid) < ALPHA:
                lo = mid
            else:
                hi = mid
        c_star = 0.5 * (lo + hi)
        results[name]['width_at_cov09'] = float((2 * c_star * w).mean())
        results[name]['scale_to_cov09'] = float(c_star)

    # conditional coverage by cp_prob tertiles
    tert = np.quantile(cp_t, [0, 1 / 3, 2 / 3, 1.0])
    tert[0], tert[-1] = -np.inf, np.inf
    group = np.clip(np.digitize(cp_t, tert[1:-1]), 0, 2)
    cond = {}
    for name, mm in methods.items():
        half = mm['widths_t'].reshape(-1, 1, 1)
        inside = np.abs(pt - yt) <= half + 1e-12
        cond[name] = [{'group': g,
                       'n': int((group == g).sum()),
                       'coverage': float(inside[group == g].mean()),
                       'mean_width': float((2 * mm['widths_t'][group == g]).mean())}
                      for g in range(3)]

    out = {'setting': setting, 'model': args.model, 'data': args.data, 'alpha': ALPHA,
           'n_val': len(pv), 'n_test': len(pt), 'methods': results,
           'by_cp_prob_tertile': cond}
    print('\nmarginal coverage / mean width:')
    for name, r in results.items():
        print('  {:<10s} cov={:.4f} width={:.4f} | width@cov=0.9: {:.4f} (scale {:.3f})'.format(
            name, r['marginal_cov'], r['mean_width'], r['width_at_cov09'], r['scale_to_cov09']))
    print('by cp_prob tertile (coverage / width):')
    for g in range(3):
        row = '  G{}: '.format(g)
        for name in methods:
            row += '{} {:.3f}/{:.3f}  '.format(name, cond[name][g]['coverage'], cond[name][g]['mean_width'])
        print(row)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('written: {}'.format(out_path))


if __name__ == '__main__':
    main()
