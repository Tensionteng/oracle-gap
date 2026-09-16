"""
H2 error-correlation probe (zero-training, decision-side evidence): can the
descriptor head's per-window outputs (predicted cp_prob / volatility bin /
drift bin / spectral features) predict the window's forecast error?

For a desc-trained checkpoint, collects on the test split:
  predicted signals : cp_prob = sigmoid(logit), expected bin index of
                      drift/vol/slope (softmax . [0..4]), max-class
                      confidence, 5-dim spectral predictions;
  window error      : per-window MSE from the run's pred_dumps (pred/true);
  baseline signals  : oracle (true calibrated cp_prob target and true
                      volatility of the future window), context (lookback
                      volatility = channel-mean std of first differences),
                      random (mean of seeded permutations).
and reports Spearman correlations plus risk-coverage curves (sort windows by
signal descending, drop highest-risk first, coverage 1.0 -> 0.5): remaining
mean MSE vs coverage. If the predicted-signal curve sits well below random
and close to oracle, the descriptor carries decision-usable error information
(H2 minimal evidence).

Note on loading: the model is constructed WITH aux_head='desc' so the aux head
modules exist and the checkpoint's aux_head.* weights load (strict=False just
in case); constructing with aux_head='none' would drop exactly the head we
need to read.

Usage from the TSLib root:
  uv run python analysis/error_correlation.py \
    --desc_ckpt checkpoints/<desc_setting>/checkpoint.pth \
    --dump pred_dumps/<desc_setting> --model PatchTST --data ETTh1 \
    --e_layers 1 --n_heads 2 --d_model 512 --d_ff 2048 [--device cuda:0]
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
from utils.compute_descriptor_stats import ensure_descriptor_stats
from utils.descriptor_labels import compute_descriptor_targets, mid_window_stats

DESC_BACKBONE = {'PatchTST': 'PatchSTDesc', 'iTransformer': 'iTransformerDesc'}


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def risk_coverage(err, signal, grid):
    # selective prediction: sort by signal descending, ABSTAIN the top
    # (1-c) fraction as highest-risk, report the mean MSE of the remaining
    # (lowest-signal) windows; a useful signal drives this below the overall MSE
    order = np.argsort(-signal)
    out = []
    for c in grid:
        n_drop = int(len(err) * (1 - c))
        keep = order[n_drop:]
        out.append(float(err[keep].mean()) if len(keep) else float('nan'))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc_ckpt', type=str, required=True)
    parser.add_argument('--dump', type=str, default=None,
                        help='pred_dumps dir of the same run (default: pred_dumps/<ckpt parent name>)')
    parser.add_argument('--setting', type=str, default=None)
    parser.add_argument('--model', type=str, default='PatchTST', choices=list(DESC_BACKBONE))
    parser.add_argument('--desc_mode', type=str, default='pooled')
    parser.add_argument('--vol_ms', type=int, default=0,
                        help='read the multi-scale vol signal (0.5*E[vol]+0.5*E[vol_near]) from a --vol_ms checkpoint')
    parser.add_argument('--vol_qr', type=int, default=0,
                        help='read the continuous log-vol prediction from a --vol_qr checkpoint')
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
    dump = args.dump or os.path.join('pred_dumps', setting)
    out_path = args.out or os.path.join('analysis', 'results', 'errcorr_{}.json'.format(setting))

    margs = SimpleNamespace(
        task_name='long_term_forecast', is_training=1, model_id='errcorr', model=args.model,
        data=args.data, root_path=args.root_path, data_path=data_path, features=args.features,
        target=args.target, freq=args.freq, checkpoints='./checkpoints/', max_train_windows=-1,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        seasonal_patterns='Monthly', inverse=False,
        enc_in=args.enc_in, dec_in=args.enc_in, c_out=args.enc_in,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, moving_avg=25, factor=args.factor, dropout=args.dropout,
        embed=args.embed, activation=args.activation,
        batch_size=args.batch_size, num_workers=args.num_workers,
        augmentation_ratio=0, desc_k=args.desc_k, aux_head='desc', desc_mode=args.desc_mode,
    )
    device = torch.device(args.device)
    stats = ensure_descriptor_stats(margs)
    stats = {k: torch.from_numpy(np.asarray(v)).float().to(device)
             for k, v in stats.items() if v.dtype != object}

    import importlib
    module = importlib.import_module('models.' + DESC_BACKBONE[args.model])
    model = module.Model(margs).float().to(device)
    missing, unexpected = model.load_state_dict(torch.load(args.desc_ckpt, map_location='cpu'), strict=False)
    print('loaded {} (missing={}, unexpected={})'.format(args.desc_ckpt, len(missing), len(unexpected)))
    model.eval()

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    data_set = Data(args=margs, root_path=args.root_path, data_path=data_path, flag='test',
                    size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                    target=args.target, timeenc=timeenc, freq=args.freq,
                    seasonal_patterns=margs.seasonal_patterns)
    loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    pred = np.load(os.path.join(dump, 'pred.npy'))
    true = np.load(os.path.join(dump, 'true.npy'))
    err = ((pred - true) ** 2).mean(axis=(1, 2))
    assert len(err) == len(data_set), 'dump rows {} != test windows {}'.format(len(err), len(data_set))

    sig = {k: [] for k in ['pred_cp', 'pred_drift_exp', 'pred_vol_exp', 'pred_slope_exp',
                           'pred_vol_conf', 'oracle_cp', 'oracle_vol', 'ctx_vol']}
    if args.vol_ms:
        sig['pred_volms_exp'] = []
    if args.vol_qr:
        sig['pred_volqr'] = []
    classes = torch.arange(5, device=device, dtype=torch.float)
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            _, aux = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_hidden=True)
            f_dim = -1 if args.features == 'MS' else 0
            future = batch_y[:, -args.pred_len:, f_dim:]

            sig['pred_cp'].append(torch.sigmoid(aux['cp_prob']).cpu().numpy())
            vol_p = torch.softmax(aux['vol'], dim=-1)
            sig['pred_vol_exp'].append((vol_p * classes).sum(-1).cpu().numpy())
            sig['pred_vol_conf'].append(vol_p.max(-1).values.cpu().numpy())
            if args.vol_ms and 'vol_near' in aux:
                voln_p = torch.softmax(aux['vol_near'], dim=-1)
                sig['pred_volms_exp'].append(
                    (0.5 * (vol_p * classes).sum(-1) + 0.5 * (voln_p * classes).sum(-1)).cpu().numpy())
            if args.vol_qr and 'vol_qr' in aux:
                sig['pred_volqr'].append(aux['vol_qr'].cpu().numpy())
            sig['pred_drift_exp'].append((torch.softmax(aux['drift'], dim=-1) * classes).sum(-1).cpu().numpy())
            sig['pred_slope_exp'].append((torch.softmax(aux['slope'], dim=-1) * classes).sum(-1).cpu().numpy())

            t = compute_descriptor_targets(future, args.desc_k, stats)
            sig['oracle_cp'].append(t['cp_prob'].cpu().numpy())
            _, vol_true, _ = mid_window_stats(future)
            sig['oracle_vol'].append(vol_true.cpu().numpy())
            sig['ctx_vol'].append(batch_x[:, :, f_dim:].diff(dim=1).std(dim=1).mean(dim=1).cpu().numpy())
    S = {k: np.concatenate(v).reshape(len(err), -1) for k, v in sig.items()}
    if args.desc_mode == 'channel':  # per-channel signals -> channel mean
        S = {k: v.mean(axis=1) for k, v in S.items()}
    else:
        S = {k: v[:, 0] for k, v in S.items()}

    grid = np.round(np.arange(1.0, 0.499, -0.02), 2)
    rng = np.random.RandomState(2021)
    curves = {k: risk_coverage(err, S[k], grid) for k in S}
    curves['random'] = np.mean([risk_coverage(err, rng.permutation(len(err)).astype(float), grid)
                                for _ in range(10)], axis=0).tolist()
    curves['random'] = list(curves['random'])

    def at_cov(curve, c=0.8):
        return curve[int(np.argmin(np.abs(grid - c)))]

    result = {'setting': setting, 'model': args.model, 'data': args.data, 'n_windows': int(len(err)),
              'ckpt': args.desc_ckpt, 'dump': dump,
              'mse_overall': float(err.mean()),
              'spearman_vs_window_mse': {k: spearman(S[k], err) for k in S},
              'remaining_mse_at_cov08': {k: at_cov(curves[k]) for k in curves},
              'coverage_grid': grid.tolist()}
    print('Spearman (signal vs window MSE):')
    for k, v in result['spearman_vs_window_mse'].items():
        print('  {:<16s} {:+.4f}'.format(k, v))
    print('remaining MSE @ coverage=0.8 (overall={:.4f}):'.format(err.mean()))
    for k, v in result['remaining_mse_at_cov08'].items():
        print('  {:<16s} {:.4f} ({:+.2%} vs overall)'.format(k, v, v / err.mean() - 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    np.savez(out_path.replace('.json', '_curves.npz'),
             coverage=grid, err=err, **{('sig_' + k): S[k] for k in S},
             **{('curve_' + k): np.asarray(curves[k]) for k in curves})
    print('written: {} (+ _curves.npz)'.format(out_path))


if __name__ == '__main__':
    main()
