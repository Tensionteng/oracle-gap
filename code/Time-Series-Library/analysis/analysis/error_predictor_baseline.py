"""
Trained error-predictor baseline for selective prediction (reviewer request:
"selective prediction was only compared against static signals").

Trains a small predictor of the per-window forecast error on the VALIDATION
split using only lookback-context statistics (no future information), then
evaluates it on the test split with the same risk-coverage protocol as
analysis/error_correlation.py, side by side with the descriptor head's
pred_cp signal, context volatility, the oracle volatility upper bound, and
random abstention.

Features per window (channel-mean, all from the seq_len lookback):
  mean, std, diff-std, trend slope, |slope|, spectral centroid, spectral
  entropy, lag-1 autocorrelation, max |diff|, scale-free CUSUM score of the
  context. Predictors: ridge (primary) and a 2-layer MLP (secondary).

Usage from the TSLib root:
  uv run python analysis/error_predictor_baseline.py \
    --desc_ckpt checkpoints/<desc_setting>/checkpoint.pth --model PatchTST \
    --data ETTh1 --e_layers 1 --n_heads 2 --d_model 512 --d_ff 2048 [--device cuda:0]
"""
import argparse
import json
import math
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


def context_features(x):
    """x: [B, L, C] -> [B, 10] channel-mean context statistics."""
    B, L, C = x.shape
    feats = [x.mean(dim=(1, 2)),
             x.std(dim=(1, 2), unbiased=False),
             x.diff(dim=1).std(dim=(1, 2), unbiased=False)]
    t = torch.arange(L, device=x.device, dtype=x.dtype)
    t = t - t.mean()
    slope = ((x - x.mean(dim=1, keepdim=True)) * t.view(1, L, 1)).mean(dim=1) / (t * t).mean()
    feats += [slope.mean(dim=1), slope.abs().mean(dim=1)]
    xc = x - x.mean(dim=1, keepdim=True)
    spec = torch.fft.rfft(xc, dim=1)
    power = spec.real ** 2 + spec.imag ** 2
    p = power / (power.sum(dim=1, keepdim=True) + 1e-8)
    n_freq = p.shape[1]
    freqs = torch.linspace(0.0, 1.0, n_freq, device=x.device)
    feats.append((p * freqs.view(1, -1, 1)).sum(dim=1).mean(dim=1))
    feats.append((-(p * (p + 1e-8).log()).sum(dim=1) / math.log(n_freq)).mean(dim=1))
    x0, x1 = xc[:, :-1, :], xc[:, 1:, :]
    lag1 = ((x0 * x1).mean(dim=(1, 2)) / (xc.var(dim=(1, 2), unbiased=False) + 1e-8))
    feats.append(lag1)
    feats.append(x.diff(dim=1).abs().amax(dim=(1, 2)))
    mu = xc.mean(dim=1, keepdim=True)
    sd = xc.std(dim=1, unbiased=False, keepdim=True) + 1e-8
    cusum = ((xc - mu).cumsum(dim=1).abs().amax(dim=1) / (sd.squeeze(1) * math.sqrt(L))).mean(dim=1)
    feats.append(cusum)
    return torch.stack(feats, dim=1)


def collect(model, loader, args, stats, device):
    """Per-window: pred_cp signal, context features, forecast MSE, oracle vol."""
    sig_cp, feats, errs, ovol = [], [], [], []
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
            sig_cp.append(torch.sigmoid(aux['cp_prob']).cpu().numpy())
            feats.append(context_features(batch_x[:, :, f_dim:]).cpu().numpy())
            errs.append(((outputs - future) ** 2).mean(dim=(1, 2)).cpu().numpy())
            t = compute_descriptor_targets(future, args.desc_k, stats)
            ovol.append(mid_window_stats(future)[1].cpu().numpy())
    return (np.concatenate(sig_cp), np.concatenate(feats),
            np.concatenate(errs), np.concatenate(ovol))


def risk_coverage(err, signal, grid):
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
    args.data_path = data_path
    setting = args.setting or os.path.basename(os.path.dirname(args.desc_ckpt))
    out_path = args.out or os.path.join('analysis', 'results', 'errpred_{}.json'.format(setting))

    margs = SimpleNamespace(
        task_name='long_term_forecast', is_training=1, model_id='errpred', model=args.model,
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

    def make_loader(flag):
        ds = Data(args=margs, root_path=args.root_path, data_path=data_path, flag=flag,
                  size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                  target=args.target, timeenc=timeenc, freq=args.freq,
                  seasonal_patterns=margs.seasonal_patterns)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, drop_last=False)

    cp_v, X_v, err_v, _ = collect(model, make_loader('val'), args, stats, device)
    cp_t, X_t, err_t, ovol_t = collect(model, make_loader('test'), args, stats, device)
    print('val windows={} test windows={}'.format(len(err_v), len(err_t)))

    # ridge predictor on validation windows
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_v)
    ridge = Ridge(alpha=1.0).fit(sc.transform(X_v), err_v)
    pred_ridge = ridge.predict(sc.transform(X_t))

    # small MLP predictor (same features)
    torch.manual_seed(2021)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(X_v.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 1)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xv_t = torch.from_numpy(sc.transform(X_v)).float().to(device)
    yv_t = torch.from_numpy(err_v).float().to(device)
    for it in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(mlp(Xv_t).squeeze(-1), yv_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_mlp = mlp(torch.from_numpy(sc.transform(X_t)).float().to(device)).squeeze(-1).cpu().numpy()

    ctx_vol_t = X_t[:, 2]  # diff-std feature
    grid = np.round(np.arange(1.0, 0.499, -0.02), 2)
    rng = np.random.RandomState(2021)
    signals = {'pred_cp': cp_t, 'errpred_ridge': pred_ridge, 'errpred_mlp': pred_mlp,
               'ctx_vol': ctx_vol_t, 'oracle_vol': ovol_t}
    curves = {k: risk_coverage(err_t, v, grid) for k, v in signals.items()}
    curves['random'] = list(np.mean([risk_coverage(err_t, rng.permutation(len(err_t)).astype(float), grid)
                                     for _ in range(10)], axis=0))

    def at_cov(curve, c=0.8):
        return curve[int(np.argmin(np.abs(grid - c)))]

    def spearman(x, y):
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)

    result = {'setting': setting, 'model': args.model, 'data': args.data,
              'n_val': int(len(err_v)), 'n_test': int(len(err_t)),
              'mse_overall': float(err_t.mean()),
              'spearman_vs_window_mse': {**{k: spearman(v, err_t) for k, v in signals.items()},
                                         'errpred_ridge': spearman(pred_ridge, err_t),
                                         'errpred_mlp': spearman(pred_mlp, err_t)},
              'remaining_mse_at_cov08': {k: at_cov(curves[k]) for k in curves},
              'coverage_grid': grid.tolist(),
              'note': 'error predictors trained on the validation split (context statistics only)'}
    print('Spearman (signal vs window MSE):')
    for k, v in result['spearman_vs_window_mse'].items():
        print('  {:<16s} {:+.4f}'.format(k, v))
    print('remaining MSE @ coverage=0.8 (overall={:.4f}):'.format(err_t.mean()))
    for k, v in result['remaining_mse_at_cov08'].items():
        print('  {:<16s} {:.4f} ({:+.2%} vs overall)'.format(k, v, v / err_t.mean() - 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print('written: {}'.format(out_path))


if __name__ == '__main__':
    main()
