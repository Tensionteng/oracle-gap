#!/usr/bin/env python
"""H2 error-correlation probe on the fine-tuned Chronos-Bolt desc arm
(supporting evidence for section 7: the selective-prediction signal is a
zero-cost by-product of the descriptor head, needs no validation-set training).

For a desc-arm bolt checkpoint (fine-tuned Chronos-Bolt + descriptor head):
  signal   : per-window cp_prob = channel-mean sigmoid(cp_prob logit) read
             from the frozen fine-tuned model's aux head;
  error    : per-window MSE from the run's own pred_dumps (official rollout
             evaluation protocol);
  baselines: random abstention, ctx_vol (lookback volatility), oracle_vol
             (future-window volatility, upper bound).
Outputs Spearman correlations and risk-coverage (abstain highest-signal
first), reusing the same protocol as TSLib analysis/error_correlation.py.

Usage from tsfm_stage2/:
  uv run python errcorr_bolt.py --data ETTh1 [--device cuda:0]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from finetune_chronos_bolt import ETTWindows, BoltWithAux


def risk_coverage(err, signal, grid):
    order = np.argsort(-signal)
    out = []
    for c in grid:
        n_drop = int(len(err) * (1 - c))
        keep = order[n_drop:]
        out.append(float(err[keep].mean()) if len(keep) else float('nan'))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='')
    p.add_argument('--dump', default='')
    p.add_argument('--data', default='ETTh1')
    p.add_argument('--root_path', default=os.path.join(ROOT, '..', 'Time-Series-Library', 'dataset', 'ETT-small'))
    p.add_argument('--data_path', default=None)
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='')
    args = p.parse_args()
    run_name = 'tsfm_bolts_descw01_{}'.format(args.data)
    ckpt = args.ckpt or os.path.join(ROOT, 'checkpoints', run_name)
    dump = args.dump or os.path.join(ROOT, 'pred_dumps', run_name)
    data_path = args.data_path or (args.data + '.csv')
    out_path = args.out or os.path.join(ROOT, 'results', 'errcorr_bolt_{}.json'.format(args.data))

    device = args.device
    model_wrapper = BoltWithAux(ckpt, aux_head='desc', device=device)
    head_path = os.path.join(ckpt, 'aux_head.pt')
    model_wrapper.head.load_state_dict(torch.load(head_path, map_location=device))
    print('loaded {} + aux_head.pt'.format(ckpt))
    model_wrapper.eval()

    test_ds = ETTWindows(args.root_path, data_path, 'test', args.seq_len, args.pred_len,
                         data=args.data)
    pred = np.load(os.path.join(dump, 'pred.npy'))
    true = np.load(os.path.join(dump, 'true.npy'))
    err = ((pred - true) ** 2).mean(axis=(1, 2))
    assert len(err) == len(test_ds), 'dump rows {} != test windows {}'.format(len(err), len(test_ds))

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    sig_cp, ctx_vol, oracle_vol = [], [], []
    with torch.no_grad():
        for context, future in loader:
            B, L, C = context.shape
            ctx = context.permute(0, 2, 1).reshape(B * C, L).to(device)
            _, aux = model_wrapper(ctx)
            cp = torch.sigmoid(aux['cp_prob']).reshape(B, C).mean(dim=1)
            sig_cp.append(cp.cpu().numpy())
            ctx_vol.append(context.diff(dim=1).std(dim=1).mean(dim=1).numpy())
            oracle_vol.append(future.diff(dim=1).std(dim=1).mean(dim=1).numpy())
    sig_cp = np.concatenate(sig_cp)
    ctx_vol = np.concatenate(ctx_vol)
    oracle_vol = np.concatenate(oracle_vol)

    from scipy.stats import spearmanr
    grid = np.round(np.arange(1.0, 0.499, -0.02), 2)
    rng = np.random.RandomState(2021)
    signals = {'pred_cp': sig_cp, 'ctx_vol': ctx_vol, 'oracle_vol': oracle_vol}
    curves = {k: risk_coverage(err, v, grid) for k, v in signals.items()}
    curves['random'] = list(np.mean([risk_coverage(err, rng.permutation(len(err)).astype(float), grid)
                                     for _ in range(10)], axis=0))

    def at_cov(curve, c=0.8):
        return curve[int(np.argmin(np.abs(grid - c)))]

    result = {'run': run_name, 'ckpt': ckpt, 'dump': dump, 'data': args.data,
              'n_windows': int(len(err)), 'mse_overall': float(err.mean()),
              'spearman_vs_window_mse': {k: float(spearmanr(v, err).statistic) for k, v in signals.items()},
              'remaining_mse_at_cov08': {k: at_cov(curves[k]) for k in curves},
              'curves': {k: curves[k] for k in curves},
              'coverage_grid': grid.tolist()}
    print('Spearman (signal vs window MSE):')
    for k, v in result['spearman_vs_window_mse'].items():
        print('  {:<12s} {:+.4f}'.format(k, v))
    print('remaining MSE @ coverage=0.8 (overall={:.4f}):'.format(err.mean()))
    for k, v in result['remaining_mse_at_cov08'].items():
        print('  {:<12s} {:.4f} ({:+.2%} vs overall)'.format(k, v, v / err.mean() - 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print('written: {}'.format(out_path))


if __name__ == '__main__':
    main()
