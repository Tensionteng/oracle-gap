"""
Agreement check: CUSUM change-point proxy (utils/descriptor_labels.py) vs
ruptures PELT on sampled training windows. CPU-only diagnostic used to
calibrate CP_CAL_A / CP_CAL_B.

Run from the TSLib root, e.g.:
  uv run python analysis/verify_cp_labels.py --data ETTh1 --pred_len 96 --k 24 --n 200

Reported agreement (final calibration A=4.0 B=7.5, n=200 windows x 7 channels
= 1400 channel-windows, PELT model='l2' min_size=3 pen=4.0, proxy threshold
prob>0.5; run 2026-09-09 on CPU):
  ETTh1 train: per-channel agreement 0.6886 (PELT pos rate 0.286, proxy pos
               rate 0.269; window-level any-channel agreement 0.7300)
  ETTm1 train: per-channel agreement 0.6836 (PELT pos rate 0.119, proxy pos
               rate 0.392)
  Note: maximizing raw agreement degenerates toward an always-negative proxy
  (best grid cell ~0.71-0.87 via near-zero positive rates); the in-use
  constants instead place the prob>0.5 threshold at the PELT-positive-rate-
  matched score quantile, keeping the auxiliary BCE signal balanced.
  The script also prints a small (A, B) grid so the constants in
  utils/descriptor_labels.py can be re-checked.
"""
import argparse
import math
from types import SimpleNamespace

import numpy as np
import torch
import ruptures as rpt

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_provider
from utils.descriptor_labels import CP_CAL_A, CP_CAL_B


def build_args(data, root_path, data_path, seq_len, label_len, pred_len):
    return SimpleNamespace(
        task_name='long_term_forecast', is_training=1, model_id='verify_cp', model='PatchTST',
        data=data, root_path=root_path, data_path=data_path, features='M', target='OT',
        freq='h', checkpoints='./checkpoints/', max_train_windows=-1,
        seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        seasonal_patterns='Monthly', inverse=False,
        batch_size=256, num_workers=0, embed='timeF', augmentation_ratio=0,
    )


def per_channel_cusum_scores(future, k):
    """future: [L, C] numpy -> [C] scale-free CUSUM scores on the first k steps."""
    w = torch.from_numpy(future[:k]).float().T.unsqueeze(0)      # [1, C, k] -> treat channels as batch
    w = w.permute(0, 2, 1)                                       # [1, k, C]
    mu = w.mean(dim=1, keepdim=True)
    sd = w.std(dim=1, unbiased=False, keepdim=True) + 1e-8
    s = (w - mu).cumsum(dim=1).abs()
    score = s.amax(dim=1) / (sd.squeeze(1) * math.sqrt(k))       # [1, C]
    return score.squeeze(0).numpy()


def pelt_channel_positive(x, pen, min_size=3):
    bps = rpt.Pelt(model='l2', min_size=min_size, jump=1).fit(x).predict(pen=pen)
    # ruptures returns segment ends incl. len(x); a real change point is < len
    return any(bp < len(x) for bp in bps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--k', type=int, default=24)
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--pen', type=float, default=4.0,
                        help='PELT penalty; default 4.0 gives a ~30% positive rate on ETT train windows')
    args = parser.parse_args()
    data_path = args.data_path or (args.data + '.csv')
    pen = args.pen

    dargs = build_args(args.data, args.root_path, data_path, args.seq_len, args.label_len, args.pred_len)
    train_data, _ = data_provider(dargs, flag='train')

    n_total = len(train_data)
    idxs = np.linspace(0, n_total - 1, min(args.n, n_total)).round().astype(np.int64)
    idxs = np.unique(idxs)

    scores, pelt_pos = [], []
    for i in idxs:
        seq_y = train_data[int(i)][1]                            # [label_len+pred_len, C]
        future = np.asarray(seq_y[-args.pred_len:], dtype=np.float64)
        scores.append(per_channel_cusum_scores(future, args.k))  # [C]
        w = future[:args.k]
        pelt_pos.append([pelt_channel_positive(w[:, c], pen) for c in range(w.shape[1])])
    scores = np.stack(scores)                                    # [N, C]
    pelt_pos = np.array(pelt_pos)                                # [N, C] bool

    print('data={} pred_len={} k={} pen={:.3f} | windows={} channels={}'.format(
        args.data, args.pred_len, args.k, pen, scores.shape[0], scores.shape[1]))
    print('CUSUM score quantiles [10/25/50/75/90]%: {}'.format(
        np.round(np.quantile(scores, [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist()))
    print('PELT positive rate (per-channel): {:.3f}'.format(pelt_pos.mean()))

    # agreement with the constants actually used in training
    prob = 1.0 / (1.0 + np.exp(-(CP_CAL_A * scores - CP_CAL_B)))
    proxy_pos = prob > 0.5
    agree = (proxy_pos == pelt_pos).mean()
    print('calibration in use: A={} B={}'.format(CP_CAL_A, CP_CAL_B))
    print('proxy positive rate (per-channel): {:.3f}'.format(proxy_pos.mean()))
    print('PER-CHANNEL AGREEMENT: {:.4f}'.format(agree))

    # window-level views
    win_proxy = proxy_pos.any(axis=1)
    win_pelt_any = pelt_pos.any(axis=1)
    win_pelt_maj = pelt_pos.mean(axis=1) >= 0.5
    print('window-level agreement (any-channel PELT): {:.4f}'.format((win_proxy == win_pelt_any).mean()))
    print('window-level agreement (majority PELT):   {:.4f}'.format((win_proxy == win_pelt_maj).mean()))

    # small calibration grid for re-checking the constants
    print('\n(A, B) grid, per-channel agreement (top 5):')
    grid = []
    for a in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        for b in [3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0]:
            g = (((a * scores - b) > 0) == pelt_pos).mean()
            grid.append((g, a, b))
    for g, a, b in sorted(grid, reverse=True)[:5]:
        print('  A={:.1f} B={:.1f} -> {:.4f}'.format(a, b, g))


if __name__ == '__main__':
    main()
