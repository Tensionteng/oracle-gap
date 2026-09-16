"""
Behavioral metrics for a prediction dump: undershoot / overshoot / wrong-side
rates and the amplitude ratio, computed per point against a lookback-mean
anchor (same anchor convention as regionfocal ctxmean).

Reads a directory with pred.npy / true.npy ([N, pred_len, C]) - either
pred_dumps/<setting>/ or the stock results/<setting>/ - and rebuilds the test
dataset to recover the input window of each row (test window_index i ==
dataset row i, input = data_x[i:i+seq_len]).

Usage from the TSLib root:
  uv run python analysis/behavior_metrics.py \
    --dump pred_dumps/long_term_forecast_<setting> --data ETTh1 \
    [--data_path ETTh1.csv --seq_len 96 --pred_len 96]

Output: analysis/results/behavior_<name>.json
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump', type=str, required=True,
                        help='dir with pred.npy/true.npy (pred_dumps/ or results/)')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--anchor', type=str, default='ctxmean', choices=['ctxmean', 'zero'])
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path or (args.data + '.csv')
    name = os.path.basename(args.dump.rstrip('/'))
    out_path = args.out or os.path.join('analysis', 'results', 'behavior_{}.json'.format(name))

    pred = np.load(os.path.join(args.dump, 'pred.npy')).astype(np.float64)
    true = np.load(os.path.join(args.dump, 'true.npy')).astype(np.float64)
    assert pred.shape == true.shape and pred.shape[1] == args.pred_len

    if args.anchor == 'ctxmean':
        margs = SimpleNamespace(
            task_name='long_term_forecast', max_train_windows=-1, augmentation_ratio=0)
        Data = data_dict[args.data]
        timeenc = 1
        data_set = Data(args=margs, root_path=args.root_path, data_path=data_path, flag='test',
                        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                        target=args.target, timeenc=timeenc, freq=args.freq,
                        seasonal_patterns='Monthly')
        widx_path = os.path.join(args.dump, 'window_index.npy')
        widx = np.load(widx_path) if os.path.exists(widx_path) else np.arange(pred.shape[0])
        data_x = np.asarray(data_set.data_x)                      # [T, C] scaled
        anchor = np.stack([data_x[int(i):int(i) + args.seq_len].mean(axis=0) for i in widx])
        anchor = anchor[:, None, :]                               # [N, 1, C]
        # drop_last / feature slicing parity with the dump
        anchor = anchor[:, :, -true.shape[-1]:] if true.shape[-1] != anchor.shape[-1] else anchor
    else:
        anchor = np.zeros((pred.shape[0], 1, pred.shape[2]))

    d = true - anchor
    p = pred - anchor
    same_side = p * d > 0
    under = same_side & (np.abs(p) < np.abs(d))
    over = same_side & (np.abs(p) > np.abs(d))
    wrong = p * d < 0
    out = {
        'dump': args.dump, 'anchor': args.anchor, 'n_windows': int(pred.shape[0]),
        'pred_len': int(pred.shape[1]), 'channels': int(pred.shape[2]),
        'undershoot_rate': float(under.mean()),
        'overshoot_rate': float(over.mean()),
        'wrong_side_rate': float(wrong.mean()),
        'amp_ratio_std': float(pred.std() / (true.std() + 1e-12)),
        'mse': float(((pred - true) ** 2).mean()),
    }
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('written: {}'.format(out_path))


if __name__ == '__main__':
    main()
