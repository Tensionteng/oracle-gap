#!/usr/bin/env python
"""Assemble a unified dump for a TSLib PatchTST run on a gapbench wide CSV.

TSLib's test() writes results/<setting>/{pred,true}.npy in the scaler space
(default --inverse 0). This script locates that directory by --model_id glob,
rebuilds the matching input contexts by replicating Dataset_Custom's split +
StandardScaler exactly (verified against Dataset_Custom itself), and writes
<out>/{pred,true,inputs}.npy.

--tail_only: keep only the final test window of each channel, i.e. exactly the
adapter's tail window (equal-length collections), so the task model can be
scored on the same N=n_items window set as the TSFM tail dump.
"""
import argparse
import glob
import json
import os

import numpy as np

TSLIB = '" + os.environ.get("MTP4TS_ROOT", ".") + "/Time-Series-Library'


def build_test_inputs(csv_path, seq_len, pred_len):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    cols.remove('OT')
    cols.remove('date')
    df = df[['date'] + cols + ['OT']]
    df_data = df[df.columns[1:]]
    n = len(df)
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    b1, b2 = n - num_test - seq_len, n
    scaler = StandardScaler()
    scaler.fit(df_data.values[:num_train])
    data = scaler.transform(df_data.values)
    n_win = (b2 - b1) - seq_len - pred_len + 1
    inputs = np.stack([data[b1 + i:b1 + i + seq_len] for i in range(n_win)])
    return inputs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_id', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--seq_len', type=int, default=96)
    ap.add_argument('--pred_len', type=int, default=96)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tail_only', action='store_true')
    ap.add_argument('--tail_strided', type=int, default=0,
                    help='keep the K strided (stride=pred_len) tail windows per channel '
                         'that fall inside the test region; matches windows_full multi-window mode')
    ap.add_argument('--results_root', default=os.path.join(TSLIB, 'results'))
    args = ap.parse_args()

    hits = sorted(glob.glob(os.path.join(args.results_root,
                                         'long_term_forecast_{}_*'.format(args.model_id))))
    if not hits:
        raise FileNotFoundError('no TSLib results for model_id {}'.format(args.model_id))
    res_dir = hits[-1]
    print('results dir: {}'.format(res_dir))
    pred = np.load(os.path.join(res_dir, 'pred.npy'))
    true = np.load(os.path.join(res_dir, 'true.npy'))
    inputs = build_test_inputs(args.csv, args.seq_len, args.pred_len)
    assert inputs.shape[0] == pred.shape[0], (inputs.shape, pred.shape)
    print('windows={} pred {} true {} inputs {}'.format(pred.shape[0], pred.shape, true.shape, inputs.shape))

    if args.tail_strided > 0:
        # test row of the k-th strided tail window (end = n - k*pred_len):
        #   b1 + i + seq_len + pred_len = n - k*pred_len  =>  i = n_win-1-k*(pred_len/stride)
        n_win = pred.shape[0]
        stride = args.pred_len
        ks = [k for k in range(args.tail_strided) if n_win - 1 - k * stride >= 0]
        rows = [n_win - 1 - k * stride for k in ks]          # ends descending from n
        # re-axis [K, L, C] -> per-channel windows [C*K, L, 1], series-outer then
        # ends descending — the windows_full multi-window ordering
        pred = pred[rows].transpose(2, 0, 1).reshape(-1, pred.shape[1], 1)
        true = true[rows].transpose(2, 0, 1).reshape(-1, true.shape[1], 1)
        inputs = inputs[rows].transpose(2, 0, 1).reshape(-1, inputs.shape[1], 1)
        print('tail_strided K={} -> windows={}'.format(len(ks), pred.shape[0]))
    elif args.tail_only:
        # final sliding position == last (seq_len+pred_len) of every item;
        # re-axis to N=n_items windows with C=1
        pred = pred[-1].transpose(1, 0)[:, :, None]    # [C, L, 1]
        true = true[-1].transpose(1, 0)[:, :, None]
        inputs = inputs[-1].transpose(1, 0)[:, :, None]
        print('tail_only -> windows={} (n_items)'.format(pred.shape[0]))

    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    print('scaler-space mse:{:.6f}, mae:{:.6f}'.format(mse, mae))
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'pred.npy'), pred)
    np.save(os.path.join(args.out, 'true.npy'), true)
    np.save(os.path.join(args.out, 'inputs.npy'), inputs)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump({'model_id': args.model_id, 'results_dir': res_dir, 'csv': args.csv,
                   'tail_only': args.tail_only, 'mse_scaled': mse, 'mae_scaled': mae}, f, indent=2)
    print('dump saved to {}'.format(args.out))


if __name__ == '__main__':
    main()
