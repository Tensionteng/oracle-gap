#!/usr/bin/env python
"""Build a unified sliding-window dump for a TSLib PatchTST run on ETTh1/ETTm1.

Same idea as collect_tslib_dump.py but replicates Dataset_ETT_hour/minute's
fixed borders + StandardScaler (fit on train rows only) instead of the
Dataset_Custom 7/1/2 split. Writes <out>/{pred,true,inputs}.npy in scaler
space, N = full sliding test set (2785 for ETTh1, 11425 for ETTm1).
"""
import argparse
import glob
import json
import os

import numpy as np

TSLIB = '/mnt/jd/users/tengshiyuan.1/codes/mtp4ts/Time-Series-Library'


def build_test_inputs(csv_path, freq, seq_len, pred_len):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(csv_path)
    df_data = df[df.columns[1:]]
    u = 24 if freq == 'h' else 24 * 4
    border1s = [0, 12 * 30 * u - seq_len, 12 * 30 * u + 4 * 30 * u - seq_len]
    border2s = [12 * 30 * u, 12 * 30 * u + 4 * 30 * u, 12 * 30 * u + 8 * 30 * u]
    scaler = StandardScaler()
    scaler.fit(df_data.values[border1s[0]:border2s[0]])
    data = scaler.transform(df_data.values)
    b1, b2 = border1s[2], border2s[2]
    n_win = (b2 - b1) - seq_len - pred_len + 1
    inputs = np.stack([data[b1 + i:b1 + i + seq_len] for i in range(n_win)])
    return inputs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_id', required=True)
    ap.add_argument('--data', required=True, choices=['ETTh1', 'ETTm1'])
    ap.add_argument('--seq_len', type=int, default=96)
    ap.add_argument('--pred_len', type=int, default=96)
    ap.add_argument('--out', required=True)
    ap.add_argument('--results_root', default=os.path.join(TSLIB, 'results'))
    args = ap.parse_args()

    hits = sorted(glob.glob(os.path.join(
        args.results_root,
        'long_term_forecast_{}_PatchTST_{}_*'.format(args.model_id, args.data))))
    if not hits:
        raise FileNotFoundError('no TSLib results for model_id {}'.format(args.model_id))
    res_dir = hits[-1]
    print('results dir: {}'.format(res_dir))
    pred = np.load(os.path.join(res_dir, 'pred.npy'))
    true = np.load(os.path.join(res_dir, 'true.npy'))
    csv_path = os.path.join(TSLIB, 'dataset', 'ETT-small', args.data + '.csv')
    freq = 'h' if args.data.endswith('h1') else 'm'
    inputs = build_test_inputs(csv_path, freq, args.seq_len, args.pred_len)
    assert inputs.shape[0] == pred.shape[0], (inputs.shape, pred.shape)
    print('windows={} pred {} true {} inputs {}'.format(pred.shape[0], pred.shape, true.shape, inputs.shape))

    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    print('scaler-space mse:{:.6f}, mae:{:.6f}'.format(mse, mae))
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'pred.npy'), pred)
    np.save(os.path.join(args.out, 'true.npy'), true)
    np.save(os.path.join(args.out, 'inputs.npy'), inputs)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump({'model_id': args.model_id, 'results_dir': res_dir, 'csv': csv_path,
                   'protocol': 'sliding', 'mse_scaled': mse, 'mae_scaled': mae}, f, indent=2)
    print('dump saved to {}'.format(args.out))


if __name__ == '__main__':
    main()
