#!/usr/bin/env python
"""GIFT-Eval arrow -> unified window specs for the oracle-gap benchmark pilot.

Two window modes over each univariate item collection:
  tail        : the last (seq_len + pred_len) steps of every item, one window
                per item (GIFT-style tail evaluation). Items shorter than
                seq_len + pred_len, or with NaN / near-constant input in the
                window, are skipped.
  tslib_test  : windows at exactly the positions TSLib's Dataset_Custom test
                split produces on the wide table (7/1/2 time split, sliding
                96->96 over the last 20%), so a TSFM can be scored on the very
                windows a TSLib task model is tested on. Requires equal-length
                items.

Outputs (raw scale, per-window space harmonization happens in gap.py):
  npz {inputs [N, seq_len, C], true [N, pred_len, C], item_ids [N]}
  tail mode keeps C=1 per window (N = n_items); tslib_test stacks all items as
  channels of one multivariate window (N = n_window_positions, C = n_items).

Also exports the wide CSV (date + one column per item, last column renamed OT)
that TSLib Dataset_Custom consumes.
"""
import argparse
import os

import numpy as np

GIFT_ROOT = '" + os.environ.get("GIFT_ROOT", "data/gifteval") + "'
MIN_INPUT_STD = 1e-6  # below this the per-window normalization is degenerate


def load_collection(dataset, sub=''):
    """Return (item_ids, list of 1-D float64 arrays, freq). Univariate only."""
    import datasets
    path = os.path.join(GIFT_ROOT, dataset, sub) if sub else os.path.join(GIFT_ROOT, dataset)
    files = sorted(f for f in os.listdir(path) if f.endswith('.arrow'))
    if not files:
        raise FileNotFoundError('no arrow file under {}'.format(path))
    d = datasets.Dataset.from_file(os.path.join(path, files[0]))
    item_ids, series = [], []
    freq = d[0]['freq']
    for i in range(len(d)):
        r = d[i]
        t = np.asarray(r['target'])
        if t.ndim != 1:
            raise ValueError('{} item {} target is multivariate {}'.format(dataset, i, t.shape))
        item_ids.append(str(r['item_id']))
        series.append(t.astype(np.float64))
    return item_ids, series, freq


def build_tail(series, item_ids, seq_len, pred_len):
    inputs, trues, kept = [], [], []
    need = seq_len + pred_len
    n_skip = 0
    for iid, s in zip(item_ids, series):
        if len(s) < need:
            n_skip += 1
            continue
        x = s[-need:-pred_len]
        y = s[-pred_len:]
        if np.isnan(x).any() or np.isnan(y).any() or x.std() < MIN_INPUT_STD:
            n_skip += 1
            continue
        inputs.append(x[:, None].astype(np.float32))
        trues.append(y[:, None].astype(np.float32))
        kept.append(iid)
    print('tail windows: kept {}/{} items (skipped {} too short / NaN / constant)'.format(
        len(kept), len(series), n_skip))
    return np.stack(inputs), np.stack(trues), np.array(kept)


def tslib_borders(n, seq_len):
    """TSLib Dataset_Custom split borders."""
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_vali = n - num_train - num_test
    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, n]
    return border1s, border2s, num_train


def build_tslib_test(series, item_ids, seq_len, pred_len):
    lens = {len(s) for s in series}
    if len(lens) != 1:
        raise ValueError('tslib_test mode needs equal-length items, got {}'.format(sorted(lens)[:5]))
    n = lens.pop()
    if any(np.isnan(s).any() for s in series):
        raise ValueError('tslib_test mode: NaN present (TSLib wide table cannot carry NaN)')
    data = np.stack(series, axis=1)                      # [n, n_items]
    border1s, border2s, _ = tslib_borders(n, seq_len)
    b1, b2 = border1s[2], border2s[2]
    n_win = (b2 - b1) - seq_len - pred_len + 1
    if n_win <= 0:
        raise ValueError('test region {} steps < seq_len+pred_len {}'.format(b2 - b1, seq_len + pred_len))
    inputs = np.stack([data[b1 + i:b1 + i + seq_len] for i in range(n_win)]).astype(np.float32)
    trues = np.stack([data[b1 + i + seq_len:b1 + i + seq_len + pred_len] for i in range(n_win)]).astype(np.float32)
    # one id per (window position) x item channel, for traceability
    ids = np.array(['w{}:{}'.format(i, iid) for i in range(n_win) for iid in item_ids])
    print('tslib_test windows: {} positions x {} channels (borders {}..{} of n={})'.format(
        n_win, len(item_ids), b1, b2, n))
    return inputs, trues, ids


def export_wide_csv(series, item_ids, freq, out_csv):
    import pandas as pd
    lens = {len(s) for s in series}
    if len(lens) != 1:
        raise ValueError('wide CSV export needs equal-length items')
    n = lens.pop()
    # synthetic regular time axis so TSLib timeF features are well defined;
    # normalize GIFT freq strings to pandas>=2 conventions (H->h, T->min)
    pf = freq.lower().replace('t', 'min')
    dates = pd.date_range('2000-01-01', periods=n, freq=pf)
    cols = {iid: s for iid, s in zip(item_ids, series)}
    df = pd.DataFrame(cols)
    # Dataset_Custom requires a 'target' column named OT: rename the last item
    df = df.rename(columns={item_ids[-1]: 'OT'})
    df.insert(0, 'date', dates)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print('wide CSV -> {}  ({} rows x {} channels + date)'.format(out_csv, n, len(item_ids)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--sub', default='')
    ap.add_argument('--mode', choices=['tail', 'tslib_test'], required=True)
    ap.add_argument('--seq_len', type=int, default=96)
    ap.add_argument('--pred_len', type=int, default=96)
    ap.add_argument('--out', required=True)
    ap.add_argument('--export_csv', default='')
    args = ap.parse_args()

    item_ids, series, freq = load_collection(args.dataset, args.sub)
    print('collection {} {}: {} items, freq {}'.format(args.dataset, args.sub, len(series), freq))
    if args.mode == 'tail':
        inputs, trues, ids = build_tail(series, item_ids, args.seq_len, args.pred_len)
    else:
        inputs, trues, ids = build_tslib_test(series, item_ids, args.seq_len, args.pred_len)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, inputs=inputs, true=trues, item_ids=ids)
    print('saved {}: inputs {} true {}'.format(args.out, inputs.shape, trues.shape))
    if args.export_csv:
        export_wide_csv(series, item_ids, freq, args.export_csv)


if __name__ == '__main__':
    main()
