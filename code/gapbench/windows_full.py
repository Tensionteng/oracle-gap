#!/usr/bin/env python
"""Build tail-protocol windows for the full GIFT-Eval scan.

Rules (settled in the pilot):
  - one (dataset, sub) arrow shard set at a time; only data-*.arrow files count
    (HF cache-*.arrow artifacts are ignored); multi-shard dirs are concatenated;
  - multivariate items (target ndim=2, shape [C, L]) are flattened to C
    univariate series named '<item_id>:ch<c>'; a single-item multivariate
    dataset (jena_weather, ett1/2, bizitobs_*) thus becomes one row with C>1;
  - tail window = last seq_len + pred_len steps of each series; series shorter
    than that are skipped;
  - collections with < MIN_SERIES_FOR_SINGLE surviving series switch to strided
    multi-window tails (stride=pred_len, up to MAX_WIN_PER_ITEM per series) so
    the window-level residual permutation has something to permute;
  - NaN policy: linear-interpolate inside the window, then ffill/bfill; windows
    still containing NaN, or with input std < 1e-6, are dropped;
  - a (dataset, sub) cell is kept when at least MIN_WINDOWS windows survive.

Writes windows_full/<safe_key>_tail.npz {inputs, true, item_ids} and appends a
row to windows_full/manifest.json.
"""
import argparse
import glob
import json
import os

import numpy as np

GIFT_ROOT = '" + os.environ.get("GIFT_ROOT", "data/gifteval") + "'
MIN_INPUT_STD = 1e-6
MIN_SERIES_FOR_SINGLE = 30
MAX_WIN_PER_ITEM = 20
MIN_WINDOWS = 5


def safe_key(dataset, sub):
    return dataset + ('_' + sub if sub else '')


def load_collection(dataset, sub=''):
    """Return (series_ids, list of 1-D arrays, freq). Flattens multivariate items."""
    import datasets
    path = os.path.join(GIFT_ROOT, dataset, sub) if sub else os.path.join(GIFT_ROOT, dataset)
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(path, 'data-*.arrow')))
    if not files:
        raise FileNotFoundError('no data-*.arrow under {}'.format(path))
    ids, series = [], []
    freq = None
    for fn in files:
        d = datasets.Dataset.from_file(os.path.join(path, fn))
        for i in range(len(d)):
            r = d[i]
            t = np.asarray(r['target'], dtype=np.float64)
            freq = r['freq']
            if t.ndim == 1:
                ids.append(str(r['item_id']))
                series.append(t)
            elif t.ndim == 2:
                for c in range(t.shape[0]):
                    ids.append('{}:ch{}'.format(r['item_id'], c))
                    series.append(t[c])
            else:
                raise ValueError('{} target ndim={} unsupported'.format(fn, t.ndim))
    return ids, series, freq


def impute_window(x):
    """Linear-interpolate NaN, then ffill/bfill. Returns None if unrecoverable."""
    if not np.isnan(x).any():
        return x
    n = len(x)
    idx = np.arange(n)
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return None
    xi = np.interp(idx, idx[ok], x[ok])
    return xi


def windows_for_series(s, seq_len, pred_len, multi):
    """Tail windows for one series; multi=True gives strided multi-windows."""
    need = seq_len + pred_len
    if len(s) < need:
        return []
    ends = [len(s)] if not multi else \
        list(range(len(s), max(len(s) - MAX_WIN_PER_ITEM * pred_len, need - 1), -pred_len))
    out = []
    for e in ends:
        x = impute_window(s[e - need:e - pred_len])
        y = impute_window(s[e - pred_len:e])
        if x is None or y is None or x.std() < MIN_INPUT_STD or np.isnan(y).any():
            continue
        out.append((x.astype(np.float32), y.astype(np.float32)))
    return out


def build(dataset, sub, seq_len=96, pred_len=96):
    ids, series, freq = load_collection(dataset, sub)
    nan_frac = float(np.mean([np.isnan(s).mean() for s in series[:20]])) if series else 0.0
    # first pass with single-window rule to decide multi-window mode
    single_ok = sum(1 for s in series if len(s) >= seq_len + pred_len)
    multi = single_ok < MIN_SERIES_FOR_SINGLE
    inputs, trues, kept = [], [], []
    for iid, s in zip(ids, series):
        for x, y in windows_for_series(s, seq_len, pred_len, multi):
            inputs.append(x[:, None])
            trues.append(y[:, None])
            kept.append(iid)
    if not kept:
        return (np.zeros((0, seq_len, 1), np.float32), np.zeros((0, pred_len, 1), np.float32),
                np.array([]), dict(freq=freq, n_series=len(series), multi_window=multi,
                                    nan_frac=round(nan_frac, 4)))
    return (np.stack(inputs), np.stack(trues), np.array(kept),
            dict(freq=freq, n_series=len(series), multi_window=multi,
                 nan_frac=round(nan_frac, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='windows_full')
    ap.add_argument('--seq_len', type=int, default=96)
    ap.add_argument('--pred_len', type=int, default=96)
    ap.add_argument('--only', default='', help='comma list of safe keys to build')
    args = ap.parse_args()

    inv = json.load(open('inventory.json'))
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    only = set(args.only.split(',')) if args.only else None

    for key in sorted(inv):
        v = inv[key]
        if 'error' in v:
            manifest[key] = dict(status='skip', reason='inventory error: ' + v['error'][:80])
            continue
        dataset, _, sub = key.partition('/')
        sk = safe_key(dataset, sub)
        if only and sk not in only:
            continue
        if sk in manifest and manifest[sk].get('status') == 'ok':
            continue
        try:
            inputs, trues, kept, meta = build(dataset, sub, args.seq_len, args.pred_len)
            if len(kept) < MIN_WINDOWS:
                manifest[key] = dict(status='skip',
                                     reason='only {} usable windows (min/med len {}/{})'.format(
                                         len(kept), v['len_min'], v['len_med']))
                print('SKIP {:34s} usable windows {}'.format(key, len(kept)))
                continue
            out = os.path.join(args.out_dir, sk + '_tail.npz')
            np.savez(out, inputs=inputs, true=trues, item_ids=kept)
            meta.update(status='ok', npz=out, n_windows=int(len(kept)),
                        n_items=int(v['n_items']), len_med=int(v['len_med']))
            manifest[key] = meta
            print('OK   {:34s} N={:<6d} multi={} nan={:.3f}'.format(
                key, len(kept), meta['multi_window'], meta['nan_frac']))
        except Exception as e:
            manifest[key] = dict(status='error', reason=str(e)[:200])
            print('ERR  {:34s} {}'.format(key, str(e)[:120]))
        json.dump(manifest, open(manifest_path, 'w'), indent=1)
    json.dump(manifest, open(manifest_path, 'w'), indent=1)
    n_ok = sum(1 for m in manifest.values() if m.get('status') == 'ok')
    print('manifest: {} ok / {} total'.format(n_ok, len(manifest)))


if __name__ == '__main__':
    main()
