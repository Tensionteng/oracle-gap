#!/usr/bin/env python
"""PatchTST task-model arm for the full scan: export wide CSVs for eligible
datasets, then (via the driver shell script) train PatchTST in TSLib under the
custom 7/1/2 protocol and collect tail-window dumps.

Eligibility (pilot rules): equal-length series, median length >= 960 (so the
custom test region int(0.2n) >= 192 fits one 96->96 window), total channels in
the low hundreds, and NaN fraction small enough that whole-series linear
interpolation + ffill/bfill is a cosmetic fix (datasets with structured
missing blocks like electricity are excluded up front).

Whole-series NaN are interpolated before writing the CSV (TSLib cannot carry
NaN); a series still containing NaN afterwards is dropped from the table and
counted in the manifest note.
"""
import argparse
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windows_full import load_collection

HERE = os.path.dirname(os.path.abspath(__file__))

# dataset key -> (pd freq for the synthetic date axis, TSLib --freq flag)
ARM_DATASETS = {
    'solar/H': ('h', 'h'),
    'SZ_TAXI/15T': ('15min', 't'),
    'LOOP_SEATTLE/H': ('h', 'h'),
    'M_DENSE/H': ('h', 'h'),
    'ett1/H': ('h', 'h'),
    'ett2/H': ('h', 'h'),
    'jena_weather/H': ('h', 'h'),
    'bizitobs_l2c/H': ('h', 'h'),
    'hierarchical_sales/D': ('d', 'd'),
    # second batch (added after the first pass): same eligibility rules
    'LOOP_SEATTLE/5T': ('5min', 't'),
    'solar/10T': ('10min', 't'),
    'ett1/15T': ('15min', 't'),
    'ett2/15T': ('15min', 't'),
    'jena_weather/10T': ('10min', 't'),
    'bizitobs_l2c/5T': ('5min', 't'),
}


def impute_series(s):
    if not np.isnan(s).any():
        return s
    idx = np.arange(len(s))
    ok = ~np.isnan(s)
    if ok.sum() < 2:
        return None
    return np.interp(idx, idx[ok], s[ok])


def export(key, pd_freq, out_dir):
    import pandas as pd
    dataset, _, sub = key.partition('/')
    ids, series, freq = load_collection(dataset, sub)
    lens = {len(s) for s in series}
    if len(lens) != 1:
        raise ValueError('{} not equal-length: {}'.format(key, sorted(lens)[:5]))
    cols, dropped = {}, 0
    for iid, s in zip(ids, series):
        si = impute_series(s)
        if si is None or np.isnan(si).any():
            dropped += 1
            continue
        cols[iid] = si
    n = lens.pop()
    dates = pd.date_range('2000-01-01', periods=n, freq=pd_freq)
    df = pd.DataFrame(cols)
    last = list(cols)[-1]
    df = df.rename(columns={last: 'OT'})
    df.insert(0, 'date', dates)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, key.replace('/', '_') + '.csv')
    df.to_csv(out, index=False)
    print('{}: {} rows x {} channels (dropped {}) -> {}'.format(key, n, len(cols), dropped, out))
    return out, len(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'csv'))
    ap.add_argument('--keys', default=','.join(ARM_DATASETS.keys()))
    args = ap.parse_args()
    manifest = {}
    for key in args.keys.split(','):
        pd_freq, tslib_freq = ARM_DATASETS[key]
        try:
            csv_path, n_ch = export(key, pd_freq, args.out_dir)
            manifest[key] = dict(csv=csv_path, n_channels=n_ch, tslib_freq=tslib_freq,
                                 status='ok')
        except Exception as e:
            manifest[key] = dict(status='error', reason=str(e)[:200])
            print('ERR {} {}'.format(key, str(e)[:150]))
    json.dump(manifest, open(os.path.join(HERE, 'patchtst_arm_manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
