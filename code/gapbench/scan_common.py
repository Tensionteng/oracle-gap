#!/usr/bin/env python
"""Shared scan loop for all gapbench scanners (any venv: only needs
numpy + torch + the model library). Writes results/gap_<model>_<cell>.json.
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

import sys
sys.path.insert(0, HERE)
from gap import gap_report


def cell_notes(meta, key):
    notes = []
    if meta.get('multi_window'):
        notes.append('multi_window')
    if meta.get('nan_frac', 0) > 0.05:
        notes.append('nan_heavy')
    if key == 'covid_deaths':
        notes.append('heavy_tail')
    return ','.join(notes)


def load_cells(cells_arg=''):
    manifest = json.load(open(os.path.join(HERE, 'windows_full', 'manifest.json')))
    cells = sorted(k for k, m in manifest.items() if m.get('status') == 'ok')
    if cells_arg:
        want = set(cells_arg.split(','))
        cells = [k for k in cells if k.replace('/', '_') in want]
    return manifest, cells


def scan_cells(model_name, predict, manifest, cells, reps=20):
    """predict: fn(ctx [M, L] float32 np, H int) -> median [M, H] np."""
    os.makedirs(RESULTS, exist_ok=True)
    for key in cells:
        sk = key.replace('/', '_')
        out_path = os.path.join(RESULTS, 'gap_{}_{}.json'.format(model_name, sk))
        if os.path.exists(out_path):
            try:
                if json.load(open(out_path)).get('status') == 'ok':
                    continue
            except Exception:
                pass
        meta = manifest[key]
        t0 = time.time()
        try:
            w = np.load(meta['npz'], allow_pickle=True)
            inputs, true = w['inputs'], w['true']              # [N, L, C] raw
            N, L, C = inputs.shape
            ctx = np.ascontiguousarray(inputs.transpose(0, 2, 1).reshape(N * C, L))
            pred = predict(ctx, true.shape[1]).reshape(N, C, -1).transpose(0, 2, 1)
            r = gap_report(pred.astype(np.float64), true.astype(np.float64),
                           inputs.astype(np.float64), reps=reps)
            rec = dict(model=model_name, dataset=key, status='ok',
                       n_windows=r['N'], mse=r['mse'], mae=r['mae'],
                       under_rate=r['base']['under'], over_rate=r['base']['over'],
                       wrong_rate=r['base']['wrong'], amp_base=r['base']['amp'],
                       amp_oracle=r['oracle']['amp'], gap=r['gap'],
                       ci95_lo=r['gap'] - r['gap_ci95'], ci95_hi=r['gap'] + r['gap_ci95'],
                       notes=cell_notes(meta, key), seconds=round(time.time() - t0, 2))
            print('OK  {:12s} {:34s} N={:<6d} mse={:10.4f} gap={:+.3f}+-{:.3f} {:.0f}s'.format(
                model_name, key, rec['n_windows'], rec['mse'], rec['gap'],
                r['gap_ci95'], rec['seconds']), flush=True)
        except Exception as e:
            rec = dict(model=model_name, dataset=key, status='error', reason=str(e)[:300],
                       notes=cell_notes(meta, key))
            print('ERR {:12s} {:34s} {}'.format(model_name, key, str(e)[:200]), flush=True)
        with open(out_path, 'w') as f:
            json.dump(rec, f, indent=1)


def write_load_fail(model_name, manifest, cells, reason):
    os.makedirs(RESULTS, exist_ok=True)
    for key in cells:
        sk = key.replace('/', '_')
        with open(os.path.join(RESULTS, 'gap_{}_{}.json'.format(model_name, sk)), 'w') as f:
            json.dump(dict(model=model_name, dataset=key, status='model_load_fail',
                           reason=reason[:300]), f, indent=1)
