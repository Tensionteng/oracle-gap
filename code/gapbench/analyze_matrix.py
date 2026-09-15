#!/usr/bin/env python
"""Four-slice analysis over results/matrix.csv (see task book):
(a) gap vs parameter ladder (bolt 9/21/48/205M, TimeMoE 50/200M, moirai
    14/311M + moirai2): per-model mean/median gap and meat fraction;
(b) dataset ranking by cross-model mean gap;
(c) TSFM vs PatchTST on the task-model-arm cells;
(d) failure cells.
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

PARAMS_M = {  # approximate parameter counts (M), for ladder sorting
    'bolt_tiny': 8, 'bolt_mini': 20, 'bolt_small': 48, 'bolt_base': 205,
    'chronos2': 120, 'sundial': 128, 'moirai11_small': 14, 'moirai2_small': 15,
    'moirai11_large': 311, 'timemoe_50m': 50, 'timemoe_200m': 200,
    'timer_84m': 84, 'tirex': 35,
}
MEAT = 0.02


def load():
    rows = []
    with open(os.path.join(HERE, 'results', 'matrix.csv')) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    ok = [r for r in rows if r['status'] == 'ok']
    for r in ok:
        r['gap_f'] = float(r['gap'])
        r['mse_f'] = float(r['mse'])
    return rows, ok


def main():
    rows, ok = load()
    tsfm = [r for r in ok if r['model'] != 'patchtst']

    print('=' * 70)
    print('(a) gap vs parameter scale (all %d TSFM cells)' % len(tsfm))
    print('%-14s %6s %8s %8s %8s %9s %9s' % ('model', 'M', 'mean', 'median', 'frac>+.02', 'frac<-0.02', 'n_cells'))
    by_model = defaultdict(list)
    for r in tsfm:
        by_model[r['model']].append(r['gap_f'])
    for m, gs in sorted(by_model.items(), key=lambda kv: PARAMS_M.get(kv[0], 999)):
        gs = np.array(gs)
        print('%-14s %6d %+8.4f %+8.4f %9.2f %9.2f %9d' % (
            m, PARAMS_M.get(m, -1), gs.mean(), np.median(gs),
            (gs > MEAT).mean(), (gs < -MEAT).mean(), len(gs)))

    print()
    print('(b) dataset ranking by mean gap across TSFM models (top/bottom 12)')
    by_ds = defaultdict(list)
    for r in tsfm:
        by_ds[r['dataset']].append(r['gap_f'])
    rank = sorted(by_ds.items(), key=lambda kv: -np.mean(kv[1]))
    for d, gs in rank[:12]:
        print('  MEAT  %-34s mean %+.3f  (min %+.3f max %+.3f, k=%d)' % (
            d, np.mean(gs), np.min(gs), np.max(gs), len(gs)))
    print('  ...')
    for d, gs in rank[-12:]:
        print('  none  %-34s mean %+.3f  (min %+.3f max %+.3f, k=%d)' % (
            d, np.mean(gs), np.min(gs), np.max(gs), len(gs)))

    print()
    print('(c) TSFM vs PatchTST on task-model-arm cells (tail protocol)')
    pt = {r['dataset']: r for r in ok if r['model'] == 'patchtst'}
    print('%-24s %9s | %-14s %9s | %-14s %9s' % ('dataset', 'ptst_gap', 'best_tsfm', 'its_gap', 'median_tsfm', 'med_gap'))
    for d in sorted(pt):
        gs = [(r['model'], r['gap_f']) for r in tsfm if r['dataset'] == d]
        if not gs:
            continue
        gaps = [g for _, g in gs]
        best = max(gs, key=lambda x: x[1])
        med = np.median(gaps)
        print('%-24s %+9.3f | %-14s %+9.3f | %-14s %+9.3f  (k=%d)' % (
            d, float(pt[d]['gap']), best[0], best[1], '', med, len(gs)))

    print()
    print('(d) failed cells')
    bad = [r for r in rows if r['status'] != 'ok']
    if not bad:
        print('  none')
    for r in bad:
        print('  %-14s %-34s %s: %s' % (r['model'], r['dataset'], r['status'], r.get('reason', '')[:100]))


if __name__ == '__main__':
    main()
