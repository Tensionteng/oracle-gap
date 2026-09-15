#!/usr/bin/env python
"""Assemble results/gap_*.json into results/matrix.csv.

Columns: model, dataset, n_windows, mse, under_rate, amp_base, amp_oracle,
gap, ci95_lo, ci95_hi, notes. Adds the PatchTST task-model arm: full-scan arm
dumps (dumps_full/<key>_tail/patchtst, gap.json) and the two pilot arm dumps
(dumps/{solarH,sztaxi15T}_tail/patchtst), keyed back to manifest dataset names.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

COLS = ['model', 'dataset', 'n_windows', 'mse', 'under_rate', 'amp_base',
        'amp_oracle', 'gap', 'ci95_lo', 'ci95_hi', 'notes', 'status', 'reason']


def gapjson_to_row(model, dataset, r):
    return dict(model=model, dataset=dataset, status='ok', n_windows=r['N'],
                mse=r['mse'], under_rate=r['base']['under'], amp_base=r['base']['amp'],
                amp_oracle=r['oracle']['amp'], gap=r['gap'],
                ci95_lo=r['gap'] - r['gap_ci95'], ci95_hi=r['gap'] + r['gap_ci95'],
                notes='task_model_arm', reason='')


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, 'gap_*.json'))):
        r = json.load(open(f))
        if r.get('status') != 'ok':
            rows.append(dict(model=r.get('model'), dataset=r.get('dataset'),
                             status=r.get('status', 'error'), reason=r.get('reason', ''),
                             notes=r.get('notes', '')))
            continue
        rows.append(dict(model=r['model'], dataset=r['dataset'], status='ok',
                         n_windows=r['n_windows'], mse=r['mse'],
                         under_rate=r['under_rate'], amp_base=r['amp_base'],
                         amp_oracle=r['amp_oracle'], gap=r['gap'],
                         ci95_lo=r['ci95_lo'], ci95_hi=r['ci95_hi'],
                         notes=r.get('notes', ''), reason=''))

    # PatchTST arm: full-scan dumps (map safe key -> manifest key exactly)
    manifest = json.load(open(os.path.join(HERE, 'windows_full', 'manifest.json')))
    safe2key = {k.replace('/', '_'): k for k in manifest}
    for f in sorted(glob.glob(os.path.join(HERE, 'dumps_full', '*_tail', 'patchtst', 'gap.json'))):
        sk = f.split(os.sep)[-3][:-5]                        # solar_H_tail -> solar_H
        key = safe2key.get(sk, sk)
        rows.append(gapjson_to_row('patchtst', key, json.load(open(f))))
    # pilot arm dumps (solar/H, SZ_TAXI/15T) if not already present
    pilot = {'solarH_tail': 'solar/H', 'sztaxi15T_tail': 'SZ_TAXI/15T'}
    have = {(r['model'], r['dataset']) for r in rows}
    for d, key in pilot.items():
        if ('patchtst', key) in have:
            continue
        f = os.path.join(HERE, 'dumps', d, 'patchtst', 'gap.json')
        if os.path.exists(f):
            rows.append(gapjson_to_row('patchtst', key, json.load(open(f))))

    import csv
    out = os.path.join(RESULTS, 'matrix.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in COLS})
    n_ok = sum(1 for r in rows if r['status'] == 'ok')
    n_bad = len(rows) - n_ok
    print('matrix.csv: {} rows ({} ok, {} failed)'.format(len(rows), n_ok, n_bad))


if __name__ == '__main__':
    main()
