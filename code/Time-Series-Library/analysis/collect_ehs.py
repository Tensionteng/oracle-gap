#!/usr/bin/env python
"""Collect EHS v2 confounder-controlled lookback results.

Parses the last 'mse:..., mae:...' line of every log under logs/ehs_v2/
(main matrix), logs/ehs_v2/sat/ (saturation probe) and logs/ehs_v2/tsfm/
(Chronos zero-shot), aggregates over seeds (mean+-std), and writes
logs/ehs_v2/SUMMARY.md.

Log naming:
  main: {dataset}_{model}_sl{seq_len}_s{seed}.log
  sat:  same, under sat/ (seq_len up to 5760)
  tsfm: {dataset}_Chronos_sl{seq_len}.log  (zero-shot, no seeds)
"""
import os
import re
import sys
import numpy as np

LOG_DIR = 'logs/ehs_v2'
MODELS = ['iTransformer', 'DLinear', 'PatchTST', 'TimesNet']
SEQ_LENS = [96, 336, 720, 1440, 2880]
SAT_SEQ_LENS = SEQ_LENS + [5760]
SEEDS = [2021, 2022, 2023]

METRIC_RE = re.compile(r'mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)')
MAIN_RE = re.compile(r'(.+?)_(iTransformer|DLinear|PatchTST|TimesNet)_sl(\d+)_s(\d+)\.log$')
TSFM_RE = re.compile(r'(.+?)_(Chronos)_sl(\d+)\.log$')


def last_metric(path):
    """Return (mse, mae) from the last metric line, or None."""
    val = None
    with open(path, errors='ignore') as f:
        for line in f:
            m = METRIC_RE.search(line)
            if m:
                val = (float(m.group(1)), float(m.group(2)))
    return val


def collect(directory, name_re, tags):
    """tags: ordered regex group names, e.g. ('ds','model','sl','seed')."""
    out = {}
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        m = name_re.match(fn)
        if not m:
            continue
        key = m.groupdict() if hasattr(m, 'groupdict') else None
        g = m.groups()
        entry = dict(zip(tags, g))
        entry['sl'] = int(entry['sl'])
        if 'seed' in entry:
            entry['seed'] = int(entry['seed'])
        val = last_metric(os.path.join(directory, fn))
        out[(tuple(entry[t] for t in tags))] = val
    return out


def agg(records, datasets, models, seq_lens, seeds):
    """records[(ds, model, sl, seed)] = (mse, mae)|None
    -> mean/std dicts keyed by (ds, model, sl); cell = (mean, std, n_seeds)"""
    mse = {}
    mae = {}
    missing = []
    for ds in datasets:
        for mo in models:
            for sl in seq_lens:
                vals, vals_a = [], []
                for s in seeds:
                    v = records.get((ds, mo, sl, s))
                    if v is None:
                        missing.append((ds, mo, sl, s))
                        continue
                    vals.append(v[0])
                    vals_a.append(v[1])
                if vals:
                    mse[(ds, mo, sl)] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
                    mae[(ds, mo, sl)] = (float(np.mean(vals_a)), float(np.std(vals_a)), len(vals_a))
    return mse, mae, missing


def fmt(cell, bold=False):
    if cell is None:
        return '-'
    s = f'{cell[0]:.4f}±{cell[1]:.4f}'
    if cell[2] < 3:
        s += f'[n={cell[2]}]'
    return f'**{s}**' if bold else s


def model_table(metric, datasets, model, seq_lens, title):
    lines = [f'### {title}', '',
             '| dataset | ' + ' | '.join(f'sl={s}' for s in seq_lens) + ' | best sl |',
             '|---|' + '---|' * (len(seq_lens) + 1)]
    best_rows = {}
    for ds in datasets:
        cells = [metric.get((ds, model, sl)) for sl in seq_lens]
        valid = [(i, c[0]) for i, c in enumerate(cells) if c is not None]
        if not valid:
            continue
        best_i = min(valid, key=lambda x: x[1])[0]
        best_rows[ds] = seq_lens[best_i]
        row = '| ' + ds + ' | ' + ' | '.join(
            fmt(c, bold=(i == best_i)) for i, c in enumerate(cells)) + f' | {seq_lens[best_i]} |'
        lines.append(row)
    return '\n'.join(lines), best_rows


def main():
    main_rec = collect(LOG_DIR, MAIN_RE, ('ds', 'model', 'sl', 'seed'))
    sat_rec = collect(os.path.join(LOG_DIR, 'sat'), MAIN_RE, ('ds', 'model', 'sl', 'seed'))
    tsfm_rec = collect(os.path.join(LOG_DIR, 'tsfm'), TSFM_RE, ('ds', 'model', 'sl'))

    datasets = sorted({k[0] for k in main_rec})
    mse, mae, missing = agg(main_rec, datasets, MODELS, SEQ_LENS, SEEDS)

    parts = ['# EHS v2: confounder-controlled lookback (all arms use --max_train_windows = N_min of the sl=2880 arm)',
             '',
             'Cells: mean±std test MSE over seeds {2021,2022,2023}; **bold** = best seq_len of the row.',
             '`[n=k]` marks cells computed from fewer than 3 finished seeds (time-boxed cut).',
             '']

    best_all = {}
    for mo in MODELS:
        tbl, best_rows = model_table(mse, datasets, mo, SEQ_LENS, f'Model: {mo} (MSE)')
        parts.append(tbl)
        parts.append('')
        best_all[mo] = best_rows

    # best-seq_len summary (dataset x model)
    parts += ['### Best seq_len summary (by mean test MSE)', '',
              '| dataset | ' + ' | '.join(MODELS) + ' |',
              '|---|' + '---|' * len(MODELS)]
    for ds in datasets:
        parts.append('| ' + ds + ' | ' + ' | '.join(
            str(best_all[mo].get(ds, '-')) for mo in MODELS) + ' |')
    parts.append('')

    # MAE tables folded in a compact way
    for mo in MODELS:
        tbl, _ = model_table(mae, datasets, mo, SEQ_LENS, f'Model: {mo} (MAE)')
        parts.append(tbl)
        parts.append('')

    # saturation probe
    if sat_rec:
        sat_datasets = sorted({k[0] for k in sat_rec})
        sat_models = ['iTransformer', 'DLinear']
        sat_mse, _, sat_missing = agg(sat_rec, sat_datasets, sat_models, SAT_SEQ_LENS, SEEDS)
        parts += ['## Saturation probe (electricity/traffic, N_min recomputed at sl=5760; MSE)', '']
        for mo in sat_models:
            tbl, _ = model_table(sat_mse, sat_datasets, mo, SAT_SEQ_LENS, f'Model: {mo} (MSE, N_min@5760)')
            parts.append(tbl)
            parts.append('')
    else:
        sat_missing = []

    # TSFM zero-shot
    tsfm_datasets_all = sorted({k[0] for k in main_rec})
    if tsfm_rec:
        tsfm_datasets = sorted({k[0] for k in tsfm_rec})
        parts += ['## TSFM zero-shot (Chronos bolt-base, single run; MSE / MAE)', '',
                  '| dataset | ' + ' | '.join(f'sl={s}' for s in SEQ_LENS) + ' |',
                  '|---|' + '---|' * len(SEQ_LENS)]
        for ds in tsfm_datasets:
            cells = []
            for sl in SEQ_LENS:
                v = tsfm_rec.get((ds, 'Chronos', sl))
                cells.append('-' if v is None else f'{v[0]:.4f}/{v[1]:.4f}')
            parts.append('| ' + ds + ' | ' + ' | '.join(cells) + ' |')
        parts.append('')

    # missing arms with reasons
    def reason(logpath):
        if not os.path.exists(logpath):
            return 'timeout-cut (never started)'
        txt = open(logpath, errors='ignore').read()
        if 'Traceback' in txt:
            m = re.findall(r'(\w+(?:Error|Exception))\b', txt)
            return f'failed ({m[-1] if m else "see log"})'
        return 'timeout-cut (killed mid-run)'

    rows = []
    for ds, mo, sl, s in missing:
        rows.append((f'main: {ds} {mo} sl={sl} seed={s}',
                     reason(os.path.join(LOG_DIR, f'{ds}_{mo}_sl{sl}_s{s}.log'))))
    for ds, mo, sl, s in sat_missing:
        rows.append((f'sat: {ds} {mo} sl={sl} seed={s}',
                     reason(os.path.join(LOG_DIR, 'sat', f'{ds}_{mo}_sl{sl}_s{s}.log'))))
    tsfm_missing = []
    for ds in tsfm_datasets_all:
        for sl in SEQ_LENS:
            if tsfm_rec.get((ds, 'Chronos', sl)) is None:
                tsfm_missing.append((ds, sl))
                rows.append((f'tsfm: {ds} Chronos sl={sl}',
                             reason(os.path.join(LOG_DIR, 'tsfm', f'{ds}_Chronos_sl{sl}.log'))))
    if rows:
        parts += ['## Missing/failed arms', '']
        for label, why in rows:
            parts.append(f'- {label} — {why}')
        parts.append('')

    out = os.path.join(LOG_DIR, 'SUMMARY.md')
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    print(f'wrote {out}')
    print(f'main arms with results: {sum(1 for v in main_rec.values() if v)} / {len(main_rec)} logs')
    print(f'sat arms with results:  {sum(1 for v in sat_rec.values() if v)} / {len(sat_rec)} logs')
    print(f'tsfm arms with results: {sum(1 for v in tsfm_rec.values() if v)} / {len(tsfm_rec)} logs')
    if missing:
        print(f'missing main arms: {len(missing)}')
        for x in missing[:20]:
            print('  ', x)


if __name__ == '__main__':
    main()
