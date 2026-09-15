#!/usr/bin/env python
"""Unified oracle-gap computation for gapbench dumps.

Same oracle construction as TSLib analysis/oracle_diagnostic.py (window-level
bootstrap permutation of residuals; anchor = context mean), but consumes the
unified dump {pred,true,inputs}.npy directly instead of rebuilding contexts
from an external CSV, and adds reps/CI reporting.

Scale decision: multi-series collections mix wildly different magnitudes, so
before any pooling every window triple (inputs, pred, true) is divided by its
own per-channel input-window std (instance normalization). This makes the
cross-window residual permutation meaningful and both model families
comparable (Chronos-Bolt dumps are raw-scale, PatchTST dumps scaler-space);
the ctxmean anchor absorbs the level. MSE is reported in this normalized
space. Windows whose input std is < 1e-6 are dropped (degenerate).
"""
import argparse
import json
import os

import numpy as np

MIN_INPUT_STD = 1e-6
EPS = 1e-8


def metrics(pred, true, anchor):
    p = pred - anchor
    d = true - anchor
    same = p * d > 0
    under = same & (np.abs(p) < np.abs(d))
    over = same & (np.abs(p) > np.abs(d))
    wrong = ~same
    amp = pred.std() / (true.std() + EPS)
    return dict(under=float(under.mean()), over=float(over.mean()),
                wrong=float(wrong.mean()), amp=float(amp))


def gap_report(pred, true, inputs, reps=20, seed=0):
    s = inputs.std(axis=1, keepdims=True)                    # [N, 1, C] per-channel
    keep = (s.reshape(s.shape[0], -1).min(axis=1)) >= MIN_INPUT_STD
    if not keep.all():
        print('dropped {}/{} windows with degenerate input std'.format((~keep).sum(), len(keep)))
    pred, true, inputs, s = pred[keep], true[keep], inputs[keep], s[keep]
    pred_n = pred / (s + EPS)
    true_n = true / (s + EPS)
    anchor = (inputs / (s + EPS)).mean(axis=1, keepdims=True)

    mse = float(np.mean((pred_n - true_n) ** 2))
    mae = float(np.mean(np.abs(pred_n - true_n)))
    base_m = metrics(pred_n, true_n, anchor)
    resid = true_n - pred_n
    rng = np.random.default_rng(seed)
    oracle_ms = []
    for _ in range(reps):
        perm = rng.permutation(len(pred_n))
        oracle_ms.append(metrics(pred_n, pred_n + resid[perm], anchor))
    om = {k: float(np.mean([m[k] for m in oracle_ms])) for k in oracle_ms[0]}
    amp_reps = np.array([m['amp'] for m in oracle_ms])
    amp_std = float(amp_reps.std())
    amp_ci = float(1.96 * amp_std / np.sqrt(reps))
    gap = om['amp'] - base_m['amp']
    return dict(N=int(len(pred_n)), mse=mse, mae=mae, base=base_m, oracle=om,
                oracle_amp_std=amp_std, oracle_amp_ci95=amp_ci,
                gap=gap, gap_ci95=amp_ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--json_out', default='')
    args = ap.parse_args()

    pred = np.load(os.path.join(args.dump, 'pred.npy'))
    true = np.load(os.path.join(args.dump, 'true.npy'))
    inputs = np.load(os.path.join(args.dump, 'inputs.npy'))
    r = gap_report(pred, true, inputs, reps=args.reps, seed=args.seed)

    print('== {} (N={}, pred std(inst-norm)={:.4f}, true std={:.4f}) =='.format(
        args.tag or args.dump, r['N'], (pred / (inputs.std(axis=1, keepdims=True) + EPS)).std(),
        (true / (inputs.std(axis=1, keepdims=True) + EPS)).std()))
    print('normed MSE {:.4f}  MAE {:.4f}'.format(r['mse'], r['mae']))
    print('{:8}{:>9}{:>9}{:>9}{:>9}'.format('', 'under', 'over', 'wrong', 'amp'))
    print('{:8}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}'.format(
        'base', r['base']['under'], r['base']['over'], r['base']['wrong'], r['base']['amp']))
    print('{:8}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}   (+-{:.3f} CI95)'.format(
        'oracle', r['oracle']['under'], r['oracle']['over'], r['oracle']['wrong'],
        r['oracle']['amp'], r['oracle_amp_ci95']))
    print('\namp gap(oracle-base) = {:+.3f} +- {:.3f}  ->  '.format(r['gap'], r['gap_ci95']) +
          ('base smoother than oracle: untapped structure' if r['gap'] > 0.02 else
           'base ~= oracle: rational shrinkage, nothing to mine' if r['gap'] > -0.02 else
           'base wilder than oracle (check anchor)'))
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(r, f, indent=2)


if __name__ == '__main__':
    main()
