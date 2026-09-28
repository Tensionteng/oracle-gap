"""
Post-hoc global calibration baseline (control for regionfocal).

Question: can the regionfocal gain be matched by fitting a cheap global
recalibration on the VALIDATION split and applying it to the test split,
i.e. is a train-time conditional repair necessary at all?

Variants (all parameters fit on val preds -> applied to test preds):
  - scalar_gain   : s = <y,p>/<p,p> over all val points (1 param)
  - channel_gain  : per-channel least-squares gain (C params)
  - step_gain     : per-horizon-step least-squares gain (H params)
  - step_affine   : per-step a_h + b_h * p_h (2H params)

Each cell points at a dump dir containing pred.npy/true.npy (test) and
val_pred.npy/val_true.npy (val, written by --save_val_pred 1 reruns).

Usage from the TSLib root:
  uv run python analysis/posthoc_calibration.py
Output: analysis/results/posthoc_calibration.json + markdown to stdout.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = '/mnt/jd/users/tengshiyuan.1/codes/mtp4ts'
TSLIB = os.path.join(ROOT, 'Time-Series-Library')

CELLS = {
    # cell name: (base dump glob, rfau1 dump glob) - resolved under both
    # results/ and pred_dumps/
    'ECL': ('long_term_forecast_ECL_96_96_PatchTST_custom_*_Exp_0',
            'long_term_forecast_ECL_96_96_rfau1_PatchTST_custom_*_Exp_rf1_0'),
    'm4_weekly': ('long_term_forecast_m4w_96_96_PatchTST_custom_*_Exp_0',
                  'long_term_forecast_m4w_96_96_rfau1_PatchTST_custom_*_Exp_rf_0'),
}


def find_dump(pattern):
    for base in ('results', 'pred_dumps'):
        hits = [d for d in glob.glob(os.path.join(TSLIB, base, pattern))
                if os.path.exists(os.path.join(d, 'val_pred.npy'))]
        if hits:
            return hits[0]
    raise AssertionError('no dump with val_pred.npy for {}'.format(pattern))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def fit_eval(vp, vt, tp, tt):
    """All fits on (vp, vt); all evals on (tp, tt). Shapes [N, H, C]."""
    out = {}
    # scalar gain
    s = float(np.sum(vp * vt) / np.sum(vp * vp))
    out['scalar_gain'] = {'param': s, 'test_mse': mse(tp * s, tt)}
    # per-channel gain  [C]
    sc = np.sum(vp * vt, axis=(0, 1)) / np.sum(vp * vp, axis=(0, 1))
    out['channel_gain'] = {'test_mse': mse(tp * sc[None, None, :], tt)}
    # per-step gain  [H]
    sh = np.sum(vp * vt, axis=(0, 2)) / np.sum(vp * vp, axis=(0, 2))
    out['step_gain'] = {'test_mse': mse(tp * sh[None, :, None], tt)}
    # per-step affine  a_h + b_h p
    xm = vp.mean(axis=(0, 2), keepdims=False)  # [H]
    ym = vt.mean(axis=(0, 2))
    b = (np.sum((vp - xm[None, :, None]) * (vt - ym[None, :, None]), axis=(0, 2))
         / np.sum((vp - xm[None, :, None]) ** 2, axis=(0, 2)))
    a = ym - b * xm
    out['step_affine'] = {'test_mse': mse(tp * b[None, :, None] + a[None, :, None], tt)}
    return out


def main():
    report = {}
    for cell, (base_pat, rf_pat) in CELLS.items():
        entry = {}
        for tag, pat in (('base', base_pat), ('rfau1', rf_pat)):
            d = find_dump(pat)
            vp = np.load(os.path.join(d, 'val_pred.npy')).astype(np.float64)
            vt = np.load(os.path.join(d, 'val_true.npy')).astype(np.float64)
            tp = np.load(os.path.join(d, 'pred.npy')).astype(np.float64)
            tt = np.load(os.path.join(d, 'true.npy')).astype(np.float64)
            assert vp.shape == vt.shape and tp.shape == tt.shape
            assert vp.shape[1:] == tp.shape[1:], '{}: val/test window shape mismatch'.format(d)
            r = fit_eval(vp, vt, tp, tt)
            r['test_mse_base'] = mse(tp, tt)
            r['dump'] = os.path.basename(d)
            entry[tag] = r
        report[cell] = entry

    os.makedirs(os.path.join(TSLIB, 'analysis', 'results'), exist_ok=True)
    with open(os.path.join(TSLIB, 'analysis', 'results', 'posthoc_calibration.json'), 'w') as f:
        json.dump(report, f, indent=2)

    md = ['# Post-hoc calibration vs regionfocal (test MSE, normalized space)', '',
          '| cell | base | scalar | channel | step | step+affine | rfau1 (train-time) |',
          '|---|---|---|---|---|---|---|']
    for cell, entry in report.items():
        b, r = entry['base'], entry['rfau1']
        md.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
            cell, b['test_mse_base'], b['scalar_gain']['test_mse'],
            b['channel_gain']['test_mse'], b['step_gain']['test_mse'],
            b['step_affine']['test_mse'], r['test_mse_base']))
        md.append('| {} (calibrated rfau1) | - | {:.4f} | {:.4f} | {:.4f} | {:.4f} | - |'.format(
            cell, r['scalar_gain']['test_mse'], r['channel_gain']['test_mse'],
            r['step_gain']['test_mse'], r['step_affine']['test_mse']))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
