"""
Level-lag predictability probe (mechanism analysis for regionfocal's wins).

Hypothesis: regionfocal helps exactly when the per-window residual's SYSTEMATIC
component (level lag, measured by the residual-mean t-statistic) is predictable
from the lookback context. saugeenday shows t = +0.24 but no gain - if its
lag is not context-predictable (lag R^2 ~ 0), that explains the dissociation.

For each (dump, dataset) grid cell:
  - residual mean per window and its t-statistic across windows;
  - ridge regression (time-blocked 60/40 split) of the window residual mean
    on ~8 lookback-context statistics (channel-mean: mean, std, diff-std,
    slope, last-24-step mean, spectral centroid, lag-1 autocorrelation,
    scale-free CUSUM score) -> out-of-sample R^2 (lag predictability);
  - dispersion gap recomputed in the oracle_diagnostic protocol (residual
    permutation oracle, ctxmean anchor, 10 reps).
Outputs a markdown table + analysis/results/lag_predictability.json.

Runs standalone: uv run python analysis/lag_predictability.py
"""
import json
import math
import os
import sys
import glob

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.oracle_diagnostic import load_inputs, metrics

ROOT = os.environ.get('MTP4TS_ROOT', '.')
TSLIB = os.path.join(ROOT, 'Time-Series-Library')


def find_dump(pattern):
    hits = [d for d in glob.glob(os.path.join(TSLIB, 'results', pattern))
            if os.path.exists(os.path.join(d, 'pred.npy'))]
    assert hits, 'no dump for {}'.format(pattern)
    return hits[0]


# (name, dump glob (base run), root_path, data_path, data_key, rf_effect_str)
GRID = [
    ('m4w', 'long_term_forecast_m4w_96_96_PatchTST_custom_*_Exp_0',
     ROOT + '/gapbench/csv/', 'm4_weekly_W.csv', 'custom', '-4.96%'),
    ('usbirths', 'long_term_forecast_usbirths_96_96_PatchTST_custom_*_Exp_0',
     ROOT + '/gapbench/csv/', 'us_births_W.csv', 'custom', '-3.06%'),
    ('sauge', 'long_term_forecast_sauge_96_96_PatchTST_custom_*_Exp_0',
     ROOT + '/gapbench/csv/', 'saugeenday_D.csv', 'custom', '+1.11%'),
    ('SZTAXI', 'long_term_forecast_SZ_TAXI_96_96_PatchTST_custom_*_Exp_0',
     ROOT + '/gapbench/csv/', 'SZ_TAXI_15T.csv', 'custom', '+0.33%'),
    ('ECL', 'long_term_forecast_ECL_96_96_PatchTST_custom_*_Exp_0',
     TSLIB + '/dataset/electricity/', 'electricity.csv', 'custom', '-2.33%'),
    ('m4w-FreTS', 'long_term_forecast_m4w_96_96_FreTS_custom_*_Exp_0',
     ROOT + '/gapbench/csv/', 'm4_weekly_W.csv', 'custom', '-6.6%'),
    ('m4w-TimesNet', 'long_term_forecast_m4w_96_96_TimesNet_custom_*_Exp_0',
     ROOT + '/gapbench/csv/', 'm4_weekly_W.csv', 'custom', '+14.4%'),
    ('traffic-FreTS', 'long_term_forecast_traffic_96_96_frets_FreTS_custom_*_Exp_0',
     TSLIB + '/dataset/traffic/', 'traffic.csv', 'custom', '+1.0%'),
    ('traffic-iTransformer', 'long_term_forecast_traffic_96_96_iTransformer_custom_*_Exp_0',
     TSLIB + '/dataset/traffic/', 'traffic.csv', 'custom', '+4.4%'),
    ('traffic-PatchTST', 'long_term_forecast_traffic_96_96_PatchTST_custom_*_Exp_0',
     TSLIB + '/dataset/traffic/', 'traffic.csv', 'custom', '-0.9%'),
    ('ETTh1', 'long_term_forecast_ETTh1_96_96_PatchTST_ETTh1_*_Exp_0',
     TSLIB + '/dataset/ETT-small/', 'ETTh1.csv', 'ETTh1', None),
    ('ETTm1', 'long_term_forecast_ETTm1_96_96_PatchTST_ETTm1_*_Exp_0',
     TSLIB + '/dataset/ETT-small/', 'ETTm1.csv', 'ETTm1', None),
]


def context_features(x):
    """x: [N, L, C] -> [N, 8] channel-mean context statistics."""
    L = x.shape[1]
    mean = x.mean(axis=(1, 2))
    std = x.std(axis=(1, 2))
    dstd = np.diff(x, axis=1).std(axis=(1, 2))
    t = np.arange(L, dtype=np.float64) - (L - 1) / 2
    slope = ((x - x.mean(axis=1, keepdims=True)) * t[None, :, None]).mean(axis=1) / (t * t).mean()
    slope = slope.mean(axis=1)
    lastk = x[:, -24:, :].mean(axis=(1, 2))
    xc = x - x.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    p = power / (power.sum(axis=1, keepdims=True) + 1e-8)
    freqs = np.linspace(0, 1, p.shape[1])
    centroid = (p * freqs[None, :, None]).sum(axis=1).mean(axis=1)
    x0, x1 = xc[:, :-1, :], xc[:, 1:, :]
    lag1 = (x0 * x1).mean(axis=(1, 2)) / (xc.var(axis=(1, 2)) + 1e-8)
    cs = np.cumsum(xc, axis=1)
    cusum = (np.abs(cs).max(axis=1) / (xc.std(axis=1) + 1e-8) / math.sqrt(L)).mean(axis=1)
    return np.stack([mean, std, dstd, slope, lastk, centroid, lag1, cusum], axis=1)


def amp_gap(pred, true, anchor, reps=10, seed=0):
    base_m = metrics(pred, true, anchor)
    resid = true - pred
    rng = np.random.default_rng(seed)
    om = []
    for _ in range(reps):
        om.append(metrics(pred, pred + resid[rng.permutation(len(pred))], anchor)['amp'])
    return float(np.mean(om) - base_m['amp'])


def main():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    rows = []
    for name, dump, root, dp, dkey, rf_str in GRID:
        dump = find_dump(dump)
        pred = np.load(os.path.join(dump, 'pred.npy')).astype(np.float64)
        true = np.load(os.path.join(dump, 'true.npy')).astype(np.float64)
        xs = load_inputs(root, dp, dkey, 96, len(pred))
        resid = true - pred
        r_mean = resid.mean(axis=(1, 2))
        t_stat = float(r_mean.mean() / (r_mean.std() / math.sqrt(len(r_mean)) + 1e-12))
        X = context_features(xs)
        n_tr = int(len(X) * 0.6)
        sc = StandardScaler().fit(X[:n_tr])
        r2 = float(r2_score(
            r_mean[n_tr:],
            Ridge(alpha=1.0).fit(sc.transform(X[:n_tr]), r_mean[:n_tr]).predict(sc.transform(X[n_tr:]))))
        gap = amp_gap(pred, true, xs.mean(axis=1, keepdims=True))
        rows.append({'dataset': name, 'n_windows': int(len(pred)),
                     'mse': float((resid ** 2).mean()), 'resid_t': t_stat,
                     'lag_r2': r2, 'dispersion_gap': gap, 'rf_effect': rf_str})
        print('{} done: t={:+.3f} lag_r2={:+.4f} gap={:+.4f}'.format(name, t_stat, r2, gap))

    md = ['# Lag-predictability vs regionfocal effect', '',
          '| dataset | N | dispersion gap | resid t | lag R2 | rf effect |',
          '|---|---|---|---|---|---|']
    for r in rows:
        md.append('| {} | {} | {:+.4f} | {:+.3f} | {:+.4f} | {} |'.format(
            r['dataset'], r['n_windows'], r['dispersion_gap'], r['resid_t'],
            r['lag_r2'], r['rf_effect'] or 'n/a'))
    text = '\n'.join(md)
    print('\n' + text)
    out = os.path.join(TSLIB, 'analysis', 'results', 'lag_predictability.json')
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    with open(out.replace('.json', '.md'), 'w') as f:
        f.write(text)
    print('written: {} (+ .md)'.format(out))


if __name__ == '__main__':
    main()
