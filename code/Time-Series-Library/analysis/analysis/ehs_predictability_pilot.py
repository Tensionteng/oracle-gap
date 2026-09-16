"""Pilot: can cheap context statistics predict the long-context benefit of a dataset?

For each dataset: compute (a) periodicity strength, (b) drift index per channel, averaged;
correlate with the controlled-sweep long-context benefit (MSE@sl96 - MSE@sl2880).
"""
import numpy as np
import pandas as pd

DATASETS = {
    'ETTh1': ('dataset/ETT-small/ETTh2.csv'.replace('ETTh2', 'ETTh1'), 24),
    'ETTm1': ('dataset/ETT-small/ETTm1.csv', 96),
    'exchange_rate': ('dataset/exchange_rate/exchange_rate.csv', None),
    'weather': ('dataset/weather/weather.csv', 144),
    'electricity': ('dataset/electricity/electricity.csv', 24),
    'traffic': ('dataset/traffic/traffic.csv', 24),
}
# controlled-sweep MSE (from logs/lookback_ctrl/SUMMARY.md)
MSE = {
    'ETTh1':        {96: 0.4003, 336: 0.4004, 720: 0.3947, 1440: 0.3970, 2880: 0.4576},
    'ETTm1':        {96: 0.3432, 336: 0.3047, 720: 0.3139, 1440: 0.3409, 2880: 0.3419},
    'exchange_rate':{96: 0.0890, 336: 0.1080, 720: 0.1255, 1440: 0.1583, 2880: 0.3330},
    'weather':      {96: 0.1756, 336: 0.1631, 720: 0.1719, 1440: 0.1989, 2880: 0.2247},
    'electricity':  {96: 0.1487, 336: 0.1339, 720: 0.1320, 1440: 0.1314, 2880: 0.1300},
    'traffic':      {96: 0.4073, 336: 0.3639, 720: 0.3586, 1440: 0.3476, 2880: 0.3403},
}


def periodicity_strength(x):
    """Spectral peak ratio: max non-DC periodogram power / total power (after de-mean)."""
    x = x - x.mean()
    if x.std() < 1e-8:
        return 0.0
    P = np.abs(np.fft.rfft(x)) ** 2
    P[0] = 0
    return float(P.max() / (P.sum() + 1e-12))


def drift_index(x, n_seg=20):
    """Regime-variance ratio: variance of segment means / overall variance (0=stationary level, ->1 strong level drift)."""
    segs = np.array_split(x, n_seg)
    means = np.array([s.mean() for s in segs])
    return float(means.var() / (x.var() + 1e-12))


def acf_long_memory(x, lag_frac=0.25):
    """Autocorrelation at long lag (25% of length): long-memory signature."""
    x = x - x.mean()
    lag = max(1, int(len(x) * lag_frac))
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


rows = []
for name, (path, _) in DATASETS.items():
    df = pd.read_csv(path)
    X = df.drop(columns=[c for c in df.columns if c.lower() == 'date']).values.astype(float)
    X = X[: min(len(X), 12000)]  # uniform window for stats
    ch = np.random.default_rng(0).choice(X.shape[1], size=min(50, X.shape[1]), replace=False)
    per = np.mean([periodicity_strength(X[:, j]) for j in ch])
    drf = np.mean([drift_index(X[:, j]) for j in ch])
    lm = np.mean([acf_long_memory(X[:, j]) for j in ch])
    benefit = MSE[name][96] - MSE[name][2880]  # >0 means long context helps
    best_sl = min(MSE[name], key=MSE[name].get)
    rows.append((name, per, drf, lm, benefit, best_sl))
    print(f"{name:15} periodicity={per:.3f} drift={drf:.3f} long_acf={lm:.3f}  benefit(mse96-mse2880)={benefit:+.4f} best_sl={best_sl}")

from scipy.stats import spearmanr
arr = {k: np.array([r[i] for r in rows]) for i, k in enumerate(['name', 'per', 'drf', 'lm', 'benefit', 'best_sl'])}
for stat in ['per', 'drf', 'lm']:
    rho, p = spearmanr(arr[stat].astype(float), arr['benefit'].astype(float))
    print(f"spearman({stat}, long-context benefit) = {rho:+.3f} (p={p:.3f})")
