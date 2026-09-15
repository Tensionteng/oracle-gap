"""EHS predictability statistics v2.

Lessons from v1 (analysis/ehs_predictability_pilot.py): naive spectral-peak periodicity is
contaminated by trend low-frequency power (exchange_rate scored 0.558 spuriously).
v2 fixes: detrend before spectrum; add calendar-period-aligned ACF; per-channel outputs.

Usage: python analysis/ehs_stats.py <csv_path> --periods 24,168 [--n-channels 50]
"""
import argparse
import numpy as np
import pandas as pd


def detrend(x):
    t = np.arange(len(x))
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def periodicity_detrended(x):
    """Spectral peak ratio after linear detrending."""
    xd = detrend(x - x.mean())
    if xd.std() < 1e-8:
        return 0.0
    P = np.abs(np.fft.rfft(xd)) ** 2
    P[0] = 0
    return float(P.max() / (P.sum() + 1e-12))


def acf_at(x, lag):
    x = x - x.mean()
    if lag >= len(x) or x.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def calendar_acf(x, periods):
    """Mean ACF at known calendar period lags (phase-aligned periodicity)."""
    vals = [acf_at(x, p) for p in periods]
    return float(np.nanmean(vals)) if vals else np.nan


def drift_index(x, n_seg=20):
    segs = np.array_split(x, n_seg)
    means = np.array([s.mean() for s in segs if len(s)])
    return float(means.var() / (x.var() + 1e-12))


def long_acf(x, lag_frac=0.25):
    return acf_at(x, max(1, int(len(x) * lag_frac)))


def dataset_stats(path, periods, n_channels=50, max_len=12000, seed=0):
    df = pd.read_csv(path)
    X = df.drop(columns=[c for c in df.columns if c.lower() == 'date']).values.astype(float)
    X = X[:max_len]
    rng = np.random.default_rng(seed)
    ch = rng.choice(X.shape[1], size=min(n_channels, X.shape[1]), replace=False)
    out = {'periodicity_dt': [], 'calendar_acf': [], 'drift': [], 'long_acf': []}
    for j in ch:
        x = X[:, j]
        out['periodicity_dt'].append(periodicity_detrended(x))
        out['calendar_acf'].append(calendar_acf(x, periods))
        out['drift'].append(drift_index(x))
        out['long_acf'].append(long_acf(x))
    return {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in out.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--periods', default='24,168', help='calendar period lags in timesteps')
    ap.add_argument('--n-channels', type=int, default=50)
    args = ap.parse_args()
    periods = [int(p) for p in args.periods.split(',')]
    stats = dataset_stats(args.csv, periods, args.n_channels)
    print(f"{args.csv}")
    for k, (m, s) in stats.items():
        print(f"  {k:16} mean={m:+.4f} std={s:.4f}")
