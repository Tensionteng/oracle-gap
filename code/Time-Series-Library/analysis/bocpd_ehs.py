#!/usr/bin/env python
"""Mechanism experiment 4: BOCPD theoretical anchor ("EHS = posterior expected run length").

Simplified Bayesian Online Changepoint Detection (Adams & MacKay 2007):
  - observation model: Gaussian with unknown mean AND variance
    (Normal-Inverse-Gamma prior -> Student-t predictive; the known-variance
    "unknown mean only" variant degenerates here: posterior CP prob collapses
    to the prior hazard, i.e. likelihood carries no information);
  - constant hazard H scanned over {1/50, 1/100, 1/250, 1/500, 1/1000};
  - run-length posterior truncated at R_MAX, vectorized across channels.

Per dataset: train segment (same borders as data_provider), <=24 channels evenly
subsampled, standardized. Statistic: time-averaged posterior expected run length
E[r] (after burn-in), plus mean posterior changepoint probability.

Correlation: Spearman between per-dataset E[r] and the measured best seq_len
(iTransformer / DLinear main-matrix arms of logs/ehs_v2/SUMMARY.md).

Outputs:
  logs/mechanism/exp4/bocpd_summary.md, bocpd_results.npz
  logs/mechanism/figs/exp4_runlength_curves.png, exp4_bocpd_scatter.png
"""
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGS = os.path.join(ROOT, 'logs', 'mechanism', 'figs')
OUT = os.path.join(ROOT, 'logs', 'mechanism', 'exp4')
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

R_MAX = 3000
BURN_IN = 500
N_CH = 24
KAPPA0 = 1.0  # prior pseudo-count for the unknown mean
HAZARDS = [1 / 50, 1 / 100, 1 / 250, 1 / 500, 1 / 1000]

# name: (csv path, train rows or frac)
DATASETS = {
    'ETTh1': ('dataset/ETT-small/ETTh1.csv', 12 * 30 * 24),
    'ETTh2': ('dataset/ETT-small/ETTh2.csv', 12 * 30 * 24),
    'ETTm1': ('dataset/ETT-small/ETTm1.csv', 12 * 30 * 24 * 4),
    'ETTm2': ('dataset/ETT-small/ETTm2.csv', 12 * 30 * 24 * 4),
    'exchange_rate': ('dataset/exchange_rate/exchange_rate.csv', 0.7),
    'weather': ('dataset/weather/weather.csv', 0.7),
    'electricity': ('dataset/electricity/electricity.csv', 0.7),
    'traffic': ('dataset/traffic/traffic.csv', 0.7),
}

# measured best seq_len from logs/ehs_v2/SUMMARY.md main matrix
BEST_SL = {
    #            iTr  DLin
    'ETTh1': (720, 1440),
    'ETTh2': (96, 1440),
    'ETTm1': (336, 336),
    'ETTm2': (336, 1440),
    'exchange_rate': (96, 96),
    'weather': (336, 2880),
    'electricity': (2880, 2880),
    'traffic': (2880, 2880),
}


def load_train(name):
    path, take = DATASETS[name]
    df = pd.read_csv(os.path.join(ROOT, path))
    data = df.drop(columns=['date']).values.astype(np.float64)
    n = take if isinstance(take, int) else int(len(data) * take)
    data = data[:n]
    C = data.shape[1]
    idx = np.linspace(0, C - 1, min(N_CH, C)).astype(int)
    data = data[:, idx]
    mu = data.mean(0, keepdims=True)
    sd = data.std(0, keepdims=True) + 1e-8
    return ((data - mu) / sd).astype(np.float64)  # [T, C]


def bocpd_runlength(x, H, r_max=R_MAX, kappa0=KAPPA0, alpha0=1.0, beta0=1.0, m0=0.0):
    """x: [T, C] standardized. NIG prior -> Student-t predictive.
    Returns E[r]_t [T], p_cp_t [T] (channel-averaged)."""
    from scipy.special import gammaln
    T, C = x.shape
    R = r_max
    p = np.zeros((C, R))
    p[:, 0] = 1.0
    m = np.full((C, R), m0)
    kap = np.full((C, R), kappa0)
    al = np.full((C, R), alpha0)
    be = np.full((C, R), beta0)
    Er = np.zeros(T)
    Pcp = np.zeros(T)
    log1mH = np.log1p(-H)
    ridx = np.arange(R)
    for t in range(T):
        xt = x[t]  # [C]
        # Student-t predictive logpdf per run-length slot
        nu = 2.0 * al
        s2 = be * (kap + 1.0) / (al * kap)               # [C, R]
        lp = (gammaln(al + 0.5) - gammaln(al)
              - 0.5 * np.log(np.pi * nu * s2)
              - (al + 0.5) * np.log1p((xt[:, None] - m) ** 2 / (nu * s2)))
        lg = np.log(p + 1e-300) + lp
        mx = lg.max(axis=1, keepdims=True)
        w = np.exp(lg - mx)                              # unnormalized, per channel
        tot = w.sum(axis=1, keepdims=True)               # [C, 1]
        # changepoint mass and growth mass
        p_cp = H * tot
        grow = w * np.exp(log1mH)                        # [C, R]
        new_p = np.empty_like(p)
        new_p[:, 0] = p_cp[:, 0]
        new_p[:, 1:] = grow[:, :-1]
        new_p[:, -1] += grow[:, -1]                      # absorbing truncation slot
        new_p /= new_p.sum(axis=1, keepdims=True)
        # NIG parameter update: slot r+1 inherits slot r updated with x_t
        new_kap = np.empty_like(kap)
        new_m = np.empty_like(m)
        new_al = np.empty_like(al)
        new_be = np.empty_like(be)
        new_kap[:, 1:] = kap[:, :-1] + 1.0
        new_m[:, 1:] = (kap[:, :-1] * m[:, :-1] + xt[:, None]) / new_kap[:, 1:]
        new_al[:, 1:] = al[:, :-1] + 0.5
        new_be[:, 1:] = be[:, :-1] + kap[:, :-1] * (xt[:, None] - m[:, :-1]) ** 2 / (2.0 * (kap[:, :-1] + 1.0))
        new_kap[:, 0] = kappa0 + 1.0
        new_m[:, 0] = (kappa0 * m0 + xt) / (kappa0 + 1.0)
        new_al[:, 0] = alpha0 + 0.5
        new_be[:, 0] = beta0 + kappa0 * (xt - m0) ** 2 / (2.0 * (kappa0 + 1.0))
        # truncation slot: also keep its own stats updated (approx)
        new_kap[:, -1] = kap[:, -1] + 1.0
        new_m[:, -1] = (kap[:, -1] * m[:, -1] + xt) / new_kap[:, -1]
        new_al[:, -1] = al[:, -1] + 0.5
        new_be[:, -1] = be[:, -1] + kap[:, -1] * (xt - m[:, -1]) ** 2 / (2.0 * (kap[:, -1] + 1.0))
        p, m, kap, al, be = new_p, new_m, new_kap, new_al, new_be
        Er[t] = (p * ridx).sum(1).mean()
        Pcp[t] = p[:, 0].mean()
    return Er, Pcp


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind='mergesort')
    sa = a[order]
    r = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    rho = (rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    n = len(x)
    # t-approx p-value (two-sided); rough for n=8, report rho primarily
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    from math import erf
    p_approx = 2 * (1 - 0.5 * (1 + erf(abs(t) / np.sqrt(2))))
    return rho, p_approx


def main():
    t0 = time.time()
    results = {}  # (name, H) -> (meanEr, meanPcp)
    curves = {}   # name -> (Er, Pcp) at H=1/250
    for name in DATASETS:
        x = load_train(name)
        for H in HAZARDS:
            Er, Pcp = bocpd_runlength(x, H)
            results[(name, H)] = (Er[BURN_IN:].mean(), Pcp[BURN_IN:].mean())
            if abs(H - 1 / 250) < 1e-9:
                curves[name] = (Er, Pcp)
        print(f'[bocpd] {name} done ({time.time() - t0:.0f}s)', flush=True)

    # ---- table + Spearman ----
    names = list(DATASETS.keys())
    lines = ['# Exp4: BOCPD expected run length vs measured best seq_len', '',
             f'R_MAX={R_MAX}, burn-in={BURN_IN}, channels<={N_CH}, kappa0={KAPPA0}', '',
             '| dataset | ' + ' | '.join(f'H={h:g}' for h in HAZARDS) +
             ' | best sl iTr | best sl DLin |', '|---|' + '---|' * (len(HAZARDS) + 2)]
    for name in names:
        row = [f'{results[(name, H)][0]:.0f}' for H in HAZARDS]
        lines.append(f'| {name} | ' + ' | '.join(row) +
                     f' | {BEST_SL[name][0]} | {BEST_SL[name][1]} |')
    lines.append('')
    for H in HAZARDS:
        er = [results[(n, H)][0] for n in names]
        rho_i, p_i = spearman(er, [BEST_SL[n][0] for n in names])
        rho_d, p_d = spearman(er, [BEST_SL[n][1] for n in names])
        rho_m, p_m = spearman(er, [np.mean(BEST_SL[n]) for n in names])
        lines.append(f'- H={H:g}: Spearman(E[r], best sl) iTransformer rho={rho_i:.3f} (p~{p_i:.3f}), '
                     f'DLinear rho={rho_d:.3f} (p~{p_d:.3f}), mean-of-two rho={rho_m:.3f} (p~{p_m:.3f})')
    lines.append('')
    lines.append('Note: with a constant hazard, the per-step posterior CP probability p(r_t=0) '
                 'is identically H by construction (both posterior branches scale with the same '
                 'predictive mass), so the data-driven statistic is the run-length distribution: '
                 'E[r] above; downward resets of E[r] in the curves mark detected changepoints.')
    with open(os.path.join(OUT, 'bocpd_summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    np.savez(os.path.join(OUT, 'bocpd_results.npz'),
             **{f'Er_{n}': curves[n][0] for n in names},
             **{f'Pcp_{n}': curves[n][1] for n in names})

    # ---- figures ----
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, name in zip(axes.flat, names):
        Er, Pcp = curves[name]
        T = len(Er)
        w0, w1 = BURN_IN, min(T, BURN_IN + 4000)
        tt = np.arange(w0, w1)
        ax.plot(tt, Er[w0:w1], lw=0.8, color='tab:blue')
        ax.set_yscale('log')
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('t (train steps)', fontsize=8)
        # run-length reset strength: marks detected changepoints
        rs = np.clip(1.0 - Er[1:] / np.maximum(Er[:-1], 1e-9), 0, 1)
        ax2 = ax.twinx()
        ax2.plot(tt, rs[w0:w1], lw=0.6, color='tab:red', alpha=0.6)
        ax2.set_ylim(0, 1.0)
        if ax in axes[:, 0]:
            ax.set_ylabel('E[run length] (blue, log)', fontsize=8)
        if ax in axes[:, -1]:
            ax2.set_ylabel('reset strength (red)', fontsize=8)
    fig.suptitle('BOCPD posterior expected run length (H=1/250); red = run-length resets (detected CPs)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'exp4_runlength_curves.png'), dpi=150)
    plt.close(fig)

    H = 1 / 250
    er = np.array([results[(n, H)][0] for n in names])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for model, mi, marker, color in [('iTransformer', 0, 'o', 'tab:blue'),
                                     ('DLinear', 1, 's', 'tab:orange')]:
        y = np.array([BEST_SL[n][mi] for n in names])
        ax.scatter(er, y, marker=marker, color=color, label=model, s=60, zorder=3)
    for i, n in enumerate(names):
        ax.annotate(n, (er[i], BEST_SL[n][0]), fontsize=7, xytext=(4, 4),
                    textcoords='offset points')
    rho_i, p_i = spearman(er, [BEST_SL[n][0] for n in names])
    rho_d, p_d = spearman(er, [BEST_SL[n][1] for n in names])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('BOCPD posterior expected run length E[r] (H=1/250)')
    ax.set_ylabel('measured best seq_len (ehs_v2 main matrix)')
    ax.set_title(f'E[r] vs best lookback\nSpearman: iTr rho={rho_i:.2f} (p~{p_i:.3f}), '
                 f'DLin rho={rho_d:.2f} (p~{p_d:.3f})')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'exp4_bocpd_scatter.png'), dpi=150)
    plt.close(fig)
    print(f'[bocpd] all done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
