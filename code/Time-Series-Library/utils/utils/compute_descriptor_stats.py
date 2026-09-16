"""
Train-set statistics for the descriptor targets (utils/descriptor_labels.py).

Scans the training split once and stores, in descriptor_cache/<data>_pl<pred_len>.npz:
  - cp_score_q: quantiles [10, 25, 50, 75, 90]% of the scale-free CUSUM score
    (kept for calibration reference / diagnostics; the probability mapping
    itself uses the fixed constants in utils/descriptor_labels.py);
  - cp_score_edges: quantiles [20, 40, 60, 80]% of the CUSUM score, used as
    5-group structural keys by analysis/gain_localization.py;
  - drift_edges / vol_edges / slope_edges: [10, 30, 70, 90]% quantiles of the
    mid-window statistics, used as fixed 5-bin ordinal classification edges;
  - spec_mean / spec_std: mean and std of the 5-dim spectral features, used
    for z-score normalization.

ensure_descriptor_stats(args) is called from exp/exp_descriptor_forecast.py at
init when args.aux_head == 'desc'; it computes and caches the npz on first use.
The cache lives under TSLIB/descriptor_cache/ (never inside dataset/).
"""
import os

import numpy as np
import torch

from data_provider.data_factory import data_provider
from utils.descriptor_labels import cusum_score_pos, mid_window_stats, spectral_features

CACHE_DIR = 'descriptor_cache'
_EDGE_QUANTILES = [0.1, 0.3, 0.7, 0.9]
_CP_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
_CP_EDGE_QUANTILES = [0.2, 0.4, 0.6, 0.8]  # 5 equal-mass CUSUM-score groups for gain localization


def stats_cache_path(args):
    key = os.path.splitext(os.path.basename(args.data_path))[0]
    suffix = '{}_pl{}'.format(key, args.pred_len)
    # a subsampled train split (smoke tests) must not poison the full-data cache
    if getattr(args, 'max_train_windows', -1) is not None and args.max_train_windows > 0:
        suffix += '_mtw{}'.format(args.max_train_windows)
    return os.path.join(CACHE_DIR, suffix + '.npz')


def compute_descriptor_stats(args):
    train_data, train_loader = data_provider(args, flag='train')
    f_dim = -1 if args.features == 'MS' else 0
    scores, drifts, vols, slopes, specs = [], [], [], [], []
    drifts_ch, vols_ch, slopes_ch, specs_ch = [], [], [], []
    vols_near, vols_near_ch = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            future = batch_y[:, -args.pred_len:, f_dim:].float()
            score, _ = cusum_score_pos(future, args.desc_k)
            drift, vol, slope = mid_window_stats(future)
            spec = spectral_features(future)
            scores.append(score.numpy())
            drifts.append(drift.numpy())
            vols.append(vol.numpy())
            slopes.append(slope.numpy())
            specs.append(spec.numpy())
            # channel-preserving variants (for --desc_mode channel)
            drift_c, vol_c, slope_c = mid_window_stats(future, reduce=None)
            spec_c = spectral_features(future, reduce=None)
            drifts_ch.append(drift_c.reshape(-1).numpy())
            vols_ch.append(vol_c.reshape(-1).numpy())
            slopes_ch.append(slope_c.reshape(-1).numpy())
            specs_ch.append(spec_c.reshape(-1, spec_c.shape[-1]).numpy())
            # near-window volatility (for --vol_ms)
            k_near = max(min(args.desc_k, future.shape[1]), 2)
            vn = future[:, :k_near, :].diff(dim=1).std(dim=1, unbiased=False)
            vols_near.append(vn.mean(dim=1).numpy())
            vols_near_ch.append(vn.reshape(-1).numpy())
    scores = np.concatenate(scores)
    drifts = np.concatenate(drifts)
    vols = np.concatenate(vols)
    slopes = np.concatenate(slopes)
    specs = np.concatenate(specs, axis=0)
    drifts_ch = np.concatenate(drifts_ch)
    vols_ch = np.concatenate(vols_ch)
    slopes_ch = np.concatenate(slopes_ch)
    specs_ch = np.concatenate(specs_ch, axis=0)
    vols_near = np.concatenate(vols_near)
    vols_near_ch = np.concatenate(vols_near_ch)

    stats = {
        'cp_score_q': np.quantile(scores, _CP_QUANTILES).astype(np.float32),
        'cp_score_edges': np.quantile(scores, _CP_EDGE_QUANTILES).astype(np.float32),
        'drift_edges': np.quantile(drifts, _EDGE_QUANTILES).astype(np.float32),
        'vol_edges': np.quantile(vols, _EDGE_QUANTILES).astype(np.float32),
        'vol_log_edges': np.quantile(np.log(vols + 1e-8), _EDGE_QUANTILES).astype(np.float32),
        'vol_near_edges': np.quantile(vols_near, _EDGE_QUANTILES).astype(np.float32),
        'slope_edges': np.quantile(slopes, _EDGE_QUANTILES).astype(np.float32),
        'spec_mean': specs.mean(axis=0).astype(np.float32),
        'spec_std': specs.std(axis=0).astype(np.float32),
        'drift_edges_ch': np.quantile(drifts_ch, _EDGE_QUANTILES).astype(np.float32),
        'vol_edges_ch': np.quantile(vols_ch, _EDGE_QUANTILES).astype(np.float32),
        'vol_log_edges_ch': np.quantile(np.log(vols_ch + 1e-8), _EDGE_QUANTILES).astype(np.float32),
        'vol_near_edges_ch': np.quantile(vols_near_ch, _EDGE_QUANTILES).astype(np.float32),
        'slope_edges_ch': np.quantile(slopes_ch, _EDGE_QUANTILES).astype(np.float32),
        'spec_mean_ch': specs_ch.mean(axis=0).astype(np.float32),
        'spec_std_ch': specs_ch.std(axis=0).astype(np.float32),
        'n_windows': np.int64(scores.shape[0]),
        'desc_k': np.int64(args.desc_k),
    }
    path = stats_cache_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp{}.npz'.format(os.getpid())
    np.savez(tmp, **stats)
    os.replace(tmp, path)  # atomic: concurrent first-runs on the same key never corrupt
    print('descriptor stats computed on {} train windows -> {}'.format(scores.shape[0], path))
    return stats


REQUIRED_KEYS = ('cp_score_q', 'cp_score_edges', 'drift_edges', 'vol_edges', 'slope_edges',
                 'spec_mean', 'spec_std', 'drift_edges_ch', 'vol_edges_ch', 'slope_edges_ch',
                 'spec_mean_ch', 'spec_std_ch', 'vol_log_edges', 'vol_log_edges_ch',
                 'vol_near_edges', 'vol_near_edges_ch')


def ensure_descriptor_stats(args):
    path = stats_cache_path(args)
    stats = dict(np.load(path)) if os.path.exists(path) else None
    if stats is None or any(k not in stats for k in REQUIRED_KEYS):
        import fcntl
        os.makedirs(CACHE_DIR, exist_ok=True)
        # 同键并发首跑时只让一个进程计算，其余阻塞等待后读缓存
        with open(path + '.lock', 'w') as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            if os.path.exists(path):
                stats = dict(np.load(path))
            if stats is None or any(k not in stats for k in REQUIRED_KEYS):
                stats = compute_descriptor_stats(args)
    print('descriptor stats loaded from {}'.format(path))
    return stats
