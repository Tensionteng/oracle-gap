"""
Future-window structural descriptor targets for the auxiliary descriptor head.

All functions are fully vectorized torch ops and run on any device. The input
is a batch of future windows `future` of shape [B, L, C] (the last pred_len
steps of seq_y, in the dataset's StandardScaler space).

Descriptor set (three scales):
  - near window (first k steps): CUSUM change-point probability + position;
  - mid window (full window): drift / volatility / trend-slope magnitudes as
    5-bin ordinal class labels (bin edges = train-set quantiles, loaded from
    the stats npz by utils/compute_descriptor_stats.py);
  - whole window: 5-dim spectral features (spectral centroid, spectral
    entropy, low/mid/high band energy ratios), z-scored with the train-set
    mean/std from the stats npz.
"""
import math

import torch

# Empirical calibration constants for the change-point probability mapping
#   p_cp = sigmoid(CP_CAL_A * score - CP_CAL_B)
# where score = max_t |CUSUM_t| / (std * sqrt(k)) is the scale-free CUSUM
# statistic of the mean-removed near window, averaged over channels. For pure
# noise this statistic follows the supremum of a Brownian bridge (mean ~0.87,
# 95th percentile ~1.36, the Kolmogorov-Smirnov critical value); autocorrelated
# series sit higher (ETTh1 train median ~1.6). CP_CAL_A / CP_CAL_B were
# calibrated by maximizing agreement with ruptures PELT (model='l2',
# min_size=3, pen=4.0) detections on ETT training windows: prob>0.5 then fires
# at score ~1.85, the PELT-positive-rate-matched quantile (see
# analysis/verify_cp_labels.py for the agreement report).
CP_CAL_A = 4.0
CP_CAL_B = 7.5

_EPS = 1e-8


def cusum_score_pos(future, k, reduce='mean'):
    """CUSUM change-point statistic on the near window (first k steps).

    future: [B, L, C] -> (score, pos), shapes [B] with reduce='mean'
    (channel-averaged) or [B, C] with reduce=None (channel-preserving).
    score: scale-free max |CUSUM|; pos: argmax location of the |CUSUM| path,
    normalized to [0, 1].
    """
    B, L, C = future.shape
    k = max(min(k, L), 2)
    w = future[:, :k, :]
    mu = w.mean(dim=1, keepdim=True)
    sd = w.std(dim=1, keepdim=True, unbiased=False) + _EPS
    s = (w - mu).cumsum(dim=1)                                   # [B, k, C]
    sabs = s.abs()
    score = sabs.amax(dim=1) / (sd.squeeze(1) * math.sqrt(k))    # [B, C]
    pos = sabs.argmax(dim=1).float() / (k - 1)                   # [B, C]
    if reduce == 'mean':
        return score.mean(dim=1), pos.mean(dim=1)
    return score, pos


def mid_window_stats(future, reduce='mean'):
    """Mid-window statistics over the full future window.

    future: [B, L, C] -> (drift, volatility, slope), shapes [B] with
    reduce='mean' or [B, C] with reduce=None.
    drift: last-third mean minus first-third mean; volatility: std of the
    first difference; slope: least-squares linear fit.
    """
    B, L, C = future.shape
    seg = max(L // 3, 1)
    drift = future[:, -seg:, :].mean(dim=1) - future[:, :seg, :].mean(dim=1)   # [B, C]
    vol = future.diff(dim=1).std(dim=1, unbiased=False)                        # [B, C]
    t = torch.arange(L, device=future.device, dtype=future.dtype)
    t = t - t.mean()
    slope = ((future - future.mean(dim=1, keepdim=True)) * t.view(1, L, 1)).mean(dim=1) \
        / (t * t).mean()                                                       # [B, C]
    if reduce == 'mean':
        return drift.mean(dim=1), vol.mean(dim=1), slope.mean(dim=1)
    return drift, vol, slope


def spectral_features(future, reduce='mean'):
    """Whole-window spectral features via the rFFT power spectrum.

    future: [B, L, C] -> [B, 5] with reduce='mean' or [B, C, 5] with
    reduce=None: spectral centroid (normalized by the Nyquist frequency),
    spectral entropy (normalized by log(n_freq)), and low/mid/high band energy
    ratios (equal thirds of the rFFT bins). All five are scale-invariant.
    """
    B, L, C = future.shape
    x = future - future.mean(dim=1, keepdim=True)
    spec = torch.fft.rfft(x, dim=1)
    power = spec.real ** 2 + spec.imag ** 2                      # [B, F, C]
    p = power / (power.sum(dim=1, keepdim=True) + _EPS)
    n_freq = p.shape[1]
    freqs = torch.linspace(0.0, 1.0, n_freq, device=future.device, dtype=future.dtype)
    centroid = (p * freqs.view(1, -1, 1)).sum(dim=1)             # [B, C]
    entropy = -(p * (p + _EPS).log()).sum(dim=1) / math.log(n_freq)
    b = max(n_freq // 3, 1)
    low = p[:, :b, :].sum(dim=1)
    mid = p[:, b:2 * b, :].sum(dim=1)
    high = p[:, 2 * b:, :].sum(dim=1)
    feats = torch.stack([centroid, entropy, low, mid, high], dim=-1)  # [B, C, 5]
    if reduce == 'mean':
        return feats.mean(dim=1)                                      # [B, 5]
    return feats                                                      # [B, C, 5]


def compute_descriptor_targets(future, k, stats, reduce='mean', vol_log=False, vol_ms=False,
                               vol_qr=False):
    """Full descriptor target dict for a batch of future windows.

    future: [B, L, C]; stats: dict of torch tensors (on the same device) with
    keys drift_edges / vol_edges / slope_edges ([4] bin edges) and spec_mean /
    spec_std ([5] z-score parameters), as stored in the stats npz. In channel
    mode (reduce=None) the per-channel variants drift_edges_ch etc. are used
    and all outputs keep the C dimension: cp_prob/cp_pos [B, C], *_cls
    [B, C] long, spectral [B, C, 5]. With vol_log=True the volatility label is
    binned in log space (log(vol) with vol_log_edges), which spreads the
    right-skewed raw vol distribution more evenly across bins. With
    vol_ms=True an additional near-window volatility label 'vol_near_cls'
    (diff std of the first k steps, binned with vol_near_edges) is returned.
    """
    channel_mode = reduce is None
    suffix = '_ch' if channel_mode else ''
    score, pos = cusum_score_pos(future, k, reduce=reduce)
    cp_prob = torch.sigmoid(CP_CAL_A * score - CP_CAL_B)         # [B] or [B, C]
    drift, vol, slope = mid_window_stats(future, reduce=reduce)
    spec = spectral_features(future, reduce=reduce)
    spec = (spec - stats['spec_mean' + suffix]) / (stats['spec_std' + suffix] + _EPS)
    vol_key = 'vol_log_edges' + suffix if vol_log else 'vol_edges' + suffix
    vol_label = (vol + _EPS).log() if vol_log else vol
    out = {
        'cp_prob': cp_prob,                                      # in (0, 1)
        'cp_pos': pos,                                           # in [0, 1]
        'drift_cls': torch.bucketize(drift, stats['drift_edges' + suffix]),  # long, 0..4
        'vol_cls': torch.bucketize(vol_label, stats[vol_key]),               # long, 0..4
        'slope_cls': torch.bucketize(slope, stats['slope_edges' + suffix]),  # long, 0..4
        'spectral': spec,                                        # z-scored
    }
    if vol_ms:
        # multi-scale volatility: near-window (first k steps) diff std binned
        # with its own train-quantile edges, alongside the full-window vol
        vol_near = future[:, :max(min(k, future.shape[1]), 2), :].diff(dim=1).std(dim=1, unbiased=False)
        if reduce == 'mean':
            vol_near = vol_near.mean(dim=1)
        out['vol_near_cls'] = torch.bucketize(vol_near, stats['vol_near_edges' + suffix])
    if vol_qr:
        # log-volatility regression target (continuous, replaces the binned CE)
        out['vol_log'] = (vol + _EPS).log()
    return out
