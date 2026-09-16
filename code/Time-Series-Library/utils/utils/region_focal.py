"""
RegionFocal region-weighted loss (anti-over-smoothing), used via
--task_loss regionfocal.

Notation: prediction yhat, truth y (both [B, L, C], standardized space),
anchor m. Deviations d = y - m, p = yhat - m.

Regions (soft gates, temperatures fixed at tau1 = tau2 = 0.1):
  direction consistency  s = sigmoid(p * d / tau1)
  magnitude ratio        r = |p| / (|d| + 1e-8)
  U (undershoot) membership u = s * sigmoid((1 - r) / tau2)
  O (overshoot)  membership o = s * sigmoid((r - 1) / tau2)
  per-point weight w = 1 + alpha_U * u + alpha_O * o + alpha_W * (1 - s)

Loss: L = mean[w * (yhat - y)^2] + lambda_base * mean_{B,C}(mean_L(yhat-y))^2
(the base term squares the per-window mean residual, then averages over B/C).

Default semantics: w is detached; alpha_O = 0; alpha_W follows alpha_U / 2
(unless --rf_aw >= 0); lambda_base = 0. rf_soft=0 replaces the sigmoid gates
by hard thresholds; rf_detach=0 lets gradients flow through w as well.

Note on the sanity expectation for a full-anchor prediction (yhat = m): with
the soft gate, s = sigmoid(0) = 0.5 sits at its midpoint, so u = 0.5 *
sigmoid(1/tau2) ~= 0.5 (o ~= 0) - the U channel dominates but u does not
reach 1.0 under the fixed formula; analysis/sanity_regionfocal.py asserts the
formula-consistent values.
"""
import torch
from torch import nn

_FREQ_PERIOD = {'h': 24, 't': 96, 'd': 7, 'w': 4, 'm': 12}


class RegionFocalLoss(nn.Module):
    def __init__(self, au=1.0, ao=0.0, aw=-1.0, lambda_base=0.0,
                 anchor='ctxmean', soft=1, detach=1, tau1=0.1, tau2=0.1,
                 seq_len=96, freq='h'):
        super().__init__()
        self.au = au
        self.ao = ao
        self.aw = au / 2.0 if aw < 0 else aw
        self.lambda_base = lambda_base
        self.anchor = anchor
        self.soft = soft
        self.detach = detach
        self.tau1 = tau1
        self.tau2 = tau2
        self.seq_len = seq_len
        self.freq = freq
        self._warned = False

    def compute_anchor(self, batch_x, pred_len):
        """batch_x: [B, seq_len, C] (already f_dim-sliced) -> m [B, pred_len, C]."""
        B = batch_x.shape[0]
        C = batch_x.shape[2]
        if self.anchor == 'zero':
            return torch.zeros(B, pred_len, C, device=batch_x.device, dtype=batch_x.dtype)
        if self.anchor == 'seasonal':
            period = _FREQ_PERIOD.get(self.freq)
            if period is None or self.seq_len < period:
                if not self._warned:
                    print('regionfocal: seasonal anchor unavailable (freq={}, seq_len={}); '
                          'falling back to ctxmean'.format(self.freq, self.seq_len))
                    self._warned = True
            else:
                idx = self.seq_len - period + (torch.arange(pred_len, device=batch_x.device) % period)
                return batch_x[:, idx, :]
        # ctxmean (default / fallback): lookback per-channel mean broadcast
        return batch_x.mean(dim=1, keepdim=True).expand(B, pred_len, C)

    def memberships(self, p, d):
        """p, d: [B, L, C] -> (s, u, o) same shape."""
        r = p.abs() / (d.abs() + 1e-8)
        if self.soft:
            s = torch.sigmoid(p * d / self.tau1)
            u = s * torch.sigmoid((1.0 - r) / self.tau2)
            o = s * torch.sigmoid((r - 1.0) / self.tau2)
        else:
            s = (p * d > 0).float()
            u = s * (r < 1.0).float()
            o = s * (r > 1.0).float()
        return s, u, o

    def point_weights(self, p, d, gate=None):
        """p, d: [B, L, C]; gate: optional per-window (per-channel) scaling of
        the alpha_U term, [B] or [B, C] (already sigmoid-ed and detached by
        the caller). Returns w [B, L, C]."""
        s, u, o = self.memberships(p, d)
        au = self.au
        if gate is not None:
            while gate.dim() < p.dim():
                gate = gate.unsqueeze(1)            # [B,1] / [B,1,C]
            au = self.au * gate
        return 1.0 + au * u + self.ao * o + self.aw * (1.0 - s)

    def forward(self, pred, true, anchor=None, gate=None):
        if anchor is None:
            anchor = torch.zeros_like(true)
        d = true - anchor
        p = pred - anchor
        w = self.point_weights(p, d, gate=gate)
        if self.detach:
            w = w.detach()
        loss = (w * (pred - true) ** 2).mean()
        if self.lambda_base:
            res = (pred - true).mean(dim=1)          # [B, C] per-window mean residual
            loss = loss + self.lambda_base * (res ** 2).mean()
        return loss
