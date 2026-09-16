import torch
from torch import nn


class DescriptorHead(nn.Module):
    """Auxiliary head predicting future-window structural descriptors.

    Two-layer shallow MLP (GELU) trunk on the pooled hidden state [B, D],
    then one linear output per descriptor:
      cp_prob  [B]    logits for the change-point probability (BCE target)
      cp_pos   [B]    normalized change-point position in [0, 1]
      drift / vol / slope  [B, n_bins]  ordinal-class logits
      spectral [B, 5] z-scored spectral features
    """

    def __init__(self, d_in, d_hidden=128, n_bins=5, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.cp_prob = nn.Linear(d_hidden, 1)
        self.cp_pos = nn.Linear(d_hidden, 1)
        self.drift = nn.Linear(d_hidden, n_bins)
        self.vol = nn.Linear(d_hidden, n_bins)
        self.vol_near = nn.Linear(d_hidden, n_bins)   # multi-scale vol (--vol_ms)
        self.vol_qr = nn.Linear(d_hidden, 1)          # log-vol regression (--vol_qr)
        self.slope = nn.Linear(d_hidden, n_bins)
        self.spectral = nn.Linear(d_hidden, 5)

    def forward(self, hidden, return_z=False):
        h = self.trunk(hidden)
        # z: the descriptor representation (trunk output, before the heads),
        # used by --desc_cond concat/film to condition the main forecast path
        out = {
            'cp_prob': self.cp_prob(h).squeeze(-1),
            'cp_pos': self.cp_pos(h).squeeze(-1),
            'drift': self.drift(h),
            'vol': self.vol(h),
            'vol_near': self.vol_near(h),
            'vol_qr': self.vol_qr(h).squeeze(-1),
            'slope': self.slope(h),
            'spectral': self.spectral(h),
        }
        if return_z:
            return out, h
        return out


class ChannelDescriptorHead(nn.Module):
    """Channel-wise variant of DescriptorHead for --desc_mode channel.

    Input is the per-channel hidden [B, C, D] (variate tokens / per-channel
    patch means); a single channel-agnostic trunk + heads are applied to every
    channel (i.e. shared weights, exactly the same parameter count as the
    pooled DescriptorHead), producing per-channel descriptor outputs:
      cp_prob / cp_pos [B, C], drift / vol / slope logits [B, C, n_bins],
      spectral [B, C, 5].
    """

    def __init__(self, d_in, d_hidden=128, n_bins=5, dropout=0.1):
        super().__init__()
        self.head = DescriptorHead(d_in, d_hidden, n_bins, dropout)

    def forward(self, hidden, return_z=False):
        B, C, D = hidden.shape
        out, z = self.head(hidden.reshape(B * C, D), return_z=True)
        out = {
            'cp_prob': out['cp_prob'].reshape(B, C),
            'cp_pos': out['cp_pos'].reshape(B, C),
            'drift': out['drift'].reshape(B, C, -1),
            'vol': out['vol'].reshape(B, C, -1),
            'vol_near': out['vol_near'].reshape(B, C, -1),
            'vol_qr': out['vol_qr'].reshape(B, C),
            'slope': out['slope'].reshape(B, C, -1),
            'spectral': out['spectral'].reshape(B, C, -1),
        }
        if return_z:
            return out, z.reshape(B, C, -1)
        return out


class ValueMtpHead(nn.Module):
    """Value-level MTP control head: pooled hidden [B, D] -> flattened future
    window values [B, pred_len * C] (auxiliary MSE). Same capacity profile as
    DescriptorHead so the desc-vs-value comparison isolates the target, not
    the head."""

    def __init__(self, d_in, out_dim, d_hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, out_dim),
        )

    def forward(self, hidden):
        return self.net(hidden)
