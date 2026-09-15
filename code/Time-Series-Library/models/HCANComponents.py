"""
HCAN components (AAAI 2025), ported from the official Informer-based
implementation in reference/HCAN to a backbone-agnostic, channel-wise form:

  - fit_bin_edges / produce_labels: per-channel equal-mass quantile bins of
    the (scaled) train target values, as in utils/Group_helper.py; labels are
    produced online from the future window (no dataloader surgery needed);
  - evidence_ce_loss: the Uncertainty-Aware Classifier's evidential
    (Dirichlet) cross-entropy with KL annealing (un_ce_loss in
    exp/exp_main.py, made device-agnostic);
  - symmetric_kl: the hierarchical consistency loss (HCL; loss_acl in the
    official code) between coarse logits and pairwise-merged fine logits;
  - HCANHead: coarse/fine classification + within-bin regression branches and
    the HAA non-local channel-fusion module, operating on per-channel
    features x [B, C, D] instead of the Informer's [B, C, pred_len] decoder
    output.

Deviations from the official code (see the port notes in the experiment
report):
  - logits are reshaped with an explicit (C, L, K) -> (L, C, K) permutation;
    the official view(B, -1, C, K) on a [B, C, L*K] tensor silently permutes
    (t, c) pairs unless pred_len == C. The clean version keeps the intended
    per-(time step, channel) semantics;
  - out-of-train-range values (y < train min or y >= train max) get bin label
    0 and are excluded from the within-bin regression, mirroring
    Group_helper's -1 padding.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def fit_bin_edges(values, num_bins):
    """Per-channel equal-mass quantile bin edges, following Group_helper.

    values: [T, C] np array (scaled train target values) -> edges [C, K+1]
    with edges[c, i] = sorted_values[int(i/K * (T-1)), c], edges[c, K] = max.
    """
    s = np.sort(values, axis=0)
    T = s.shape[0]
    idx = [int(i / num_bins * (T - 1)) for i in range(num_bins)]
    edges = np.concatenate([s[idx], s.max(axis=0, keepdims=True)], axis=0)  # [K+1, C]
    return torch.from_numpy(edges.T).float()                                # [C, K+1]


def produce_labels(y, edges):
    """Online bin labels for a future window.

    y: [..., C] torch tensor; edges: [C, K+1] -> (label [..., C] long,
    delta [..., C, K] float with -1 padding outside the active bin).
    """
    K = edges.shape[1] - 1
    left = edges[:, :-1]                                          # [C, K]
    right = edges[:, 1:]                                          # [C, K]
    label = (y.unsqueeze(-1) >= left).sum(dim=-1) - 1             # [..., C]
    valid = (y >= edges[:, 0]) & (y < edges[:, -1])
    label = label.clamp(0, K - 1)
    lo = torch.gather(left.expand(*y.shape, K), -1, label.unsqueeze(-1)).squeeze(-1)
    hi = torch.gather(right.expand(*y.shape, K), -1, label.unsqueeze(-1)).squeeze(-1)
    d = ((y - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
    delta = torch.full(y.shape + (K,), -1.0, device=y.device, dtype=y.dtype)
    delta.scatter_(-1, label.unsqueeze(-1), d.unsqueeze(-1))
    delta = torch.where(valid.unsqueeze(-1), delta,
                        torch.full_like(delta, -1.0))
    label = torch.where(valid, label, torch.zeros_like(label))
    return label, delta


def _dirichlet_kl(alpha, c):
    beta = torch.ones((1, c), device=alpha.device, dtype=alpha.dtype)
    S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    S_beta = torch.sum(beta, dim=-1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=-1, keepdim=True) + lnB + lnB_uni
    return kl


def evidence_ce_loss(labels, logits, num_classes, global_step, annealing_step):
    """un_ce_loss of the official code: expected-CE under the Dirichlet +
    annealed KL-to-uniform of the misleading evidence. labels/logits are
    flattened to [N] / [N, K] by the caller. Returns the scalar loss."""
    evidence = F.softplus(logits)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=-1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(labels, num_classes=num_classes).float()
    lam = 1 - E / S
    A = torch.sum(lam * label * (torch.digamma(S) - torch.digamma(alpha)), dim=-1, keepdim=True)
    annealing_coef = min(1.0, float(global_step) / float(annealing_step))
    alp = E * (1 - label) + 1
    B = annealing_coef * _dirichlet_kl(alp, num_classes)
    return torch.mean(A + B)


def symmetric_kl(p, q):
    """HCL/ACL consistency loss: symmetric KL between two probability maps."""
    kl_pq = F.kl_div(torch.log(p + 1e-12), q, reduction='batchmean')
    kl_qp = F.kl_div(torch.log(q + 1e-12), p, reduction='batchmean')
    return 0.5 * (kl_pq + kl_qp)


class HCANHead(nn.Module):
    """Coarse/fine bin-classification branches + HAA channel fusion on
    per-channel features x [B, C, D].

    Returns (z, aux): z [B, C, D] is the HAA-fused feature meant to feed the
    backbone's forecast path (residual, as in the official direct predictor);
    aux holds coarse_logit / coarse_delta / fine_logit / fine_delta, each
    [B, pred_len, C, K].
    """

    def __init__(self, d_in, pred_len, d_l=512, num_coarse=2, num_fine=4):
        super().__init__()
        assert num_fine == 2 * num_coarse, 'HCL merges adjacent fine bins in pairs'
        self.pred_len = pred_len
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        # direct-prediction (g) branch
        self.g_proj = nn.Linear(d_in, d_l)
        self.g_proj2 = nn.Linear(d_l, d_in)
        # coarse branch
        self.c_proj = nn.Linear(d_in, d_l)
        self.coarse_predictor = nn.Linear(d_l, pred_len * num_coarse)
        self.coarse_classify = nn.Linear(d_l, pred_len * num_coarse)
        # fine branch
        self.f_proj = nn.Linear(d_in, d_l)
        self.fine_predictor = nn.Linear(d_l, pred_len * num_fine)
        self.fine_classify = nn.Linear(d_l, pred_len * num_fine)

    def _split(self, t, K):
        # [B, C, L*K] -> [B, L, C, K] (explicit permutation; see module docstring)
        B, C, _ = t.shape
        return t.view(B, C, self.pred_len, K).permute(0, 2, 1, 3).contiguous()

    def forward(self, x):
        c = self.c_proj(x)                                        # [B, C, d_l]
        f = self.f_proj(x)
        g = self.g_proj(x)
        aux = {
            'coarse_logit': self._split(self.coarse_classify(c), self.num_coarse),
            'coarse_delta': self._split(self.coarse_predictor(c), self.num_coarse),
            'fine_logit': self._split(self.fine_classify(f), self.num_fine),
            'fine_delta': self._split(self.fine_predictor(f), self.num_fine),
        }
        # HAA non-local channel fusion
        att = F.softmax(torch.matmul(c, f.transpose(1, 2)), dim=-1)   # [B, C, C]
        y = torch.matmul(att, g)                                      # [B, C, d_l]
        z = self.g_proj2(y) + x                                       # residual
        return z, aux
