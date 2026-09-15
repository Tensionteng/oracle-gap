"""
Sanity assertions for utils/region_focal.py (CPU, no dataset needed).

  (a) yhat == y  -> loss == 0 (also with lambda_base > 0);
  (b) yhat == m (full-anchor prediction), d != 0 -> o ~= 0 and u takes the
      soft-gate value 0.5 * sigmoid(1/tau2) ~= 0.5 (s = sigmoid(0) = 0.5 sits
      at its midpoint; see the note in utils/region_focal.py), and u >> o;
  (c) rf_detach 0/1 changes the gradient graph (w.requires_grad) but not the
      forward value;
  (d) seasonal anchor on freq='h' (period 24): m[b, t, c] == x[b, seq_len-24
      + t%24, c], shapes aligned [B, L, C].

Run: uv run python analysis/sanity_regionfocal.py
"""
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.region_focal import RegionFocalLoss


def main():
    torch.manual_seed(0)
    B, L, C = 4, 96, 7

    y = torch.randn(B, L, C)
    x_ctx = torch.randn(B, 96, C)

    # (a) perfect prediction -> zero loss
    crit = RegionFocalLoss(au=1.0)
    m = crit.compute_anchor(x_ctx, L)
    la = crit(y, y, anchor=m)
    assert la.item() == 0.0, la.item()
    crit_b = RegionFocalLoss(au=1.0, lambda_base=1.0)
    assert crit_b(y, y, anchor=m).item() == 0.0
    print('(a) yhat==y -> loss=0 OK (also with lambda_base=1)')

    # (b) full-anchor prediction: U channel dominates, o ~ 0
    yhat = m.clone()
    d = y - m
    p = yhat - m
    s, u, o = crit.memberships(p, d)
    mask = d.abs() > 1e-3
    u_med = u[mask].median().item()
    o_max = o[mask].max().item()
    # soft-gate formula value: s=sigmoid(0)=0.5, r=0 -> u = 0.5*sigmoid(1/0.1)
    expected_u = 0.5 * torch.sigmoid(torch.tensor(1.0 / 0.1)).item()
    assert abs(u_med - expected_u) < 1e-3, (u_med, expected_u)
    assert o_max < 1e-4, o_max
    print('(b) yhat==m -> u={:.4f} (=0.5*sigmoid(1/tau2), soft-gate midpoint; o={:.2e}~0) OK'.format(u_med, o_max))

    # (c) detach toggle: same forward value, different grad graph
    yhat_g = (y + 0.3 * torch.randn(B, L, C)).requires_grad_(True)
    c_det = RegionFocalLoss(au=1.0, detach=1)
    c_grd = RegionFocalLoss(au=1.0, detach=0)
    l1 = c_det(yhat_g, y, anchor=m)
    l2 = c_grd(yhat_g, y, anchor=m)
    assert abs(l1.item() - l2.item()) < 1e-6, (l1.item(), l2.item())
    g1 = torch.autograd.grad(l1, yhat_g, retain_graph=True)[0]
    g2 = torch.autograd.grad(l2, yhat_g)[0]
    # detached: dL/dyhat = 2w(yhat-y); non-detached adds dw/dyhat terms
    w_det = 1.0 + c_det.au * u  # recompute weights of this config for reference
    assert not torch.allclose(g1, g2), 'detach=0 must change gradients'
    print('(c) detach toggle: forward equal ({:.6f}), gradients differ (max|dg|={:.4f}) OK'.format(
        l1.item(), (g1 - g2).abs().max().item()))

    # (d) seasonal anchor on freq='h'
    crit_s = RegionFocalLoss(au=1.0, anchor='seasonal', freq='h', seq_len=96)
    ms = crit_s.compute_anchor(x_ctx, L)
    assert ms.shape == (B, L, C)
    period = 24
    idx = 96 - period + (torch.arange(L) % period)
    assert torch.equal(ms, x_ctx[:, idx, :]), 'seasonal anchor misaligned'
    print('(d) seasonal anchor: shape {} aligned to period-24 phase OK'.format(tuple(ms.shape)))

    print('ALL SANITY CHECKS PASSED')


if __name__ == '__main__':
    main()
