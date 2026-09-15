#!/usr/bin/env python
"""Smoke tests for models/PatchTSTGated.py:
1. numerical equivalence of variant='none' (SDPA) with original PatchTST (einsum softmax)
   using a real LBv2 checkpoint;
2. forward+backward memory probe at electricity scale (enc_in=321, sl=2880).
"""
import argparse
import glob
import torch

from models import PatchTST, PatchTSTGated


def make_cfg(seq_len, enc_in, variant='none'):
    return argparse.Namespace(
        task_name='long_term_forecast', seq_len=seq_len, pred_len=96,
        d_model=512, n_heads=8, e_layers=2, d_ff=2048, dropout=0.1,
        activation='gelu', enc_in=enc_in, factor=3,
        attn_variant=variant, recmask_window=336)


def test_equivalence():
    ckpt = glob.glob('checkpoints/long_term_forecast_LBv2_exchange_rate_PatchTST_96_s2021_*/checkpoint.pth')[0]
    cfg = make_cfg(96, 8)
    m_ref = PatchTST.Model(cfg).float().eval()
    m_new = PatchTSTGated.Model(cfg).float().eval()
    sd = torch.load(ckpt, map_location='cpu')
    m_ref.load_state_dict(sd)
    missing, unexpected = m_new.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    # denbias-only params may be missing; for variant 'none' nothing should be missing
    assert not missing, missing
    x = torch.randn(4, 96, 8)
    xm = torch.randn(4, 96, 4)
    with torch.no_grad():
        o_ref = m_ref(x, xm, None, None)
        o_new = m_new(x, xm, None, None)
    diff = (o_ref - o_new).abs().max().item()
    print(f'[equiv] max|out_ref-out_new| = {diff:.3e}')
    assert diff < 1e-4, diff
    print('[equiv] PASS')


def test_recmask_noop_at_sl96():
    cfg_a = make_cfg(96, 8, 'none')
    cfg_b = make_cfg(96, 8, 'recmask')
    m_a = PatchTSTGated.Model(cfg_a).float().eval()
    m_b = PatchTSTGated.Model(cfg_b).float().eval()
    m_b.load_state_dict(m_a.state_dict(), strict=False)
    x = torch.randn(4, 96, 8)
    with torch.no_grad():
        d = (m_a(x, None, None, None) - m_b(x, None, None, None)).abs().max().item()
    print(f'[recmask@sl96 noop] max diff = {d:.3e}')
    assert d < 1e-5
    print('[recmask@sl96 noop] PASS')


def test_mask_and_denbias_shapes():
    for variant in ['denbias', 'recmask']:
        cfg = make_cfg(2880, 8, variant)
        m = PatchTSTGated.Model(cfg).float()
        # enable output_attention path
        for layer in m.encoder.attn_layers:
            layer.attention.inner_attention.output_attention = True
        x = torch.randn(2, 2880, 8)
        out = m(x, None, None, None)
        assert out.shape == (2, 96, 8), out.shape
        out.sum().backward()
        if variant == 'denbias':
            g = m.encoder.attn_layers[0].attention.inner_attention.null_logit.grad
            assert g is not None and torch.isfinite(g).all()
        print(f'[{variant}] fwd/bwd OK, out {tuple(out.shape)}')
    print('[shapes] PASS')


def mem_probe():
    if not torch.cuda.is_available():
        print('[mem] no cuda, skip')
        return
    dev = 'cuda:0'
    for variant in ['none', 'denbias', 'recmask']:
        for bs in [16, 8, 4]:
            cfg = make_cfg(2880, 321, variant)
            m = PatchTSTGated.Model(cfg).float().to(dev)
            m.train()
            opt = torch.optim.Adam(m.parameters(), lr=1e-4)
            x = torch.randn(bs, 2880, 321, device=dev)
            y = torch.randn(bs, 96, 321, device=dev)
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                for _ in range(2):
                    opt.zero_grad()
                    out = m(x, None, None, None)
                    loss = ((out - y) ** 2).mean()
                    loss.backward()
                    opt.step()
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f'[mem] {variant} bs={bs}: peak {peak:.1f} GiB OK')
            except torch.OutOfMemoryError:
                print(f'[mem] {variant} bs={bs}: OOM')
            del m, opt, x, y
            torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_equivalence()
    test_recmask_noop_at_sl96()
    test_mask_and_denbias_shapes()
    mem_probe()
