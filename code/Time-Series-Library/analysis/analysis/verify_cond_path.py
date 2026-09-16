"""
Connectivity check for --desc_cond: verify the main forecast actually depends
on the descriptor representation z. Runs one random batch twice through a
cond-trained checkpoint: normally, and with the descriptor trunk output
zeroed (z = 0). The forecasts must differ for concat/film, and must be
identical for desc_cond=none (negative control).

Usage from the TSLib root:
  uv run python analysis/verify_cond_path.py \
    --ckpt checkpoints/<setting>/checkpoint.pth --model PatchTST \
    --desc_cond concat [--desc_mode channel] [--d_model 16 ...]
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DESC_BACKBONE = {'PatchTST': 'PatchSTDesc', 'iTransformer': 'iTransformerDesc'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--model', type=str, default='PatchTST', choices=list(DESC_BACKBONE))
    parser.add_argument('--desc_cond', type=str, default='concat', choices=['none', 'concat', 'film'])
    parser.add_argument('--desc_mode', type=str, default='pooled', choices=['pooled', 'channel'])
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--enc_in', type=int, default=7)
    args = parser.parse_args()

    import importlib
    margs = SimpleNamespace(
        task_name='long_term_forecast', seq_len=args.seq_len, label_len=args.label_len,
        pred_len=args.pred_len, enc_in=args.enc_in, dec_in=args.enc_in, c_out=args.enc_in,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, factor=args.factor, dropout=0.1, embed='timeF', freq='h',
        activation='gelu', aux_head='desc', desc_mode=args.desc_mode, desc_cond=args.desc_cond,
        num_coarse=2, num_fine=4, hcan_hidden=32,
    )
    module = importlib.import_module('models.' + DESC_BACKBONE[args.model])
    model = module.Model(margs).float()
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
    print('loaded (missing={}, unexpected={})'.format(len(missing), len(unexpected)))
    model.eval()

    torch.manual_seed(0)
    B = 4
    x = torch.randn(B, args.seq_len, args.enc_in)
    xm = torch.randn(B, args.seq_len, 4)
    dec_inp = torch.cat([torch.randn(B, args.label_len, args.enc_in),
                         torch.zeros(B, args.pred_len, args.enc_in)], dim=1)
    ym = torch.randn(B, args.label_len + args.pred_len, 4)

    with torch.no_grad():
        out1 = model(x, xm, dec_inp, ym)
        # zero z at the descriptor trunk output (also feeds the aux heads)
        trunk = model.aux_head.trunk if args.desc_mode == 'pooled' else model.aux_head.head.trunk
        hook = trunk.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        out2 = model(x, xm, dec_inp, ym)
        hook.remove()

    diff = (out1 - out2).abs().max().item()
    rel = ((out1 - out2).abs().mean() / out1.abs().mean()).item()
    print('desc_cond={} desc_mode={} max|diff|={:.6f} rel={:.4%}'.format(
        args.desc_cond, args.desc_mode, diff, rel))
    if args.desc_cond == 'none':
        assert diff == 0.0, 'cond=none but forecast changed when z was zeroed'
        print('OK: negative control (no dependency on z)')
    else:
        assert diff > 1e-4, 'cond={} but forecast does NOT depend on z'.format(args.desc_cond)
        print('OK: main forecast path depends on z (conditional path connected)')


if __name__ == '__main__':
    main()
