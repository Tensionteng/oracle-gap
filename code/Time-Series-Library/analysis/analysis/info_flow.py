"""
Information-flow quantification for cascade-conditioned models
(--desc_cond concat/film): how much does the main forecast actually use the
descriptor representation z?

Runs the test set three times per checkpoint:
  clean : as-is;
  zero  : descriptor trunk output z replaced by zeros;
  perm  : z permuted across the batch dimension (right shape, wrong window);
and reports test MSE under each intervention plus deltas vs clean. A
desc_cond=none checkpoint serves as negative control (delta must be ~0).

For parallel (non-cond) models the "does descriptor info reach the main head
input" question is covered by analysis/probe_descriptor.py --probe_layer all:
for these encoder-only backbones the main head input IS the last layer's
pooled output (key 'last' in the per-layer probe JSON).

Usage from the TSLib root:
  CUDA_VISIBLE_DEVICES=6 uv run python analysis/info_flow.py \
    --ckpt checkpoints/<cond_setting>/checkpoint.pth --model PatchTST \
    --desc_cond concat --e_layers 1 --n_heads 2 --d_model 512 --d_ff 2048 \
    --device cuda:0
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_dict

DESC_BACKBONE = {'PatchTST': 'PatchSTDesc', 'iTransformer': 'iTransformerDesc'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--setting', type=str, default=None)
    parser.add_argument('--model', type=str, default='PatchTST', choices=list(DESC_BACKBONE))
    parser.add_argument('--desc_cond', type=str, required=True, choices=['none', 'concat', 'film'])
    parser.add_argument('--desc_mode', type=str, default='pooled', choices=['pooled', 'channel'])
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path or (args.data + '.csv')
    setting = args.setting or os.path.basename(os.path.dirname(args.ckpt))
    out_path = args.out or os.path.join('analysis', 'results', 'infoflow_{}.json'.format(setting))

    margs = SimpleNamespace(
        task_name='long_term_forecast', is_training=1, model_id='infoflow', model=args.model,
        data=args.data, root_path=args.root_path, data_path=data_path, features=args.features,
        target=args.target, freq=args.freq, checkpoints='./checkpoints/', max_train_windows=-1,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        seasonal_patterns='Monthly', inverse=False,
        enc_in=args.enc_in, dec_in=args.enc_in, c_out=args.enc_in,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, moving_avg=25, factor=args.factor, dropout=args.dropout,
        embed=args.embed, activation=args.activation,
        batch_size=args.batch_size, num_workers=args.num_workers,
        augmentation_ratio=0, desc_k=24, aux_head='desc', desc_mode=args.desc_mode,
        desc_cond=args.desc_cond,
    )
    device = torch.device(args.device)

    import importlib
    module = importlib.import_module('models.' + DESC_BACKBONE[args.model])
    model = module.Model(margs).float().to(device)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
    print('loaded {} (missing={}, unexpected={})'.format(args.ckpt, len(missing), len(unexpected)))
    model.eval()

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    data_set = Data(args=margs, root_path=args.root_path, data_path=data_path, flag='test',
                    size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                    target=args.target, timeenc=timeenc, freq=args.freq,
                    seasonal_patterns=margs.seasonal_patterns)
    loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    trunk = model.aux_head.trunk if args.desc_mode == 'pooled' else model.aux_head.head.trunk

    def run(mode):
        hook = None
        if mode == 'zero':
            hook = trunk.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        elif mode == 'perm':
            def perm_hook(m, i, o):
                return o[torch.randperm(o.shape[0], device=o.device)]
            hook = trunk.register_forward_hook(perm_hook)
        sq, n = [], 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                future = batch_y[:, -args.pred_len:, f_dim:]
                sq.append(((outputs - future) ** 2).mean(dim=(1, 2)).cpu().numpy())
        if hook is not None:
            hook.remove()
        per_window = np.concatenate(sq)
        return per_window

    mse = {mode: run(mode) for mode in ['clean', 'zero', 'perm']}
    out = {'setting': setting, 'model': args.model, 'data': args.data,
           'desc_cond': args.desc_cond, 'desc_mode': args.desc_mode, 'ckpt': args.ckpt,
           'n_windows': int(len(mse['clean'])),
           'mse_clean': float(mse['clean'].mean()),
           'mse_zero': float(mse['zero'].mean()),
           'mse_perm': float(mse['perm'].mean()),
           'delta_zero': float(mse['zero'].mean() - mse['clean'].mean()),
           'delta_perm': float(mse['perm'].mean() - mse['clean'].mean()),
           'delta_perm_rel': float((mse['perm'].mean() - mse['clean'].mean()) / mse['clean'].mean())}
    print(json.dumps({k: v for k, v in out.items() if k.startswith('mse') or k.startswith('delta')},
                     indent=2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('written: {}'.format(out_path))


if __name__ == '__main__':
    main()
