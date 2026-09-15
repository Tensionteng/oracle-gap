"""
Per-layer gradient-flow analysis for the descriptor auxiliary head.

On a desc-trained checkpoint, computes for one (or a few) training batches:
  L_task = MSE(forecast, future),  L_aux = descriptor auxiliary loss,
and the per-layer gradient norms |dL/dtheta_l| for both losses, reporting the
ratio |dL_aux| / |dL_task| per layer group. This answers where the auxiliary
signal actually lands and how strong it is relative to the task gradient
(which in turn explains aux_weight sensitivity).

Usage from the TSLib root:
  CUDA_VISIBLE_DEVICES=6 uv run python analysis/gradient_flow.py \
    --ckpt checkpoints/<desc_setting>/checkpoint.pth --model PatchTST --data ETTh1 \
    --e_layers 1 --n_heads 2 --d_model 512 --d_ff 2048 --device cuda:0
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_dict
from utils.compute_descriptor_stats import ensure_descriptor_stats
from utils.descriptor_labels import compute_descriptor_targets

DESC_BACKBONE = {'PatchTST': 'PatchSTDesc', 'iTransformer': 'iTransformerDesc', 'DLinear': 'DLinearDesc'}


def layer_group(name):
    if 'patch_embedding' in name or 'enc_embedding' in name:
        return 'embedding'
    if 'encoder.attn_layers' in name:
        return 'encoder.layer' + name.split('encoder.attn_layers.')[1].split('.')[0]
    if 'encoder.norm' in name:
        return 'encoder.norm'
    if name.startswith('aux_head'):
        return 'aux_head'
    if name.startswith('hidden_proj'):
        return 'hidden_proj'
    if name.startswith('head') or name.startswith('projection') or 'Linear_' in name:
        return 'forecast_head'
    return 'other'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--setting', type=str, default=None)
    parser.add_argument('--model', type=str, default='PatchTST', choices=list(DESC_BACKBONE))
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
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--desc_k', type=int, default=24)
    parser.add_argument('--desc_mode', type=str, default='pooled')
    parser.add_argument('--n_batches', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path or (args.data + '.csv')
    args.data_path = data_path
    setting = args.setting or os.path.basename(os.path.dirname(args.ckpt))
    out_path = args.out or os.path.join('analysis', 'results', 'gradflow_{}.json'.format(setting))

    margs = SimpleNamespace(
        task_name='long_term_forecast', is_training=1, model_id='gradflow', model=args.model,
        data=args.data, root_path=args.root_path, data_path=data_path, features=args.features,
        target=args.target, freq=args.freq, checkpoints='./checkpoints/', max_train_windows=-1,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        seasonal_patterns='Monthly', inverse=False,
        enc_in=args.enc_in, dec_in=args.enc_in, c_out=args.enc_in,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, moving_avg=args.moving_avg, factor=args.factor, dropout=args.dropout,
        embed=args.embed, activation=args.activation,
        batch_size=args.batch_size, num_workers=args.num_workers,
        augmentation_ratio=0, desc_k=args.desc_k, aux_head='desc', desc_mode=args.desc_mode,
    )
    device = torch.device(args.device)
    stats = ensure_descriptor_stats(margs)
    stats = {k: torch.from_numpy(np.asarray(v)).float().to(device)
             for k, v in stats.items() if v.dtype != object}

    import importlib
    module = importlib.import_module('models.' + DESC_BACKBONE[args.model])
    model = module.Model(margs).float().to(device)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
    print('loaded {} (missing={}, unexpected={})'.format(args.ckpt, len(missing), len(unexpected)))
    model.train()

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    data_set = Data(args=margs, root_path=args.root_path, data_path=data_path, flag='train',
                    size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
                    target=args.target, timeenc=timeenc, freq=args.freq,
                    seasonal_patterns=margs.seasonal_patterns)
    loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    params = [p for _, p in model.named_parameters()]
    names = [n for n, _ in model.named_parameters()]
    acc_task = {n: 0.0 for n in names}
    acc_aux = {n: 0.0 for n in names}
    task_losses, aux_losses = [], []

    for bi, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
        if bi >= args.n_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        outputs, aux = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_hidden=True)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        future = batch_y[:, -args.pred_len:, f_dim:]
        l_task = F.mse_loss(outputs, future)

        targets = compute_descriptor_targets(future, args.desc_k, stats,
                                             reduce=None if args.desc_mode == 'channel' else 'mean')
        n_bins = aux['drift'].shape[-1]
        l_aux = F.binary_cross_entropy_with_logits(aux['cp_prob'], targets['cp_prob']) \
            + F.smooth_l1_loss(aux['cp_pos'], targets['cp_pos']) \
            + F.cross_entropy(aux['drift'].reshape(-1, n_bins), targets['drift_cls'].reshape(-1)) \
            + F.cross_entropy(aux['vol'].reshape(-1, n_bins), targets['vol_cls'].reshape(-1)) \
            + F.cross_entropy(aux['slope'].reshape(-1, n_bins), targets['slope_cls'].reshape(-1)) \
            + F.mse_loss(aux['spectral'], targets['spectral'])

        g_task = torch.autograd.grad(l_task, params, retain_graph=True, allow_unused=True)
        g_aux = torch.autograd.grad(l_aux, params, retain_graph=False, allow_unused=True)
        for n, gt, ga in zip(names, g_task, g_aux):
            acc_task[n] += float(gt.norm()) if gt is not None else 0.0
            acc_aux[n] += float(ga.norm()) if ga is not None else 0.0
        task_losses.append(float(l_task))
        aux_losses.append(float(l_aux))

    groups = {}
    for n in names:
        g = layer_group(n)
        groups.setdefault(g, {'task': 0.0, 'aux': 0.0})
        groups[g]['task'] += acc_task[n] ** 2
        groups[g]['aux'] += acc_aux[n] ** 2
    rows = []
    order = sorted(groups, key=lambda g: (g != 'embedding', g))
    for g in order:
        t, a = groups[g]['task'] ** 0.5, groups[g]['aux'] ** 0.5
        rows.append({'layer': g, 'grad_task': t, 'grad_aux': a,
                     'ratio_aux_over_task': (a / t) if t > 0 else None})
    tot_t = sum(v['task'] for v in groups.values()) ** 0.5
    tot_a = sum(v['aux'] for v in groups.values()) ** 0.5

    print('\nper-layer gradient norms (avg over {} batches): L_task={:.4f} L_aux={:.4f}'.format(
        len(task_losses), np.mean(task_losses), np.mean(aux_losses)))
    print('| {:<16} | {:>12} | {:>12} | {:>10} |'.format('layer', '|g_task|', '|g_aux|', 'aux/task'))
    print('|' + '-' * 18 + '|' + '-' * 14 + '|' + '-' * 14 + '|' + '-' * 12 + '|')
    for r in rows:
        ratio = '{:.4f}'.format(r['ratio_aux_over_task']) if r['ratio_aux_over_task'] is not None else 'n/a'
        print('| {:<16} | {:>12.4f} | {:>12.4f} | {:>10} |'.format(
            r['layer'], r['grad_task'], r['grad_aux'], ratio))
    print('| {:<16} | {:>12.4f} | {:>12.4f} | {:>10.4f} |'.format(
        'TOTAL', tot_t, tot_a, tot_a / tot_t if tot_t > 0 else float('nan')))

    result = {'setting': setting, 'model': args.model, 'data': args.data, 'ckpt': args.ckpt,
              'n_batches': len(task_losses), 'batch_size': args.batch_size,
              'l_task_mean': float(np.mean(task_losses)), 'l_aux_mean': float(np.mean(aux_losses)),
              'per_layer': rows,
              'total': {'grad_task': tot_t, 'grad_aux': tot_a,
                        'ratio_aux_over_task': tot_a / tot_t if tot_t > 0 else None}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print('written: {}'.format(out_path))


if __name__ == '__main__':
    main()
