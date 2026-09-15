#!/usr/bin/env python
"""Stage-2 TSFM minimal loop (mtp4ts idea.md section 5.2):
full fine-tuning of Chronos-Bolt on ETT with an optional future-descriptor
auxiliary head.

Protocol alignment with Time-Series-Library (TSLib) so numbers are comparable
to the task-model main table:
  * train/val/test borders, StandardScaler fit on the train split and the
    96->96 window slicing replicate data_provider/data_loader.py
    (Dataset_ETT_hour / Dataset_ETT_minute, features=M, scale=True);
  * descriptor labels are computed with TSLib's utils/descriptor_labels.py
    (imported via sys.path) in channel mode (reduce=None); train-split bin
    edges / z-score stats are loaded from TSLib descriptor_cache
    (<key>_pl<pred_len>[_mtw<N>].npz) and recomputed with the same quantile
    logic only if the cache entry is missing;
  * test MSE/MAE are computed in the scaler space (TSLib default --inverse 0)
    on the median (0.5 quantile) forecast.

Chronos-Bolt specifics:
  * univariate per channel: a [B, L, C] batch becomes B*C instances;
  * native prediction_length is 64; the quantile loss is trained on the first
    64 steps of the 96-step future, evaluation uses the official
    ChronosBoltPipeline rollout to 96 (quantile-expansion continuation), the
    same path for zero-shot and fine-tuned checkpoints;
  * aux head: encoder last-layer hidden state [B*C, n_tokens, d_model] is
    masked-mean-pooled over patch tokens and fed to a shallow MLP that outputs
    6 descriptor heads (cp_prob logit, cp_pos, drift/vol/slope 5-bin logits,
    5-dim spectral), trained with the same 6-term loss as TSLib
    exp/exp_descriptor_forecast.py.

The script prints a final 'mse:..., mae:...' line so the experiment scheduler
recognizes the job as finished.
"""
import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
TSLIB = os.path.join(ROOT, '..', 'Time-Series-Library')
sys.path.insert(0, TSLIB)  # for utils.descriptor_labels (pure torch, no heavy deps)

from utils.descriptor_labels import (compute_descriptor_targets, cusum_score_pos,
                                     mid_window_stats, spectral_features)
from utils.region_focal import RegionFocalLoss

MODELS_DIR = os.path.join(ROOT, '..', 'models')
LOCAL_CACHE = os.path.join(ROOT, 'descriptor_cache')


# --------------------------------------------------------------------------- #
# data (TSLib Dataset_ETT_hour / Dataset_ETT_minute protocol)
# --------------------------------------------------------------------------- #
class ETTWindows(Dataset):
    """Replica of TSLib's windowing for features=M.

    data='ETT*' replicates Dataset_ETT_hour / Dataset_ETT_minute borders;
    data='custom' replicates Dataset_Custom's 7/1/2 borders (any wide CSV with
    a date column; all non-date columns are channels, OT included).

    Returns (context [seq_len, C], future [pred_len, C]) per window; Chronos
    sees each channel as a separate univariate instance downstream.
    """

    def __init__(self, root_path, data_path, flag='train', seq_len=96, pred_len=96,
                 mult=None, max_windows=-1, data='ETTh1'):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        assert flag in ('train', 'val', 'test')
        self.seq_len, self.pred_len = seq_len, pred_len
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        if data == 'custom':
            n = len(df_raw)
            num_train = int(n * 0.7)
            num_test = int(n * 0.2)
            set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
            border1s = [0, num_train - seq_len, n - num_test - seq_len]
            border2s = [num_train, num_train + (n - num_train - num_test), n]
            b1, b2 = border1s[set_type], border2s[set_type]
            n_train = num_train
        else:
            if mult is None:  # ETTm* are 15-min sampled: 4x the hourly borders
                mult = 4 if os.path.basename(data_path).startswith('ETTm') else 1
            n_train = 12 * 30 * 24 * mult
            n_val = 4 * 30 * 24 * mult
            n_test = 4 * 30 * 24 * mult
            set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
            border1s = [0, n_train - seq_len, n_train + n_val - seq_len]
            border2s = [n_train, n_train + n_val, n_train + n_val + n_test]
            b1, b2 = border1s[set_type], border2s[set_type]

        df_data = df_raw[df_raw.columns[1:]]  # features=M: all non-date columns
        scaler = StandardScaler()
        scaler.fit(df_data.values[0:n_train])
        data = scaler.transform(df_data.values)
        self.data = data[b1:b2].astype(np.float32)
        self.n_channels = self.data.shape[1]

        n_total = len(self.data) - seq_len - pred_len + 1
        if flag == 'train' and max_windows is not None and 0 < max_windows < n_total:
            # same evenly-spaced deterministic subsample as TSLib _train_window_subsample
            self.indices = np.unique(np.linspace(0, n_total - 1, max_windows).round().astype(np.int64))
        else:
            self.indices = np.arange(n_total, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = int(self.indices[i])
        context = self.data[s:s + self.seq_len]
        future = self.data[s + self.seq_len:s + self.seq_len + self.pred_len]
        return torch.from_numpy(context), torch.from_numpy(future)


# --------------------------------------------------------------------------- #
# descriptor stats (TSLib utils/compute_descriptor_stats.py logic)
# --------------------------------------------------------------------------- #
def stats_cache_key(args):
    key = os.path.splitext(os.path.basename(args.data_path))[0]
    suffix = '{}_pl{}'.format(key, args.pred_len)
    if args.max_train_windows is not None and args.max_train_windows > 0:
        suffix += '_mtw{}'.format(args.max_train_windows)
    return suffix


def load_or_compute_stats(args, train_ds):
    suffix = stats_cache_key(args)
    for cache_dir in (os.path.join(TSLIB, 'descriptor_cache'), LOCAL_CACHE):
        path = os.path.join(cache_dir, suffix + '.npz')
        if os.path.exists(path):
            stats = dict(np.load(path))
            print('descriptor stats loaded from {}'.format(path))
            return stats

    # fallback: recompute with TSLib's quantile logic over the same train windows
    print('descriptor stats cache miss for {}, computing on train windows'.format(suffix))
    futures = []
    loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)
    with torch.no_grad():
        for _, future in loader:
            futures.append(future.float())
    future = torch.cat(futures, dim=0)  # [N, pred_len, C]
    score, _ = cusum_score_pos(future, args.desc_k)
    drift, vol, slope = mid_window_stats(future)
    spec = spectral_features(future)
    drift_c, vol_c, slope_c = mid_window_stats(future, reduce=None)
    spec_c = spectral_features(future, reduce=None)
    q = [0.1, 0.3, 0.7, 0.9]
    stats = {
        'cp_score_q': np.quantile(score.numpy(), [0.1, 0.25, 0.5, 0.75, 0.9]).astype(np.float32),
        'drift_edges': np.quantile(drift.numpy(), q).astype(np.float32),
        'vol_edges': np.quantile(vol.numpy(), q).astype(np.float32),
        'slope_edges': np.quantile(slope.numpy(), q).astype(np.float32),
        'spec_mean': spec.mean(dim=0).numpy().astype(np.float32),
        'spec_std': spec.std(dim=0, unbiased=False).numpy().astype(np.float32),
        'drift_edges_ch': np.quantile(drift_c.reshape(-1).numpy(), q).astype(np.float32),
        'vol_edges_ch': np.quantile(vol_c.reshape(-1).numpy(), q).astype(np.float32),
        'slope_edges_ch': np.quantile(slope_c.reshape(-1).numpy(), q).astype(np.float32),
        'spec_mean_ch': spec_c.reshape(-1, spec_c.shape[-1]).mean(dim=0).numpy().astype(np.float32),
        'spec_std_ch': spec_c.reshape(-1, spec_c.shape[-1]).std(dim=0, unbiased=False).numpy().astype(np.float32),
    }
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    np.savez(os.path.join(LOCAL_CACHE, suffix + '.npz'), **stats)
    print('descriptor stats computed on {} train windows -> {}'.format(future.shape[0], LOCAL_CACHE))
    return stats


# --------------------------------------------------------------------------- #
# descriptor head
# --------------------------------------------------------------------------- #
class DescriptorHead(nn.Module):
    """Shallow MLP on the pooled encoder hidden state -> 6 descriptor outputs."""

    N_BINS = 5
    N_SPEC = 5

    def __init__(self, d_model, hidden=256):
        super().__init__()
        n_out = 1 + 1 + 3 * self.N_BINS + self.N_SPEC
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, pooled):
        out = self.net(pooled)
        b = self.N_BINS
        return {
            'cp_prob': out[:, 0],
            'cp_pos': out[:, 1],
            'drift': out[:, 2:2 + b],
            'vol': out[:, 2 + b:2 + 2 * b],
            'slope': out[:, 2 + 2 * b:2 + 3 * b],
            'spectral': out[:, 2 + 3 * b:2 + 3 * b + self.N_SPEC],
        }


def descriptor_aux_loss(aux, future, desc_k, stats_t):
    """6-term descriptor loss, channel mode, mirroring
    exp/exp_descriptor_forecast.py::_aux_loss (aux_head=desc, desc_mode=channel).

    aux outputs are [B*C, ...]; future is [B, pred_len, C] in scaler space.
    """
    targets = compute_descriptor_targets(future, desc_k, stats_t, reduce=None)
    n_bins = aux['drift'].shape[-1]
    loss = F.binary_cross_entropy_with_logits(aux['cp_prob'], targets['cp_prob'].reshape(-1))
    loss = loss + F.smooth_l1_loss(aux['cp_pos'], targets['cp_pos'].reshape(-1))
    loss = loss + F.cross_entropy(aux['drift'], targets['drift_cls'].reshape(-1))
    loss = loss + F.cross_entropy(aux['vol'], targets['vol_cls'].reshape(-1))
    loss = loss + F.cross_entropy(aux['slope'], targets['slope_cls'].reshape(-1))
    loss = loss + F.mse_loss(aux['spectral'], targets['spectral'].reshape(-1, targets['spectral'].shape[-1]))
    return loss


# --------------------------------------------------------------------------- #
# Chronos-Bolt wrapper with encoder-hidden capture
# --------------------------------------------------------------------------- #
class BoltWithAux(nn.Module):
    """ChronosBoltModelForForecasting + optional descriptor head.

    The descriptor head reads the encoder's last-layer hidden state captured
    by a forward hook on model.encoder: [B_inst, n_tokens, d_model] pooled by
    masked mean over tokens -> DescriptorHead.
    """

    def __init__(self, model_path, aux_head='none', device='cpu'):
        super().__init__()
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained(model_path, device_map=device)
        self.bolt = pipeline.model
        self.aux_head = aux_head
        self._enc_hidden = None
        self._enc_mask = None
        if aux_head == 'desc':
            d_model = self.bolt.config.d_model
            self.head = DescriptorHead(d_model).to(device)
        self.bolt.encoder.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self.bolt.encoder.register_forward_hook(self._post_hook)

    def _pre_hook(self, module, args, kwargs):
        self._enc_mask = kwargs.get('attention_mask')

    def _post_hook(self, module, args, output):
        self._enc_hidden = output.last_hidden_state if hasattr(output, 'last_hidden_state') \
            else output[0]

    def pooled_hidden(self):
        h, m = self._enc_hidden, self._enc_mask
        if h is None:
            raise RuntimeError('encoder hook did not fire')
        if m is None:
            return h.mean(dim=1)
        m = m.to(h.dtype).unsqueeze(-1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(self, context, target=None):
        """context [B_inst, L]; target [B_inst, <=64] for the native quantile loss."""
        out = self.bolt(context=context, target=target)
        aux = self.head(self.pooled_hidden()) if self.aux_head == 'desc' else None
        return out, aux


# --------------------------------------------------------------------------- #
# region-weighted pinball loss (regionfocal on the native quantile loss)
# --------------------------------------------------------------------------- #
def regionfocal_pinball(quantile_preds, target, context, quantiles, rf):
    """quantile_preds [B, Q, H], target [B, H], context [B, L] -> scalar loss.

    Per-point pinball terms for all quantile levels are weighted by the
    regionfocal weights (ctxmean anchor, soft gates, detached); the region is
    judged with the median (0.5) quantile prediction, exactly the TSLib
    utils/region_focal.py semantics."""
    med = quantiles.index(0.5)
    anchor = context.mean(dim=1, keepdim=True)             # [B, 1]
    d = target - anchor
    p = quantile_preds[:, med, :] - anchor
    w = rf.point_weights(p, d).detach()                    # [B, H]
    q = torch.tensor(quantiles, device=target.device, dtype=target.dtype).view(1, -1, 1)
    err = target.unsqueeze(1) - quantile_preds             # [B, Q, H]
    rho = torch.maximum(q * err, (q - 1.0) * err)
    return (w.unsqueeze(1) * rho).mean()


# --------------------------------------------------------------------------- #
# evaluation: official ChronosBoltPipeline rollout (identical for zero-shot
# and fine-tuned checkpoints); for pred_len > 64 the pipeline's built-in
# quantile-expansion continuation is used, so numbers match the standard
# Chronos-Bolt evaluation path.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_median(model_wrapper, context, pred_len, chunk=2048, use_amp=False):
    """context [N, seq_len] (cpu) -> median forecast [N, pred_len] (cpu)."""
    from chronos import ChronosBoltPipeline
    model_wrapper.eval()
    pipeline = ChronosBoltPipeline(model_wrapper.bolt)
    outs = []
    for s in range(0, context.shape[0], chunk):
        ctx = context[s:s + chunk]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=bool(use_amp) and next(model_wrapper.parameters()).is_cuda):
            _, median = pipeline.predict_quantiles(ctx, prediction_length=pred_len,
                                                   quantile_levels=[0.5])
        outs.append(median.float().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def evaluate(model_wrapper, dataset, args, device, tag):
    n = len(dataset) if args.max_test_windows <= 0 else min(args.max_test_windows, len(dataset))
    context = torch.stack([dataset[i][0] for i in range(n)])       # [N, L, C]
    future = torch.stack([dataset[i][1] for i in range(n)])        # [N, H, C]
    N, L, C = context.shape
    ctx_flat = context.permute(0, 2, 1).reshape(N * C, L)
    pred_flat = predict_median(model_wrapper, ctx_flat, dataset.pred_len,
                               chunk=args.eval_batch, use_amp=args.use_amp)
    preds = pred_flat.reshape(N, C, -1).permute(0, 2, 1).numpy()
    trues = future.numpy()
    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    print('[{}] windows={} shape={} mse:{:.6f}, mae:{:.6f}'.format(tag, N, preds.shape, mse, mae))
    return mse, mae, preds, trues


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train(model_wrapper, args, device, stats_t, run_dir):
    train_ds = ETTWindows(args.root_path, args.data_path, 'train', args.seq_len,
                          args.pred_len, max_windows=args.max_train_windows, data=args.data)
    val_ds = ETTWindows(args.root_path, args.data_path, 'val', args.seq_len, args.pred_len,
                        data=args.data)
    print('train windows: {}  val windows: {}'.format(len(train_ds), len(val_ds)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)

    params = list(model_wrapper.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    native_len = model_wrapper.bolt.chronos_config.prediction_length
    rf_loss = RegionFocalLoss(au=args.rf_au, anchor='ctxmean', soft=1, detach=1,
                              seq_len=args.seq_len, freq='h') \
        if args.task_loss == 'regionfocal' else None
    if rf_loss is not None:
        print('task criterion: regionfocal pinball (ctxmean anchor, soft gates, detached, au={})'.format(args.rf_au))

    best_val, best_epoch, bad = math.inf, -1, 0
    for epoch in range(args.epochs):
        model_wrapper.train()
        t0, task_acc, aux_acc, n_it = time.time(), 0.0, 0.0, 0
        for i, (seq_x, future) in enumerate(train_loader):
            seq_x = seq_x.to(device)            # [B, L, C]
            future = future.to(device)          # [B, H, C]
            B, L, C = seq_x.shape
            ctx = seq_x.permute(0, 2, 1).reshape(B * C, L)
            tgt = future[:, :native_len, :].permute(0, 2, 1).reshape(B * C, native_len)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=bool(args.use_amp) and device.startswith('cuda')):
                out, aux = model_wrapper(ctx, target=tgt)
                if args.task_loss == 'regionfocal':
                    loss = regionfocal_pinball(out.quantile_preds, tgt, ctx,
                                               list(model_wrapper.bolt.quantiles), rf_loss)
                    task_loss = loss.detach()
                else:
                    loss = out.loss
                    task_loss = out.loss.detach()
                if aux is not None:
                    aux_loss = descriptor_aux_loss(aux, future, args.desc_k, stats_t)
                    loss = loss + args.aux_weight * aux_loss
                else:
                    aux_loss = torch.zeros((), device=device)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optim.step()
            task_acc += task_loss.item()
            aux_acc += aux_loss.item()
            n_it += 1
            if (i + 1) % args.log_every == 0:
                print('epoch {} iter {}/{} task_loss {:.5f} aux_loss {:.5f}'.format(
                    epoch + 1, i + 1, len(train_loader), task_acc / n_it, aux_acc / n_it))
        print('epoch {} done in {:.1f}s | train task_loss {:.5f} aux_loss {:.5f}'.format(
            epoch + 1, time.time() - t0, task_acc / max(n_it, 1), aux_acc / max(n_it, 1)))

        val_mse, val_mae, _, _ = evaluate(model_wrapper, val_ds, args, device, 'val')
        if val_mse < best_val:
            best_val, best_epoch, bad = val_mse, epoch, 0
            save_checkpoint(model_wrapper, args, run_dir, best_val, best_epoch + 1)
        else:
            bad += 1
            print('EarlyStopping counter: {} out of {}'.format(bad, args.patience))
            if bad >= args.patience:
                print('Early stopping')
                break
    print('best epoch {} val mse {:.6f}'.format(best_epoch + 1, best_val))
    return run_dir


def save_checkpoint(model_wrapper, args, run_dir, val_mse, epoch):
    os.makedirs(run_dir, exist_ok=True)
    model_wrapper.bolt.save_pretrained(run_dir)
    if model_wrapper.aux_head == 'desc':
        torch.save(model_wrapper.head.state_dict(), os.path.join(run_dir, 'aux_head.pt'))
    with open(os.path.join(run_dir, 'train_meta.json'), 'w') as f:
        json.dump({'val_mse': val_mse, 'epoch': epoch, 'vars': vars(args)}, f, indent=2)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default=os.path.join(MODELS_DIR, 'chronos-bolt-small'))
    p.add_argument('--root_path', default=os.path.join(TSLIB, 'dataset', 'ETT-small'))
    p.add_argument('--data_path', default='ETTh1.csv')
    p.add_argument('--data', default='ETTh1')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--aux_head', choices=['none', 'desc'], default='none')
    p.add_argument('--aux_weight', type=float, default=1.0)
    p.add_argument('--task_loss', choices=['pinball', 'regionfocal'], default='pinball',
                   help='regionfocal: region-weighted pinball (ctxmean anchor, soft gates, detached weights)')
    p.add_argument('--rf_au', type=float, default=1.0, help='regionfocal alpha_U')
    p.add_argument('--desc_k', type=int, default=24)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--eval_batch', type=int, default=4096)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--use_amp', type=int, default=1)
    p.add_argument('--eval_only', action='store_true',
                   help='skip training; evaluates the base weights (zero-shot) unless --ckpt is given')
    p.add_argument('--ckpt', default='', help='fine-tuned checkpoint dir for --eval_only')
    p.add_argument('--max_train_windows', type=int, default=-1)
    p.add_argument('--max_test_windows', type=int, default=-1)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--run_name', default='')
    p.add_argument('--output_root', default=os.path.join(ROOT, 'checkpoints'))
    p.add_argument('--dump_root', default=os.path.join(ROOT, 'pred_dumps'))
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = args.run_name or 'tsfm_bolts_{}{}_{}'.format(
        args.aux_head,
        'w{}'.format(args.aux_weight) if args.aux_head == 'desc' else '',
        args.data)
    print('run_name={} device={} model_path={}'.format(run_name, device, args.model_path))

    load_path = args.ckpt if (args.eval_only and args.ckpt) else args.model_path
    model_wrapper = BoltWithAux(load_path, aux_head=args.aux_head, device=device)
    bolt = model_wrapper.bolt
    print('model: {} | d_model={} quantiles={} native_pred_len={} context_length={}'.format(
        os.path.basename(load_path.rstrip('/')), bolt.config.d_model, bolt.quantiles,
        bolt.chronos_config.prediction_length, bolt.chronos_config.context_length))
    n_params = sum(q.numel() for q in bolt.parameters())
    print('bolt params: {:.2f}M'.format(n_params / 1e6))

    if args.eval_only and args.ckpt and args.aux_head == 'desc':
        head_path = os.path.join(args.ckpt, 'aux_head.pt')
        if os.path.exists(head_path):
            model_wrapper.head.load_state_dict(torch.load(head_path, map_location=device))

    stats_t = None
    if args.aux_head == 'desc':
        train_ds_for_stats = ETTWindows(args.root_path, args.data_path, 'train',
                                        args.seq_len, args.pred_len,
                                        max_windows=args.max_train_windows, data=args.data)
        stats = load_or_compute_stats(args, train_ds_for_stats)
        stats_t = {k: torch.from_numpy(np.asarray(v)).float().to(device)
                   for k, v in stats.items() if np.asarray(v).dtype != object}

    if not args.eval_only:
        run_dir = os.path.join(args.output_root, run_name)
        train(model_wrapper, args, device, stats_t, run_dir)
        # reload best checkpoint for the test pass
        model_wrapper = BoltWithAux(run_dir, aux_head=args.aux_head, device=device)
        if args.aux_head == 'desc':
            model_wrapper.head.load_state_dict(
                torch.load(os.path.join(run_dir, 'aux_head.pt'), map_location=device))

    test_ds = ETTWindows(args.root_path, args.data_path, 'test', args.seq_len, args.pred_len,
                     data=args.data)
    t0 = time.time()
    mse, mae, preds, trues = evaluate(model_wrapper, test_ds, args, device, 'test')
    print('test eval took {:.1f}s'.format(time.time() - t0))

    dump_dir = os.path.join(args.dump_root, run_name)
    os.makedirs(dump_dir, exist_ok=True)
    np.save(os.path.join(dump_dir, 'pred.npy'), preds)
    np.save(os.path.join(dump_dir, 'true.npy'), trues)
    np.save(os.path.join(dump_dir, 'window_index.npy'), np.arange(preds.shape[0]))
    with open(os.path.join(dump_dir, 'metrics.json'), 'w') as f:
        json.dump({'mse': mse, 'mae': mae, 'run_name': run_name,
                   'eval_only': args.eval_only, 'vars': vars(args)}, f, indent=2)
    print('pred dump saved to {}'.format(dump_dir))
    print('mse:{}, mae:{}'.format(mse, mae))


if __name__ == '__main__':
    main()
