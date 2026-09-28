#!/usr/bin/env python
"""Zero-shot Chronos-Bolt dump on a unified window spec (gapbench adapter npz).

Each (window, channel) becomes one univariate instance of seq_len context;
the official ChronosBoltPipeline produces the median forecast of pred_len
steps (its built-in continuation when pred_len > 64). Raw scale throughout;
gap.py does the per-window instance normalization.

Writes <out>/{pred,true,inputs}.npy plus meta.json (timings for the cost
model in the pilot report).
"""
import argparse
import json
import os
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', required=True)
    ap.add_argument('--model_path', default='/mnt/jd/users/tengshiyuan.1/codes/mtp4ts/models/chronos-bolt-small')
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    w = np.load(args.windows, allow_pickle=True)
    inputs, true = w['inputs'], w['true']           # [N, L, C], [N, H, C] raw
    N, L, C = inputs.shape
    H = true.shape[1]
    print('windows {}: N={} L={} C={} H={} (instances={})'.format(
        args.windows, N, L, C, H, N * C))

    from chronos import BaseChronosPipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    pipeline = BaseChronosPipeline.from_pretrained(args.model_path, device_map=device)
    print('model loaded in {:.1f}s on {}'.format(time.time() - t0, device))

    ctx = np.ascontiguousarray(inputs.transpose(0, 2, 1).reshape(N * C, L))
    ctx_t = torch.from_numpy(ctx)
    preds = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, ctx_t.shape[0], args.chunk):
            _, median = pipeline.predict_quantiles(ctx_t[s:s + args.chunk],
                                                   prediction_length=H,
                                                   quantile_levels=[0.5])
            preds.append(median.float().cpu())
    pred = torch.cat(preds, dim=0).reshape(N, C, H).permute(0, 2, 1).numpy()
    infer_s = time.time() - t0

    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    print('raw-scale  mse:{:.6f}, mae:{:.6f}  infer {:.1f}s ({:.2f} ms/window-channel)'.format(
        mse, mae, infer_s, 1000 * infer_s / (N * C)))

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'pred.npy'), pred.astype(np.float32))
    np.save(os.path.join(args.out, 'true.npy'), true.astype(np.float32))
    np.save(os.path.join(args.out, 'inputs.npy'), inputs.astype(np.float32))
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump({'windows': args.windows, 'model_path': args.model_path,
                   'n_windows': N, 'n_channels': C, 'pred_len': H,
                   'infer_seconds': infer_s, 'mse_raw': mse, 'mae_raw': mae,
                   'tag': args.tag}, f, indent=2)
    print('dump saved to {}'.format(args.out))


if __name__ == '__main__':
    main()
