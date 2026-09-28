#!/usr/bin/env python
"""chronos family scanner — runs in tsfm_stage2/.venv (chronos-forecasting).
Covers chronos-bolt-tiny/mini/small/base(full) and chronos-2.
Median conventions: 0.5 quantile of the official pipeline output; for
prediction_length > 64 the pipeline's built-in continuation is used.
"""
import argparse
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_common import scan_cells, load_cells, write_load_fail

MODELS_DIR = '/mnt/jd/users/tengshiyuan.1/codes/mtp4ts/models'


def load_chronos(path):
    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained(path, device_map='cuda')

    def predict(ctx, H, chunk=2048):
        outs = []
        with torch.no_grad():
            for s in range(0, len(ctx), chunk):
                t = torch.from_numpy(np.ascontiguousarray(ctx[s:s + chunk]))
                _, median = pipeline.predict_quantiles(t, prediction_length=H,
                                                       quantile_levels=[0.5])
                outs.append(median.float().cpu().numpy())
        return np.concatenate(outs, axis=0)
    return predict


def load_chronos2(path):
    """Chronos-2: input (n_series, n_variates, history); predict_quantiles
    returns per-series lists of [n_variates, H, n_quantiles]."""
    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained(path, device_map='cuda')

    def predict(ctx, H, chunk=1024):
        outs = []
        with torch.no_grad():
            for s in range(0, len(ctx), chunk):
                t = torch.from_numpy(np.ascontiguousarray(ctx[s:s + chunk]))[:, None, :]
                _, medians = pipeline.predict_quantiles(t, prediction_length=H,
                                                        quantile_levels=[0.5])
                outs.append(torch.stack([m[0] for m in medians]).float().cpu().numpy())
        return np.concatenate(outs, axis=0)
    return predict


REGISTRY = {
    'bolt_tiny': lambda: load_chronos(os.path.join(MODELS_DIR, 'chronos-bolt-tiny')),
    'bolt_mini': lambda: load_chronos(os.path.join(MODELS_DIR, 'chronos-bolt-mini')),
    'bolt_small': lambda: load_chronos(os.path.join(MODELS_DIR, 'chronos-bolt-small')),
    'bolt_base': lambda: load_chronos(os.path.join(MODELS_DIR, 'chronos-bolt-base-full')),
    'chronos2': lambda: load_chronos2(os.path.join(MODELS_DIR, 'chronos-2')),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--cells', default='')
    ap.add_argument('--reps', type=int, default=20)
    args = ap.parse_args()
    manifest, cells = load_cells(args.cells)
    for name in args.models.split(','):
        try:
            predict = REGISTRY[name]()
        except Exception as e:
            print('MODEL LOAD FAIL {}: {}'.format(name, str(e)[:300]))
            write_load_fail(name, manifest, cells, str(e))
            continue
        print('model {} loaded'.format(name), flush=True)
        scan_cells(name, predict, manifest, cells, reps=args.reps)


if __name__ == '__main__':
    main()
