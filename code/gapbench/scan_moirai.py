#!/usr/bin/env python
"""Moirai scanner — runs in gapbench/.venv-moirai (uni2ts 2.x + gluonts).

moirai-1.1 (small/large): MoiraiForecast, sample-based (num_samples=20) ->
per-step median. moirai-2.0-R-small: Moirai2Forecast, native 9-quantile head ->
0.5 quantile. Each instance is one univariate context of seq_len steps.
"""
import argparse
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_common import scan_cells, load_cells, write_load_fail

MODELS_DIR = '/mnt/jd/users/tengshiyuan.1/codes/mtp4ts/models'
NUM_SAMPLES = 20


def _gluonts_dataset(ctx):
    from gluonts.dataset.common import ListDataset
    return ListDataset([{'target': c, 'start': '2000-01-01'} for c in ctx], freq='h')


def load_moirai11(path, patch_size=16):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained(path)
    cache = {}

    def predict(ctx, H):
        key = (H, len(ctx))
        if key not in cache:
            model = MoiraiForecast(
                module=module, prediction_length=H, context_length=ctx.shape[1],
                patch_size=patch_size, num_samples=NUM_SAMPLES, target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            cache[key] = model.create_predictor(batch_size=512)
        forecasts = list(cache[key].predict(_gluonts_dataset(ctx)))
        med = np.stack([np.median(f.samples, axis=0) for f in forecasts])
        return med
    return predict


def load_moirai2(path):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    module = Moirai2Module.from_pretrained(path)
    cache = {}

    def predict(ctx, H):
        key = (H, len(ctx))
        if key not in cache:
            model = Moirai2Forecast(
                module=module, prediction_length=H, context_length=ctx.shape[1],
                target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            cache[key] = model.create_predictor(batch_size=512)
        forecasts = list(cache[key].predict(_gluonts_dataset(ctx)))
        outs = []
        for f in forecasts:
            if hasattr(f, 'quantile'):                # QuantileForecast
                outs.append(f.quantile(0.5))
            else:                                     # SampleForecast
                outs.append(np.median(f.samples, axis=0))
        return np.stack(outs)
    return predict


REGISTRY = {
    'moirai11_small': lambda: load_moirai11(os.path.join(MODELS_DIR, 'moirai-1.1-R-small')),
    'moirai11_large': lambda: load_moirai11(os.path.join(MODELS_DIR, 'moirai-1.1-R-large')),
    'moirai2_small': lambda: load_moirai2(os.path.join(MODELS_DIR, 'moirai-2.0-R-small')),
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
