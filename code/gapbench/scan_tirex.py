#!/usr/bin/env python
"""TiRex scanner — runs in gapbench/.venv-tirex (tirex-ts package).
Quantile output [B, H, 9] (0.1..0.9); median = index 4.
"""
import argparse
import os

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_common import scan_cells, load_cells, write_load_fail

MODEL_CKPT = '/mnt/jd/users/tengshiyuan.1/codes/mtp4ts/models/tirex/model.ckpt'


def load_tirex():
    from tirex.models.tirex import TiRexZero
    model = TiRexZero.from_pretrained(MODEL_CKPT, backend='torch', device='cuda:0')

    def predict(ctx, H):
        outs = []
        for s in range(0, len(ctx), 4096):
            q, _ = model.forecast(np.ascontiguousarray(ctx[s:s + 4096]),
                                  output_type='numpy', prediction_length=H)
            outs.append(q[:, :, 4])                    # 0.5 quantile
        return np.concatenate(outs, axis=0)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='')
    ap.add_argument('--reps', type=int, default=20)
    args = ap.parse_args()
    manifest, cells = load_cells(args.cells)
    try:
        predict = load_tirex()
    except Exception as e:
        print('MODEL LOAD FAIL tirex: {}'.format(str(e)[:300]))
        write_load_fail('tirex', manifest, cells, str(e))
        return
    print('model tirex loaded', flush=True)
    scan_cells('tirex', predict, manifest, cells, reps=args.reps)


if __name__ == '__main__':
    main()
