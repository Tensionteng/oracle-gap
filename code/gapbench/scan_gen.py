#!/usr/bin/env python
"""sundial / TimeMoE / timer scanner — runs in gapbench/.venv-gen
(transformers 4.46.3, the generation-cache API these repos were written
against; transformers 5.x in the main env breaks them).
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

MODELS_DIR = '" + os.environ.get("MTP4TS_ROOT", ".") + "/models'


def load_generate_model(path, num_samples=1, batch=256, dtype=torch.float32,
                        instance_norm=False):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                 torch_dtype=dtype)
    model = model.to('cuda').eval()

    def predict(ctx, H):
        outs = []
        with torch.no_grad():
            for s in range(0, len(ctx), batch):
                x = torch.from_numpy(np.ascontiguousarray(ctx[s:s + batch]))
                x = x.to(device='cuda', dtype=torch.float32)
                if instance_norm:
                    # TimeMoE's generate() does no input normalization (its
                    # ts_generation_mixin has no revin); per-instance z-norm is
                    # the canonical usage and matches the other models'
                    # internal instance norm
                    mu = x.mean(dim=-1, keepdim=True)
                    sd = x.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
                    x = (x - mu) / sd
                kwargs = dict(max_new_tokens=H)
                if num_samples > 1:
                    kwargs['num_samples'] = num_samples
                out = model.generate(x, **kwargs)
                out = out[..., -H:]                      # [B, (S,) H]
                if instance_norm:
                    if out.dim() == 3:
                        out = out * sd.unsqueeze(1) + mu.unsqueeze(1)
                    else:
                        out = out * sd + mu
                if out.dim() == 3:
                    out = out.median(dim=1).values
                outs.append(out.float().cpu().numpy())
        return np.concatenate(outs, axis=0)
    return predict


REGISTRY = {
    'sundial': lambda: load_generate_model(os.path.join(MODELS_DIR, 'sundial-base-128m'),
                                           num_samples=20),
    'timemoe_50m': lambda: load_generate_model(os.path.join(MODELS_DIR, 'TimeMoE-50M'),
                                               instance_norm=True),
    'timemoe_200m': lambda: load_generate_model(os.path.join(MODELS_DIR, 'TimeMoE-200M'),
                                                instance_norm=True),
    'timer_84m': lambda: load_generate_model(os.path.join(MODELS_DIR, 'timer-base-84m')),
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
