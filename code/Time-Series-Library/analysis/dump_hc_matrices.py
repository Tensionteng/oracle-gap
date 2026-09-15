import torch, glob, os, sys
sys.path.insert(0, '.')
from models.iTransformerHC import sinkhorn, cayley

for ckpt_path in sorted(glob.glob('checkpoints/*iTransformerHC*/checkpoint.pth')):
    name = ckpt_path.split('/')[-2]
    mode = [m for m in ['_res', '_hc', '_mhc', '_ohc'] if m + '_' in name]
    mode = mode[0][1:] if mode else '?'
    if mode == 'res':
        continue
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    print(f"\n===== {name}")
    for key in sorted(sd):
        if key.endswith('hc_attn.theta') or key.endswith('hc_ffn.theta'):
            theta = sd[key]
            if mode == 'mhc':
                A = sinkhorn(theta)
            elif mode == 'ohc':
                A = cayley(theta)
            else:
                A = theta
            sv = torch.linalg.svdvals(A)
            tag = key.replace('.theta', '')
            print(f"  {tag}: A=\n{torch.round(A*100)/100}\n    singular_values={sv.numpy().round(3)}, min_entry={A.min().item():.3f}")
