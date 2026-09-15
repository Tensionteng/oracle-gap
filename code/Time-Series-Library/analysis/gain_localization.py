"""
Gain localization (idea.md §5.1, mechanism diagnostic 3): does the descriptor
head's gain concentrate on windows whose future is structurally rich?

Groups test windows by structural keys of the TRUE future window:
  - CUSUM change-point score, 5 groups via train-set quantile edges
    (cp_score_edges in the descriptor stats npz);
  - drift magnitude 5-bin class (drift_edges in the stats npz),
and reports per-group MSE of two runs (base vs desc) plus the relative gain
(mse_base - mse_desc) / mse_base.

Inputs are directories containing pred.npy / true.npy ([N, pred_len, C]).
Both ./pred_dumps/<setting>/ (written by Exp_Descriptor_Forecast when
--save_pred 1) and the stock ./results/<setting>/ work, since the stock
test() saves the same pred.npy / true.npy format.

Usage from the TSLib root:
  uv run python analysis/gain_localization.py \
    --base_dir results/long_term_forecast_ETTh1_96_96_PatchTST_ETTh1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0 \
    --desc_dir pred_dumps/long_term_forecast_ETTh1_96_96_desc_PatchSTDesc_ETTh1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_desc_0 \
    --data_path ETTh1.csv --pred_len 96 --setting ETTh1_96_96_patchtst

Outputs: analysis/results/gain_<setting>.json and gain_<setting>.md.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compute_descriptor_stats import stats_cache_path
from utils.descriptor_labels import cusum_score_pos, mid_window_stats
from types import SimpleNamespace


def load_dump(d):
    pred = np.load(os.path.join(d, 'pred.npy'))
    true = np.load(os.path.join(d, 'true.npy'))
    assert pred.shape == true.shape, '{}: pred/true shape mismatch'.format(d)
    return pred.astype(np.float64), true.astype(np.float64)


def group_table(name, keys, mse_base, mse_desc, n_groups=5):
    rows = []
    for g in range(n_groups):
        m = keys == g
        if m.sum() == 0:
            continue
        b, d = float(mse_base[m].mean()), float(mse_desc[m].mean())
        rows.append({'group': int(g), 'n': int(m.sum()), 'mse_base': b, 'mse_desc': d,
                     'rel_gain': (b - d) / b})
    b, d = float(mse_base.mean()), float(mse_desc.mean())
    rows.append({'group': 'all', 'n': int(len(mse_base)), 'mse_base': b, 'mse_desc': d,
                 'rel_gain': (b - d) / b})
    md = ['| {} group | n | mse_base | mse_desc | rel_gain |'.format(name),
          '|---|---|---|---|---|']
    for r in rows:
        md.append('| {} | {} | {:.5f} | {:.5f} | {:+.2%} |'.format(
            r['group'], r['n'], r['mse_base'], r['mse_desc'], r['rel_gain']))
    return rows, '\n'.join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True,
                        help='dir with pred.npy/true.npy of the base run (results/ or pred_dumps/)')
    parser.add_argument('--desc_dir', type=str, required=True,
                        help='dir with pred.npy/true.npy of the desc run')
    parser.add_argument('--setting', type=str, required=True, help='label used in output filenames')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--desc_k', type=int, default=24)
    parser.add_argument('--stats', type=str, default=None,
                        help='descriptor stats npz; default descriptor_cache/<data>_pl<pred_len>.npz')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    out_path = args.out or os.path.join('analysis', 'results', 'gain_{}.json'.format(args.setting))

    pred_b, true_b = load_dump(args.base_dir)
    pred_d, true_d = load_dump(args.desc_dir)
    assert true_b.shape == true_d.shape, 'base/desc test windows differ: {} vs {}'.format(
        true_b.shape, true_d.shape)
    assert np.allclose(true_b, true_d), 'true.npy differs between the two dumps'
    mse_b = ((pred_b - true_b) ** 2).mean(axis=(1, 2))
    mse_d = ((pred_d - true_d) ** 2).mean(axis=(1, 2))

    true_t = torch.from_numpy(true_b).float()
    scores, _ = cusum_score_pos(true_t, args.desc_k)
    drift, _, _ = mid_window_stats(true_t)

    stats_path = args.stats or stats_cache_path(SimpleNamespace(
        data_path=args.data_path, pred_len=args.pred_len, max_train_windows=-1))
    notes = []
    if os.path.exists(stats_path):
        stats = dict(np.load(stats_path))
    else:
        stats = {}
        notes.append('stats npz {} not found; using quantiles of the dumped windows themselves'.format(
            stats_path))

    if 'cp_score_edges' in stats:
        cp_edges = torch.from_numpy(stats['cp_score_edges']).float()
    else:
        if 'cp_score_q' in stats:
            notes.append('cp_score_edges missing in old stats npz; using quantiles of the dumped windows')
        cp_edges = torch.quantile(scores, torch.tensor([0.2, 0.4, 0.6, 0.8]))
    if 'drift_edges' in stats:
        drift_edges = torch.from_numpy(stats['drift_edges']).float()
    else:
        drift_edges = torch.quantile(drift, torch.tensor([0.1, 0.3, 0.7, 0.9]))

    cp_groups = torch.bucketize(scores, cp_edges).numpy()
    drift_groups = torch.bucketize(drift, drift_edges).numpy()

    cp_rows, cp_md = group_table('cusum_score', cp_groups, mse_b, mse_d)
    drift_rows, drift_md = group_table('drift_bin', drift_groups, mse_b, mse_d)

    md = ['# Gain localization: {}'.format(args.setting), '',
          'base: `{}`  '.format(args.base_dir), 'desc: `{}`'.format(args.desc_dir), '']
    for n in notes:
        md.append('> note: {}'.format(n))
    md += ['', '## by CUSUM score group (train-quantile edges)', '', cp_md, '',
           '## by drift bin (train-quantile edges)', '', drift_md, '']
    md_text = '\n'.join(md)
    print(md_text)

    result = {'setting': args.setting, 'base_dir': args.base_dir, 'desc_dir': args.desc_dir,
              'stats': stats_path, 'notes': notes, 'n_windows': int(len(mse_b)),
              'by_cusum_score': cp_rows, 'by_drift_bin': drift_rows}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    with open(out_path.replace('.json', '.md'), 'w') as f:
        f.write(md_text)
    print('written: {} (+ .md)'.format(out_path))


if __name__ == '__main__':
    main()
