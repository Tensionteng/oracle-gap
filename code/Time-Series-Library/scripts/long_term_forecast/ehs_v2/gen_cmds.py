#!/usr/bin/env python
"""Generate command lists for the EHS confounder-controlled lookback study.

Outputs:
  main_cmds.txt - 8 datasets x {iTransformer, DLinear, PatchTST, TimesNet}
                  x seq_len {96,336,720,1440,2880} x seeds {2021,2022,2023}
                  = 480 runs, all with --max_train_windows = N_min(sl=2880)
  sat_cmds.txt  - saturation probe: electricity/traffic x {iTransformer, DLinear}
                  x seq_len {96,...,2880,5760} x 3 seeds = 72 runs,
                  all with --max_train_windows = N_min(sl=5760)

Hyperparameters:
  iTransformer/DLinear: family config from the task spec.
  PatchTST/TimesNet: key hyperparams (d_model/d_ff/e_layers/n_heads/top_k/bs)
    from the official TSLib scripts under scripts/long_term_forecast/;
    lr stays at the official default 0.0001.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_LENS = [96, 336, 720, 1440, 2880]
SEEDS = [2021, 2022, 2023]

# name: (data_arg, root, csv, enc_in, d_model, d_ff, e_layers, lr, bs)
FAMILY = {
    'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7, 128, 128, 2, '0.0001', 32),
    'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7, 128, 128, 2, '0.0001', 32),
    'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7, 128, 128, 2, '0.0001', 32),
    'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7, 128, 128, 2, '0.0001', 32),
    'exchange_rate': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8, 128, 128, 2, '0.0001', 32),
    'weather': ('custom', './dataset/weather/', 'weather.csv', 21, 512, 512, 3, '0.0001', 32),
    'electricity': ('custom', './dataset/electricity/', 'electricity.csv', 321, 512, 512, 3, '0.0005', 16),
    'traffic': ('custom', './dataset/traffic/', 'traffic.csv', 862, 512, 512, 3, '0.0005', 16),
}

# official TimesNet scripts: (d_model, d_ff, e_layers, n_heads, bs), lr default
TIMESNET = {
    'ETTh1': (16, 32, 2, 8, 32), 'ETTh2': (32, 32, 2, 8, 32),
    'ETTm1': (64, 64, 2, 8, 32), 'ETTm2': (32, 32, 2, 8, 32),
    'exchange_rate': (64, 64, 2, 8, 32), 'weather': (32, 32, 2, 8, 32),
    'electricity': (256, 512, 2, 8, 32), 'traffic': (512, 512, 2, 8, 32),
}
# official PatchTST scripts: (d_model, d_ff, e_layers, n_heads, bs), lr default
PATCHTST = {
    'ETTh1': (512, 2048, 1, 2, 32), 'ETTh2': (512, 2048, 3, 4, 32),
    'ETTm1': (512, 2048, 1, 2, 32), 'ETTm2': (512, 2048, 3, 16, 32),
    'exchange_rate': (512, 2048, 2, 8, 32), 'weather': (512, 2048, 2, 4, 32),
    'electricity': (512, 2048, 2, 8, 16), 'traffic': (512, 512, 2, 8, 4),
}

# N_min = train windows of the seq_len=2880 arm (pred_len=96)
N_MIN = {'ETTh1': 5665, 'ETTh2': 5665, 'ETTm1': 31585, 'ETTm2': 31585,
         'exchange_rate': 2336, 'weather': 33912, 'electricity': 15437, 'traffic': 9305}
# saturation probe: N_min of the seq_len=5760 arm
N_MIN_SAT = {'electricity': 18412 - 5760 - 96 + 1, 'traffic': 12280 - 5760 - 96 + 1}


def cmd(name, model, sl, seed, nmin, log, gpu, mid_prefix):
    data, root, csv, enc, dm, df, el, lr, bs = FAMILY[name]
    nh = 8
    extra = ''
    if model == 'TimesNet':
        dm, df, el, nh, bs = TIMESNET[name]
        lr = '0.0001'
        extra = ' --top_k 5'
    elif model == 'PatchTST':
        dm, df, el, nh, bs = PATCHTST[name]
        lr = '0.0001'
    return (f'CUDA_VISIBLE_DEVICES={gpu} .venv/bin/python -u run.py '
            f'--task_name long_term_forecast --is_training 1 '
            f'--root_path {root} --data_path {csv} '
            f'--model_id {mid_prefix}_{name}_{model}_{sl}_s{seed} --model {model} '
            f'--data {data} --features M --seq_len {sl} --label_len 48 --pred_len 96 '
            f'--e_layers {el} --d_layers 1 --factor 3 --enc_in {enc} --dec_in {enc} --c_out {enc} '
            f'--des Exp --d_model {dm} --d_ff {df} --n_heads {nh} '
            f'--learning_rate {lr} --batch_size {bs} --itr 1 --seed {seed} '
            f'--max_train_windows {nmin}{extra} > {log} 2>&1')


def main():
    i = 0
    lines = []
    # saturation probe first (smaller, distinct N_min regime)
    os.makedirs('logs/ehs_v2/sat', exist_ok=True)
    for seed in SEEDS:
        for sl in SEQ_LENS + [5760]:
            for model in ['iTransformer', 'DLinear']:
                for name in ['electricity', 'traffic']:
                    log = f'logs/ehs_v2/sat/{name}_{model}_sl{sl}_s{seed}.log'
                    lines.append(cmd(name, model, sl, seed, N_MIN_SAT[name], log, i % 8, 'LBsat'))
                    i += 1
    with open(os.path.join(HERE, 'sat_cmds.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    sat_n = len(lines)

    lines = []
    os.makedirs('logs/ehs_v2', exist_ok=True)
    for seed in SEEDS:
        for sl in SEQ_LENS:
            for model in ['iTransformer', 'DLinear', 'PatchTST', 'TimesNet']:
                for name in FAMILY:
                    log = f'logs/ehs_v2/{name}_{model}_sl{sl}_s{seed}.log'
                    lines.append(cmd(name, model, sl, seed, N_MIN[name], log, i % 8, 'LBv2'))
                    i += 1
    with open(os.path.join(HERE, 'main_cmds.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'sat_cmds.txt: {sat_n} runs; main_cmds.txt: {len(lines)} runs')


if __name__ == '__main__':
    main()
