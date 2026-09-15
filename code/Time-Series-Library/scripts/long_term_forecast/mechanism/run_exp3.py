#!/usr/bin/env python
"""Mechanism experiment 3: PatchTSTGated repair ablation.

Runs: {exchange_rate, electricity} x seq_len {96,1440,2880} x variant {none,denbias,recmask}
      x seed {2021,2022} = 36 runs (24 variant arms + 12 uncapped vanilla controls).

'none' = PatchTSTGated with plain softmax attention (SDPA kernel; numerically equal
to the PatchTST baseline, see analysis/mechanism/smoke_patchtstgated.py).
No --max_train_windows (full training windows), hyperparameters follow the ehs_v2
main matrix (d_model=512, d_ff=2048, e_layers=2, n_heads=8, lr=1e-4;
bs: exchange 32, electricity 16/8/4 for sl 96/1440/2880 to fit memory).

Scheduling: one process per GPU, priority to the slowest arms (electricity sl=2880).
Finished runs (log tail contains 'mse:') are skipped on restart.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LOGDIR = os.path.join(ROOT, 'logs', 'mechanism', 'exp3')
os.makedirs(LOGDIR, exist_ok=True)

PY = os.path.join(ROOT, '.venv', 'bin', 'python')
SEQ_LENS = [96, 1440, 2880]
VARIANTS = ['none', 'denbias', 'recmask']
SEEDS = [2021, 2022]

DATASETS = {
    # name: (root, csv, enc_in, bs_by_sl)
    'exchange_rate': ('./dataset/exchange_rate/', 'exchange_rate.csv', 8, {96: 32, 1440: 32, 2880: 32}),
    'electricity': ('./dataset/electricity/', 'electricity.csv', 321, {96: 16, 1440: 8, 2880: 4}),
}


def build_runs():
    runs = []
    for name, (root, csv, enc, bs_map) in DATASETS.items():
        for sl in SEQ_LENS:
            for variant in VARIANTS:
                for seed in SEEDS:
                    mid = f'MECH_{name}_PTG_{variant}_{sl}_s{seed}'
                    log = os.path.join(LOGDIR, f'{name}_{variant}_sl{sl}_s{seed}.log')
                    cmd = [PY, '-u', 'run.py',
                           '--task_name', 'long_term_forecast', '--is_training', '1',
                           '--root_path', root, '--data_path', csv,
                           '--model_id', mid, '--model', 'PatchTSTGated',
                           '--data', 'custom', '--features', 'M',
                           '--seq_len', str(sl), '--label_len', '48', '--pred_len', '96',
                           '--e_layers', '2', '--d_layers', '1', '--factor', '3',
                           '--enc_in', str(enc), '--dec_in', str(enc), '--c_out', str(enc),
                           '--des', 'Exp', '--d_model', '512', '--d_ff', '2048', '--n_heads', '8',
                           '--learning_rate', '0.0001', '--batch_size', str(bs_map[sl]),
                           '--itr', '1', '--seed', str(seed), '--num_workers', '4',
                           '--attn_variant', variant, '--recmask_window', '336']
                    # priority: larger sl first, electricity before exchange
                    prio = (0 if name == 'electricity' else 1, -sl)
                    runs.append(dict(cmd=cmd, log=log, mid=mid, prio=prio))
    runs.sort(key=lambda r: r['prio'])
    return runs


def finished(log):
    if not os.path.exists(log):
        return False
    try:
        with open(log, 'rb') as f:
            f.seek(max(0, os.path.getsize(log) - 4000))
            tail = f.read().decode(errors='ignore')
        return 'mse:' in tail and 'dtw' in tail
    except OSError:
        return False


def main():
    runs = build_runs()
    todo = [r for r in runs if not finished(r['log'])]
    print(f'[sched] {len(runs)} runs total, {len(todo)} to do', flush=True)
    gpu_free = list(range(8))
    procs = {}  # pid -> (run, gpu, logfile_handle)
    while todo or procs:
        while todo and gpu_free:
            run = todo.pop(0)
            gpu = gpu_free.pop(0)
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            lf = open(run['log'], 'w')
            p = subprocess.Popen(run['cmd'], cwd=ROOT, stdout=lf,
                                 stderr=subprocess.STDOUT, env=env)
            procs[p.pid] = (run, gpu, lf)
            print(f'[sched] start {run["mid"]} on GPU {gpu} (pid {p.pid})', flush=True)
        time.sleep(20)
        done = [pid for pid, (r, g, lf) in procs.items()
                if procs[pid] and procs_pid_done(pid)]
        for pid in done:
            run, gpu, lf = procs.pop(pid)
            lf.close()
            ok = finished(run['log'])
            print(f'[sched] done {run["mid"]} on GPU {gpu}, metrics_written={ok}', flush=True)
            if not ok:
                print(f'[sched] WARN {run["mid"]} finished without metrics, check {run["log"]}', flush=True)
            gpu_free.append(gpu)
    print('[sched] all runs finished', flush=True)


def procs_pid_done(pid):
    try:
        done_pid, _ = os.waitpid(pid, os.WNOHANG)
        return done_pid == pid
    except ChildProcessError:
        return True


if __name__ == '__main__':
    main()
