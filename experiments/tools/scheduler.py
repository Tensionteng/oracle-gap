#!/usr/bin/env python3
"""实验队列调度器 v2：按任务大小并卡调度 experiments/queue/*.sh。

- 小任务（名字含 ETT 或 weather）：占 1 个槽；大任务（ECL/Traffic）：占 2 槽。
- 每卡容量 2 槽 → 小任务 2 个/卡，大任务独占一卡。
- 卡上有外来进程（非本调度器后代）时不放新任务。
- 队列清空且无在跑任务后，等 30 分钟无新任务则汇总退出。
"""
import glob
import os
import subprocess
import time

ROOT = os.environ.get('MTP4TS_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSLIB = os.path.join(ROOT, 'Time-Series-Library')
Q = os.path.join(ROOT, 'experiments/queue')
RUN = os.path.join(ROOT, 'experiments/running')
DONE = os.path.join(ROOT, 'experiments/done')
LOGS = os.path.join(ROOT, 'experiments/logs')
NGPU, CAP, POLL, EMPTY_LIMIT = 8, 3, 10, 8 * 3600  # 小任务 3/卡；大任务独占；空窗 8 小时才退出（写作期常驻）
MAX_JOBS = 16  # 全局并发上限：16×(主进程4 OMP线程+2 worker) ≈ 96 线程 < 128 核


_children = []  # 持有 Popen 引用，每轮 poll() 回收，避免僵尸进程


def log(msg):
    print('[{}] {}'.format(time.strftime('%F %T'), msg), flush=True)


def weight(name):
    if 'tsfm' in name.lower():
        return CAP  # TSFM 微调：独占一卡
    if 'ETT' in name or 'weather' in name.lower():
        return 1  # 小任务：每卡最多 CAP 个
    return CAP  # 大任务（ECL/Traffic）：独占一卡


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def our_state():
    """返回 {name: (pid, gpu)}"""
    st = {}
    for f in glob.glob(os.path.join(RUN, '*.run')):
        name = os.path.basename(f)[:-4]
        try:
            pid = int(open(f).read().strip().split()[0])
        except Exception:
            continue
        gpu_f = os.path.join(RUN, name + '.gpu')
        gpu = int(open(gpu_f).read().strip()) if os.path.exists(gpu_f) else -1
        st[name] = (pid, gpu)
    return st


def wrapper_running(pid, job_path):
    """pid 活着、非僵尸、且 cmdline 确实是该任务的 bash wrapper 才算在跑"""
    try:
        with open('/proc/{}/stat'.format(pid), 'rb') as f:
            state = f.read().rsplit(b')', 1)[1].split()[0]
        if state == b'Z':
            return False
        with open('/proc/{}/cmdline'.format(pid), 'rb') as f:
            return job_path.encode() in f.read()
    except Exception:
        return False


def sweep(st):
    for name, (pid, gpu) in list(st.items()):
        job_path = os.path.join(Q, name + '.sh')
        if wrapper_running(pid, job_path):
            continue
        # 标记失效：日志里有最终 mse 说明真跑完了，否则重新入队
        logf = os.path.join(LOGS, name + '.log')
        try:
            finished = b'mse:' in open(logf, 'rb').read()
        except Exception:
            finished = False
        if finished:
            open(os.path.join(DONE, name + '.done'), 'a').close()
            log('job finished (recovered): {}'.format(name))
        else:
            log('stale marker, requeue: {}'.format(name))
        for ext in ('.run', '.gpu'):
            p = os.path.join(RUN, name + ext)
            if os.path.exists(p):
                os.remove(p)
        del st[name]


def descendant_of(pid, ancestors):
    """沿 /proc 父链向上找，判断 pid 是否为我们某 wrapper 的后代"""
    cur = pid
    for _ in range(64):
        if cur in ancestors or cur <= 1:
            return cur in ancestors
        try:
            with open('/proc/{}/stat'.format(cur), 'rb') as f:
                rest = f.read().rsplit(b')', 1)[1].split()  # comm 可能含括号，从最后一个 ')' 后解析
                cur = int(rest[1])  # state 之后的 ppid
        except Exception:
            return False
    return False


def foreign_pids_on_gpu(g, our_wrapper_pids):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '-i', str(g), '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True, timeout=15)
    except Exception:
        return [1]  # 查询失败时保守视为有外来进程
    foreign = []
    for line in out.split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if not descendant_of(pid, our_wrapper_pids):
            foreign.append(pid)
    return foreign


def launch(job, g):
    name = os.path.basename(job)[:-3]
    log_path = os.path.join(LOGS, name + '.log')
    with open(log_path, 'ab') as lf:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        p = subprocess.Popen(['bash', job], cwd=TSLIB, env=env,
                             stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    _children.append(p)
    with open(os.path.join(RUN, name + '.run'), 'w') as f:
        f.write(str(p.pid))
    with open(os.path.join(RUN, name + '.gpu'), 'w') as f:
        f.write(str(g))
    log('launch {} on GPU {} (pid {})'.format(name, g, p.pid))


def main():
    log('scheduler v2 started, pid {}'.format(os.getpid()))
    empty_since = None
    while True:
        for c in _children:  # 回收已结束的子进程，防止僵尸堆积误判为在跑
            c.poll()
        st = our_state()
        sweep(st)
        our_pids = {pid for pid, _ in st.values()}
        used = {}
        for name, (_, g) in st.items():
            used[g] = used.get(g, 0) + weight(name)
        pending = [j for j in sorted(glob.glob(os.path.join(Q, '*.sh')))
                   if not os.path.exists(os.path.join(DONE, os.path.basename(j)[:-3] + '.done'))
                   and os.path.basename(j)[:-3] not in st]
        launched_now = 0
        for job in pending:
            if len(st) + launched_now >= MAX_JOBS:
                break
            name = os.path.basename(job)[:-3]
            w = weight(name)
            for g in range(NGPU):
                if used.get(g, 0) + w > CAP:
                    continue
                if foreign_pids_on_gpu(g, our_pids):
                    continue
                launch(job, g)
                launched_now += 1
                used[g] = used.get(g, 0) + w
                our_pids = {pid for pid, _ in our_state().values()} | {int(open(os.path.join(RUN, name + '.run')).read())}
                time.sleep(2)
                break
        st = our_state()
        sweep(st)
        pending = [j for j in glob.glob(os.path.join(Q, '*.sh'))
                   if not os.path.exists(os.path.join(DONE, os.path.basename(j)[:-3] + '.done'))
                   and os.path.basename(j)[:-3] not in st]
        if not pending and not st:
            if empty_since is None:
                empty_since = time.time()
            elif time.time() - empty_since > EMPTY_LIMIT:
                log('queue empty for 30min, collecting results and exit')
                subprocess.call(['bash', os.path.join(ROOT, 'experiments/collect_results.sh')],
                                stdout=open(os.path.join(LOGS, 'collect.log'), 'a'),
                                stderr=subprocess.STDOUT)
                return
        else:
            empty_since = None
            log('status: pending={} running={} used={}'.format(len(pending), len(st), used))
        time.sleep(POLL)


if __name__ == '__main__':
    main()
