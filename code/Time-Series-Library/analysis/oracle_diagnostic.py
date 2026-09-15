"""Oracle 判决模拟：base 模型的过平滑签名有多少是「理性收缩」？

方法：把 base 预测 ŷ 当作条件均值 μ 的代理（若 base 已是最优，则此代理精确；
若 base 过度收缩，则 oracle 幅度比被低估 → 检验保守）。用窗口级自助法
重采样残差 ε = y − ŷ 构造合成实现 ỹ = ŷ + ε'，计算「预言机」相对新实现的
欠冲率/过冲率/错侧率/幅度比，与 base 相对真实值的同名指标对比。

若 base ≈ oracle：理性收缩占满，regionfocal 类外推无肉；
若 base 明显比 oracle 更平滑：仍有未利用结构。
"""
import argparse
import numpy as np
import pandas as pd

BORDERS_H = {'ETTh1': 8640, 'ETTh2': 8640, 'ETTm1': 34560, 'ETTm2': 34560}


def load_inputs(root, data_path, data, seq_len, n_win):
    df = pd.read_csv(root + data_path)
    vals = df.drop(columns=[c for c in ('date', 'OT_date') if c in df.columns]).values.astype(np.float32)
    n = len(vals)
    if data in BORDERS_H:
        unit = BORDERS_H[data]
        train = vals[:unit]
        b1, b2 = unit + unit // 3 - seq_len, unit + 2 * (unit // 3)
    else:  # custom 协议：7/1/2 划分
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        train = vals[:num_train]
        b1, b2 = n - num_test - seq_len, n
    mean, std = train.mean(0), train.std(0)
    vals = (vals - mean) / std
    test = vals[b1:b2]
    xs = np.stack([test[i:i + seq_len] for i in range(n_win)])
    return xs  # [N, seq_len, C]


def metrics(pred, true, anchor):
    p = pred - anchor
    d = true - anchor
    same = p * d > 0
    under = same & (np.abs(p) < np.abs(d))
    over = same & (np.abs(p) > np.abs(d))
    wrong = ~same
    amp = pred.std() / (true.std() + 1e-8)
    return dict(under=float(under.mean()), over=float(over.mean()),
                wrong=float(wrong.mean()), amp=float(amp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--data', default='ETTh1')
    ap.add_argument('--root_path', default='./dataset/ETT-small/')
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--seq_len', type=int, default=96)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    pred = np.load(f'{args.dump}/pred.npy')
    true = np.load(f'{args.dump}/true.npy')
    N, L, C = pred.shape
    xs = load_inputs(args.root_path, args.data_path or f'{args.data}.csv',
                     args.data, args.seq_len, N)
    anchor = xs.mean(axis=1, keepdims=True)  # ctxmean，广播到 L

    base_m = metrics(pred, true, anchor)
    resid = true - pred
    rng = np.random.default_rng(args.seed)
    oracle_ms = []
    for _ in range(args.reps):
        perm = rng.permutation(N)
        synth = pred + resid[perm]
        oracle_ms.append(metrics(pred, synth, anchor))

    print(f'== {args.data}  (N={N}, pred std={pred.std():.4f}, true std={true.std():.4f}, '
          f'resid var={resid.var():.4f}) ==')
    print(f'{"":8}{"欠冲率":>9}{"过冲率":>9}{"错侧率":>9}{"幅度比":>9}')
    print(f'{"base":8}{base_m["under"]:>9.3f}{base_m["over"]:>9.3f}'
          f'{base_m["wrong"]:>9.3f}{base_m["amp"]:>9.3f}')
    om = {k: np.mean([m[k] for m in oracle_ms]) for k in oracle_ms[0]}
    os_ = {k: np.std([m[k] for m in oracle_ms]) for k in oracle_ms[0]}
    print(f'{"oracle":8}{om["under"]:>9.3f}{om["over"]:>9.3f}{om["wrong"]:>9.3f}'
          f'{om["amp"]:>9.3f}   (±{os_["amp"]:.3f})')
    gap = om['amp'] - base_m['amp']
    print(f'\n幅度比 gap(oracle−base) = {gap:+.3f}  →  ' +
          ('base 明显比 oracle 更平滑，存在未利用结构' if gap > 0.02 else
           'base ≈ oracle，理性收缩占满，外推无肉' if gap > -0.02 else
           'base 比 oracle 更浪（异常，检查锚点）'))


if __name__ == '__main__':
    main()
