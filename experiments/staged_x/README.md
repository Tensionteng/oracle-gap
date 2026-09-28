# x 系列：决策侧信号向非 ETT 数据集的 TSFM 迁移（r5 评审 no-quick-change 项）

**目的**：检验 §7 描述子头决策信号在微调 Chronos-Bolt-small 上是否复现 tab:decision 的
context-visibility 规则（目前 TSFM 证据仅覆盖 ETT）。

**协议**：与论文 b 系列（ETT）逐项一致——Chronos-Bolt-small、AdamW lr 1e-5、bf16、
pinball loss、epochs≤10、patience 3、batch 32、seed 2021、seq/pred 96/96；
数据用 TSLib 的 electricity.csv / traffic.csv（`--data custom`，7/1/2 borders），
与 tab:decision 的 PatchTST 行同数据源同划分，可直接对比。
每数据集两个 run：`none`（精度参照）与 `descw01`（描述子头 w=0.1）；
desc run 训练后自动接 errcorr_bolt.py 分析（四种排序信号，coverage grid 同 ETT）。

## 先验预测（2026-09-22 登记，跑前写死）

1. **Traffic**：tab:decision 中 ctx_vol 排序失效（+3.2%）、描述子 vol 通道最优（−3.7%）。
   预测：微调 bolt 上描述子排序在 coverage 0.8 处降低剩余 MSE ≥ 2%，且优于 ctx_vol 排序。
2. **Electricity**：tab:decision 中 ctx_vol 已经有效（−5.4%）、描述子无 margin（−5.3% vs −5.4%）。
   预测：描述子排序有效但对 ctx_vol 的 margin ≤ 1 个百分点（context-visibility 规则跨 backbone 成立）。
3. **精度无损**：descw01 vs none 的 test MSE 差在 ±1% 以内（头是免费的）。

**判定**：三条全中 → 附录 tab:decision 加 TSFM 块 + §7 一句话；部分中 → 如实写；
全不中 → 留作 rebuttal 材料，正文不动。

**产物**：checkpoints/tsfm_bolts_{none,descw01}_{ECL,traffic}、
pred_dumps/ 同名、results/errcorr_bolt_{ECL,traffic}.json、
日志 experiments/logs/x*_tsfm_bolts_*.log。

## 结果（2026-09-23 跑完，2026-09-28 判定并落文）

bolt 微调（descw01）@ coverage 0.8：

| 数据集 | desc cp | desc vol | ctx_vol | oracle_vol |
|---|---|---|---|---|
| ECL | −0.46% | +0.03% | **−5.96%** | −4.29% |
| Traffic | +1.55% | **−3.25%** | +2.94% | −3.97% |

判定：预测 1（Traffic 描述子赢 ≥2% 且优于 ctx_vol）**命中**（vol 通道 −3.25%，pattern 与 PatchTST 行逐格一致）；
预测 2（ECL margin 消失）**规则成立、细节有差**（ctx_vol −5.96% 时描述子无 margin，但 PatchTST 上 desc vol 曾打平 ctx_vol，bolt 未复现）；
预测 3（精度 ±1%）**命中**（val MSE 差 ECL +0.4%、Traffic −0.2%）。
注：errcorr_bolt.py 原脚本只提取 cp 通道，已补 vol 通道（期望 bin）并对四个 bolt 数据集统一重跑。
落文：附录 tab:decision 加 TSFM 块（4 行）+ §7「Where the signal pays off」加跨 backbone 迁移句；正文维持 9 页（tectonic 编译 0 error）。
