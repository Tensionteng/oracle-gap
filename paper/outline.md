# 论文详细大纲 v1

> 项目：mtp4ts（原 I8 描述子头项目演化版）
> 目标 venue：ICLR 2027（截稿约 2026-09 下旬，时间紧）或 NeurIPS 2027；若来不及则 KDD 2027
> 状态：实验 95% 封板（h 系列补尾在跑）

## 0. 标题候选

1. **Not All Smoothness Is Fixable: Diagnosing and Conditionally Repairing Over-Smoothing in Time Series Forecasting**（主推）
2. When Is Over-Smoothing Worth Fixing? An Oracle-Gap Audit of Forecasting Benchmarks
3. Predict the Structure Before You Trust the Forecast（保留原「预测空间」精神，决策向）

## 1. 一句话命题

MSE 导致的过平滑由两个成分构成——不确定下的**理性收缩**（不可动）与**未利用的结构**（可修）；我们给出区分两者的诊断（oracle gap）、预测修复何时有效的定律、一个条件化修复损失（regionfocal）、以及修复不了时的应对（描述子驱动的选择性预测）。

## 2. Contributions box（投稿页用）

1. **两成分理论**：过平滑 = 理性收缩 + 未利用结构；条件改进区间 (0,2b)；Jensen 间隙分析（为何辅助头/加权优于直接幅度匹配）。
2. **Oracle gap 诊断**：残差自助重采样的预言机签名模拟，逐（数据集×模型）度量可榨结构。
3. **gap 决定论**：gap 预测反平滑方法收益方向与幅度——任务模型 14 格 + TSFM 665 格，零矛盾。
4. **RegionFocal**：区域加权损失，gap>0 处有效（ECL -2.3%×3 seeds，4 个 horizon 不衰减），gap≤0 处退化为 MSE。
5. **决策侧**：预测变点概率驱动选择性预测，弃置 top-20% 风险窗剩余 MSE -5.3%，优于 oracle 与静态基线。
6. **Benchmark 审计**：665 格地图——ETT 全频段饱和、交通/能源有肉、规模阶梯无效应、模型互补格。

## 3. 逐节大纲

### 1 Introduction
- 钩子：过平滑被反复指控（FreDF 标签自相关定理、HCAN 高熵特征、DILATE 形状损失），所有工作默认它总该修。
- 动机实验（图 1）：SOTA 模型在 ETTh1 上欠冲率 50.3%、幅度比 0.844 —— 看似铁证；但**预言机在同一数据上的签名几乎相同**（欠冲 50.3%、幅度比 0.822）。「平滑的预测」不等于「有病的预测」。
- 提问三连：能测吗？能预测吗？能修吗/修不了怎么办？
- 贡献列表 + 路线图。
- 防御性写作位：对 FreDF/HCAN 的立场（「无条件宣称」 vs 我们「条件化」），不是否定而是补全适用条件。

### 2 Related Work
- 反平滑/损失设计线：FreDF、HCAN、DILATE、asymmetric/expectile 损失 → 共同点：无条件宣称；差异：我们给适用条件。
- MTP/前瞻监督线（留给描述子头）：Gloeckle MTP、FSP（ICLR 2026，**必须引用并划界**：它验证了 LLM 域未来摘要监督，我们是时序域结构描述子 + 决策用途）、MuToR、Moirai-2.0。
- 选择性预测/不确定性：reject-option TSF（2026 两篇）、conformal（ACI/DtACI/CPTC）→ 我们的 cp_prob 是内生免费信号。
- Benchmark 审计/元分析：GIFT-Eval、"are we making progress" 类。

### 3 Theory: The Two Components of Over-Smoothing
- 3.1 设定与记号；MSE 分解 MSE = E‖ŷ−μ‖² + Var(ε)。
- 3.2 命题 1（预言机欠冲）：幅度比上界 √(1−σ²/Var(y))；欠冲率是预言机的属性。证明草稿已在项目日志。
- 3.3 命题 2（条件改进）：收缩偏置 b 存在时，反向修正 δ 的收益 = 2⟨δ,b⟩−|δ|²；严格改进区间 (0,2b)，最优 δ*=b。
- 3.4 命题 3（Jensen 冲突）：非线性描述子直接匹配 ŷ 与 MSE 最优解冲突 → 辅助头/加权路线必要性。
- 3.5 高斯闭式例：不动点偏移 ≈0.4ασ；双井分析（锚点=条件均值退化情形）。
- 3.6 不可兼得 remark：残差相关加权 ⟹ 不动点偏离条件均值；残差不相关 ⟹ 无法靶向欠冲。

### 4 The Oracle Gap Diagnostic
- 4.1 定义：窗口级自助残差重采样，模拟预言机签名（欠冲率/幅度比期望）；gap = oracle_amp − model_amp。
- 4.2 协议：实例级标准化、tail 窗口、reps=20 CI95、锚点敏感性（ctxmean vs seasonal 报告差值）。
- 4.3 性质：模型无关、零训练、5-15s/格（TSFM）；与训练协议正交。
- 4.4 与探针/互信息的关系定位（gap 测「残差里的可榨结构」，探针测「表示里的现存信息」——互补）。

### 5 The Law: Gap Predicts When Anti-Smoothing Helps
- 5.1 任务模型主表（表 1，核心表）：9 数据集 × 5 backbone 的 (gap, regionfocal 增益) 对照——ECL(+0.030→-2.3%)、Traffic(+0.027→-0.9%)、ETT×4(负→负)、weather(-0.06→负)、SWaT(+0.006→中性)、penm(-0.005→中性)。**含 7 个阴性/中性对照点**。
- 5.2 backbone 分析：架构假说（通道独立）被 DLinear 证伪 → 统一为 gap 决定论；iTransformer 阴性案例（gap<0）。
- 5.3 TSFM 地图（图 2，665 格热力图）：数据集排名（交通/客流/能源有肉）；**规模阶梯平坦**（bolt 8M→205M 无效应）；模型互补格（ett1：PatchTST +0.077 vs TSFM -0.105；SZ_TAXI 反向）。
- 5.4 元结论：ETT 系 benchmark 饱和的证明；「这个领域还在进步吗」的定量回答。

### 6 RegionFocal: Conditional Repair
- 6.1 方法：三区定义（欠冲/过冲/错侧）、软门控、权重停梯度、锚点选择；λ_base 可选项（FiberPO 类比一句话）。
- 6.2 主结果（表 2）：ECL -2.3%（3 seeds：-2.34/-2.39/-2.19%）、四个 horizon 全部 -2.3~-2.7%（不衰减——对比 desc 头的 horizon 衰减）、Traffic -0.9%；剂量-响应峰 α*=1 与理论区间吻合。
- 6.3 行为指标（表 3）：欠冲率 0.629→0.567、幅度比 0.872→0.911（机制确认）。
- 6.4 消融：锚点（h4/h5）、硬门控（h6）、错侧权重（h7）、detach（rfgrad 对照）；HCAN 头对头（h8）。
- 6.5 新战场闭环（h0-h3）：SZ_TAXI/bizitobs 上 gap→效果复验（补尾实验，决定本节强度）。

### 7 When Not Fixable: Descriptor-Driven Selective Prediction
- 7.1 转折：gap≤0 处 regionfocal 退化，但模型仍可对「本次预测的可信度」负责。
- 7.2 描述子头（轻量、免费标签、<0.1% FLOPs）：预测未来窗口变点概率/漂移/波动率档。
- 7.3 主结果（图 3，risk-coverage 曲线）：弃置 top-20% 风险窗 → 剩余 MSE -2.7%/-5.3%（ETTh1/ETTm1）；优于 oracle 变点排序（-1.6%/-4.5%）与静态波动率基线（+0.4%/+1.2%）——前瞻 > 静态。ECL 补充格（h 系列在跑）。
- 7.4 与 CPTC 的划界：内生描述子 vs 外生状态模型。
- 7.5 局限：预测 vol 档排序质量差（oracle_vol 上界 -10% 未兑现）——future work。

### 8 Analysis: What Did Not Work（阴性结果节，reviewer 友好）
- 级联条件化（concat/FiLM）不超并行：信息流分析（z 干预实验：分布级调节 ≫ 逐窗内容）；
- 描述子门控 regionfocal 不超静态：ECL 上训练门控 0.1784 < 静态 0.1758 不成立，随机门控 0.1752——门控信息未被有效利用；
- 描述子头的点精度：任务模型尺度 ~1% 且 seed 敏感、TSFM 48M 中性偏负；探针弱阳性 + 梯度淹没 9-25× 定量解释；
- 这些阴性结果全部反刍回两成分理论（多数窗口是理性收缩）。

### 9 Discussion
- 何时有效决策树（gap 诊断 → 修复/弃置/接受）；
- 与规模叙事的关系（零样本无效应；微调设定留待后续）；
- 局限：单任务模型家族为主、GIFT-Eval 子集、点预测框架；
- 伦理/影响：选择性预测在高风险场景的责任边界。

## 4. 图表清单

| 编号 | 内容 | 证据来源 |
|---|---|---|
| 图 1 | 动机：SOTA 欠冲签名 vs 预言机签名（ETTh1） | behavior_metrics + oracle ETTh1 |
| 图 2 | 665 格 gap 热力图（模型×数据集） | gapbench/results/matrix.csv |
| 图 3 | risk-coverage 曲线（3 数据集 × 4 排序信号） | errcorr_*.json |
| 图 4 | 剂量-响应倒 U（ECL vs ETT） | c/d/f 系列 results.csv |
| 表 1 | gap↔效果定律主表（14 格 + 665 格摘要） | matrix.csv + results.csv |
| 表 2 | regionfocal 主结果（含多 seed、多 horizon） | f 系列 |
| 表 3 | 行为指标表 | behavior_*.json |
| 附录 | 证明、665 格全表、HCAN/FreDF 复现注意事项、消融全表 | — |

## 5. 断言登记（claim → 证据 → 状态）

| Claim | 证据 | 状态 |
|---|---|---|
| 过平滑两成分 | 理论推导 + oracle 签名重合（ETT 4 数据集） | ✅ |
| gap 预测收益方向 | 14 格任务模型 + 665 格 TSFM 零矛盾 | ✅ |
| regionfocal 修复有效且条件化 | ECL -2.3%×3 seeds + Traffic -0.9% + 阴性对照 | ✅（h 系列补 SZ_TAXI/bizitobs） |
| cp_prob 是有效前瞻决策信号 | -2.7%/-5.3%，优于 oracle/静态 | ✅（h 系列补 ECL） |
| 零样本 gap 无规模效应 | bolt 阶梯 8M→205M 平坦 | ✅ |
| regionfocal 架构无关（gap 决定） | 5 backbone 一致 | ✅ |

## 6. 风险登记

| 风险 | 等级 | 预案 |
|---|---|---|
| 正收益集中在少数数据集 | 高 | 阴性对照本身是贡献；h 系列补两格强 gap 数据集 |
| HCAN/FreDF 粉丝审稿人不满 | 中 | 引用其原文数字、不依赖我们的复现端口；立场是「补全适用条件」 |
| 描述子头线被 FSP 追撞 | 中 | 已划界（时序结构描述子 + 决策用途；FSP 是 LLM 预训练） |
| 时间（ICLR 2027 截稿 ~9 月底） | 高 | 若赶不上，转 NeurIPS 2027 / KDD 2027，或先挂 arXiv |
| gap 诊断的锚点敏感性 | 低-中 | 附录敏感性分析（ctxmean vs seasonal 差值报告） |

## 7. 写作排期建议

- 本周：Intro + Theory + Diagnostic 三节初稿（不依赖 h 系列结果）；
- h 系列出数后：§5 主表 + §6 修复节 + §7 决策节定稿；
- 附图走 visual composer 流程（图 2 热力图 + 图 3 曲线优先）。
