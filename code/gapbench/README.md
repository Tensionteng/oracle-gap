# gapbench — oracle-gap 诊断的 benchmark 化 harness

把 `Time-Series-Library/analysis/oracle_diagnostic.py` 的 oracle-gap 诊断推广到
GIFT-Eval 多序列集合 + TSFM 零样本模型。所有分析吃统一 dump 格式：
`{pred,true,inputs}.npy`（[N, pred_len, C] / [N, pred_len, C] / [N, seq_len, C]）。

## 全量扫描组件（2026-09-11 全量跑完）

- `windows_full.py`：盘点（inventory.json）→ 全部 (dataset, sub) 的 tail 窗口
  `windows_full/<key>_tail.npz` + manifest。规则：多变量 item 拍平成单变量序列；
  存活序列 <30 时改用 stride=96 多窗口（最多 20/序列）；窗口内 NaN 线性插值后
  仍有 NaN 或输入 std<1e-6 则丢窗；存活 <5 窗的 cell 跳过。
- `scan.py`（tsfm_stage2/.venv）：chronos 族——bolt-tiny/mini/small/base(full) +
  chronos-2（chronos2 输入为 [series, variate, len]，predict_quantiles 返回 list）。
- `scan_gen.py`（.venv-gen，transformers **4.40.2**——sundial/TimeMoE/timer 的
  自定义 generate 只在 4.40 时代 API 下工作；4.43+ 与 5.x 均挂）：TimeMoE 必须
  手工实例级 z-norm（其 mixin 无 revin，已实测 raw 输入塌缩）；timer 内部已归一
  （raw 与手工归一输出逐点一致）；sundial num_samples=20 取逐步中位数。
- `scan_moirai.py`（.venv-moirai，uni2ts **2.0.0 --no-deps** 覆盖在 1.1.1 依赖上，
  torch 保持 cu126）：moirai-1.1 small/large（采样 20 取中位数）与 moirai-2.0-R-small
  （原生 9 分位数头取 0.5）。
- `scan_tirex.py`（.venv-tirex，**tirex-ts** 包——pypi 上的 "tirex" 是同名 sklearn
  包；load_model 不认绝对路径，用 TiRexZero.from_pretrained 直载 ckpt）。
- `patchtst_arm.py` / `patchtst_arm.sh`：合格子集（等长、中位长度≥960、通道数百级、
  NaN 轻微）宽表 CSV + TSLib PatchTST 96→96（ETT 对齐超参）+ tail dump 收集
  （多序列=最后窗口；单 item 多通道=strided K≤20 窗，窗口位置与 windows_full
  逐点对齐，已数值验证 corr=1.0）。
- `collect_matrix.py` → `results/matrix.csv`；`analyze_matrix.py` 打四个切片。

## 试点流程（pilot 目录 dumps/ 为试点产物）

```bash
P=../tsfm_stage2   # uv 环境（chronos-forecasting, datasets, pandas）

# 1. arrow -> 窗口 npz（tail=每序列最后 192 步；tslib_test=TSLib custom 7/1/2 test 滑窗位置）
uv run --project $P python adapter.py --dataset solar --sub H --mode tail --out windows/solarH_tail.npz
uv run --project $P python adapter.py --dataset solar --sub H --mode tslib_test \
  --out windows/solarH_tslib_test.npz --export_csv csv/solar_H.csv

# 2. TSFM 零样本 dump（GPU）
CUDA_VISIBLE_DEVICES=0 uv run --project $P python tsfm_dump.py \
  --windows windows/solarH_tslib_test.npz --out dumps/solarH_tslib_test/bolt_zs

# 3. 任务模型：TSLib custom 协议训 PatchTST（cwd=TSLib，结果在 TSLib/results/）
cd ../Time-Series-Library && CUDA_VISIBLE_DEVICES=0 uv run python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ${MTP4TS_ROOT}/gapbench/csv/ \
  --data_path solar_H.csv --model_id gapbench_solarH_96_96 \
  --model PatchTST --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 1 --d_layers 1 --factor 3 --enc_in 137 --dec_in 137 --c_out 137 \
  --des Exp --n_heads 2 --itr 1 --freq h --num_workers 4

# 4. 收集统一 dump（--tail_only 取最后窗口=每序列 tail，与 TSFM tail dump 窗对窗对齐）
uv run --project $P python collect_tslib_dump.py --model_id gapbench_solarH_96_96 \
  --csv csv/solar_H.csv --out dumps/solarH_tslib_test/patchtst

# 5. gap（实例级标准化 + reps=20 自助残差置换 + CI95）
uv run --project $P python gap.py --dump dumps/solarH_tslib_test/patchtst --reps 20 \
  --tag patchtst@solarH --json_out dumps/solarH_tslib_test/patchtst/gap.json
```

## 约定

- dump 保持各模型原生空间（TSFM 原始尺度、PatchTST scaler 空间）；gap.py 内做
  每窗口×每通道输入 std 的实例级标准化后再池化，input std < 1e-6 的窗口丢弃。
- wide CSV 最后一列改名 `OT`（TSLib Dataset_Custom 要求）；`date` 为按数据集 freq
  合成的规则时间轴（pandas>=2 频率小写）。
- TSLib 侧验证：collect_tslib_dump.py 的 inputs 复刻逻辑已对 Dataset_Custom
  数值断言（首/末窗口逐点一致）。
- gap 符号判读沿用原工具：gap>+0.02 残差有可榨结构；|gap|<=0.02 理性收缩占满。

```bash
P=../tsfm_stage2   # uv 环境（chronos-forecasting, datasets, pandas）

# 1. arrow -> 窗口 npz（tail=每序列最后 192 步；tslib_test=TSLib custom 7/1/2 test 滑窗位置）
uv run --project $P python adapter.py --dataset solar --sub H --mode tail --out windows/solarH_tail.npz
uv run --project $P python adapter.py --dataset solar --sub H --mode tslib_test \
  --out windows/solarH_tslib_test.npz --export_csv csv/solar_H.csv

# 2. TSFM 零样本 dump（GPU）
CUDA_VISIBLE_DEVICES=0 uv run --project $P python tsfm_dump.py \
  --windows windows/solarH_tslib_test.npz --out dumps/solarH_tslib_test/bolt_zs

# 3. 任务模型：TSLib custom 协议训 PatchTST（cwd=TSLib，结果在 TSLib/results/）
cd ../Time-Series-Library && CUDA_VISIBLE_DEVICES=0 uv run python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ${MTP4TS_ROOT}/gapbench/csv/ \
  --data_path solar_H.csv --model_id gapbench_solarH_96_96 \
  --model PatchTST --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 1 --d_layers 1 --factor 3 --enc_in 137 --dec_in 137 --c_out 137 \
  --des Exp --n_heads 2 --itr 1 --freq h --num_workers 4

# 4. 收集统一 dump（--tail_only 取最后窗口=每序列 tail，与 TSFM tail dump 窗对窗对齐）
uv run --project $P python collect_tslib_dump.py --model_id gapbench_solarH_96_96 \
  --csv csv/solar_H.csv --out dumps/solarH_tslib_test/patchtst
uv run --project $P python collect_tslib_dump.py --model_id gapbench_solarH_96_96 \
  --csv csv/solar_H.csv --out dumps/solarH_tail/patchtst --tail_only

# 5. gap（实例级标准化 + reps=20 自助残差置换 + CI95）
uv run --project $P python gap.py --dump dumps/solarH_tslib_test/patchtst --reps 20 \
  --tag patchtst@solarH --json_out dumps/solarH_tslib_test/patchtst/gap.json
```

## 约定

- dump 保持各模型原生空间（TSFM 原始尺度、PatchTST scaler 空间）；gap.py 内做
  每窗口×每通道输入 std 的实例级标准化后再池化，input std < 1e-6 的窗口丢弃。
- wide CSV 最后一列改名 `OT`（TSLib Dataset_Custom 要求）；`date` 为按数据集 freq
  合成的规则时间轴（pandas>=2 频率小写）。
- TSLib 侧验证：collect_tslib_dump.py 的 inputs 复刻逻辑已对 Dataset_Custom
  数值断言（首/末窗口逐点一致）。
- gap 符号判读沿用原工具：gap>+0.02 残差有可榨结构；|gap|<=0.02 理性收缩占满。
