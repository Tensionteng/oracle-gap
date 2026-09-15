#!/usr/bin/env bash
# 生成第二波实验任务：ETT 长预测长度 + Weather/ECL/Traffic
set -euo pipefail
ROOT=${MTP4TS_ROOT}
S=$ROOT/Time-Series-Library/scripts/long_term_forecast
Q=$ROOT/experiments/queue

block() { # 提取规范脚本中第 N 个 run.py 块
  awk -v n="$2" '/^python -u run\.py/{c++} {if(c==n){if(/^$/)exit; print}}' "$1"
}

emit() { # $1=输出名; 从 stdin 读块，做公共变换后写入队列
  sed 's/^python -u /uv run python -u /' \
  | { echo '#!/usr/bin/env bash'; echo 'set -euo pipefail'; echo "cd $ROOT/Time-Series-Library"; cat; } > "$Q/$1"
  chmod +x "$Q/$1"
}

to_desc() { # $1=原模型名；输出 desc 变体变换（含尾行追加 flags）
  sed -E "s/--model $1 /--model $1Desc /; s/--model_id (\S+)/--model_id \1_desc/; s/--des 'Exp'/--des 'Exp_desc'/; \$ s/\$/ --aux_head desc --aux_weight 1.0/"
}
to_fredf() {
  sed -E "s/--model_id (\S+)/--model_id \1_fredf/; s/--des 'Exp'/--des 'Exp_fredf'/; \$ s/\$/ --task_loss fredf/"
}

# --- ETT 长预测长度：PatchTST / iTransformer × {ETTh1,ETTm1} × {192,336,720} ---
for ds in ETTh1 ETTm1; do
  for spec in "2 192" "3 336" "4 720"; do
    set -- $spec; n=$1; pl=$2
    block $S/ETT_script/PatchTST_${ds}.sh $n | sed 's/\$model_name/PatchTST/g' \
      | emit "40_base_patchtst_${ds}_pl${pl}.sh"
    block $S/ETT_script/PatchTST_${ds}.sh $n | sed 's/\$model_name/PatchTST/g' | to_desc PatchTST \
      | emit "41_desc_patchtst_${ds}_pl${pl}.sh"
    block $S/ETT_script/iTransformer_ETTh2.sh $n | sed "s/ETTh2/${ds}/g; s/\$model_name/iTransformer/g" \
      | emit "42_base_itransformer_${ds}_pl${pl}.sh"
    block $S/ETT_script/iTransformer_ETTh2.sh $n | sed "s/ETTh2/${ds}/g; s/\$model_name/iTransformer/g" | to_desc iTransformer \
      | emit "43_desc_itransformer_${ds}_pl${pl}.sh"
  done
done

# --- Weather / ECL / Traffic pl96（第 1 块） ---
for m in PatchTST iTransformer; do
  lc=$(echo $m | tr 'A-Z' 'a-z')
  block $S/Weather_script/${m}.sh 1 | sed "s/\$model_name/${m}/g" | emit "44_base_${lc}_weather.sh"
  block $S/Weather_script/${m}.sh 1 | sed "s/\$model_name/${m}/g" | to_desc $m | emit "45_desc_${lc}_weather.sh"
  block $S/Weather_script/${m}.sh 1 | sed "s/\$model_name/${m}/g" | to_fredf | emit "46_fredf_${lc}_weather.sh"
done
for m in PatchTST iTransformer DLinear; do
  lc=$(echo $m | tr 'A-Z' 'a-z')
  block $S/ECL_script/${m}.sh 1 | sed "s/\$model_name/${m}/g" | emit "47_base_${lc}_ECL.sh"
  block $S/ECL_script/${m}.sh 1 | sed "s/\$model_name/${m}/g" | to_desc $m | emit "48_desc_${lc}_ECL.sh"
  block $S/Traffic_script/${m}.sh 1 2>/dev/null | sed "s/\$model_name/${m}/g" | emit "49_base_${lc}_traffic.sh" || true
done
for m in PatchTST iTransformer; do
  lc=$(echo $m | tr 'A-Z' 'a-z')
  block $S/Traffic_script/${m}.sh 1 | sed "s/\$model_name/${m}/g" | to_desc $m | emit "50_desc_${lc}_traffic.sh"
done
ls $Q/ | wc -l