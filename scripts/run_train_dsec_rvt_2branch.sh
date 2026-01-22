#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DISTRIBUTED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export WANDB_MODE=disabled
export NO_EVAL=${NO_EVAL:-0}

# Python 与 启动命令
PYTHON=python
TORCHRUN=torchrun
TRAIN_SCRIPT=scripts/train_dsec_rvt_2branch.py

OUTPUT_DIR=/media/data/hucao/jinkai/dagr/logs_rvt_fusion
EXP_NAME=fusion_event_image_rvt_bs2_2branch
#EXP_NAME=fusion_event_image_rvt_bs8_test

# ------------------------------------------------------------------------------
# 模型配置切换区 (RVT Backbone 配置)
# ------------------------------------------------------------------------------

# RVT 基础配置
BACKBONE_TYPE=rvt
RVT_IN_CHANNELS=2          # 事件极性通道数
RVT_DIM_HEAD=32           # Attention head dimension
RVT_USE_LSTM=0            # 是否使用 LSTM 时序建模 (0=false, 1=true)
RVT_RETURN_TEMPORAL=0     # 是否返回时序特征 (0=false, 1=true)

# --- [选项 1: 小模型配置 - 快速实验] ---
# ----------------------------------------------------------
# RVT_EMBED_DIMS="64 128 256 384"
# RVT_DEPTHS="1 1 3 1"
# RVT_PARTITION_SIZE="7 7"

# --- [选项 2: 标准配置 - 推荐] (默认) ---
# ----------------------------------------------------------
RVT_EMBED_DIMS="64 128 256 512"  # Only 4 values to disable Stage5
RVT_DEPTHS="2 2 6 2"
RVT_PARTITION_SIZE="7 5"  # Changed to 7x5 for better divisibility
# Stage2=28x40: 28÷7=4✓, 40÷5=8✓
# Stage3=14x20: 14÷7=2✓, 20÷5=4✓
# Stage4=7x10:  7÷7=1✓, 10÷5=2✓
# Note: Only 4 embed_dim values to avoid creating Stage5 (which would be 3x5, too small for partition)

# --- [选项 3: 大模型配置 - 高性能] ---
# ----------------------------------------------------------
# RVT_EMBED_DIMS="128 256 512 640"
# RVT_DEPTHS="3 4 12 3"
# RVT_PARTITION_SIZE="11 11"

# --- [可选: 预训练权重加载] ---
# ----------------------------------------------------------
PRETRAINED_WEIGHT=""            # 留空表示不加载预训练权重
# PRETRAINED_WEIGHT="/path/to/rvt_pretrained.pth"  # 取消注释以加载权重

# ------------------------------------------------------------------------------
# 训练超参数与数据集
# ------------------------------------------------------------------------------

# 训练参数 (per-GPU)
BATCH_SIZE=2
EPOCHS=801
LR=0.0002
WEIGHT_DECAY=0.00001

# 数据集设置
DATASET=DSEC_Det
EXP_TREND=full           # fast | mid | full
DATASET_DIR=/media/data/hucao/zhenwu/hucao/DSEC

# 日志文件
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Training log will be saved to: $LOG_FILE"
echo "Starting training with RVT backbone..."
echo "Mode: PRETRAINED_WEIGHT='${PRETRAINED_WEIGHT}'"
echo "Dims: ${RVT_EMBED_DIMS}"
echo "Depths: ${RVT_DEPTHS}"

# 可选标志处理
NO_EVAL_FLAG=()
if [[ "${NO_EVAL}" -eq 1 ]]; then
  NO_EVAL_FLAG+=(--no_eval)
fi

# 组装通用参数
COMMON_ARGS=(
  --config config/dagr-rvt-dsec.yaml
  --dataset "$DATASET"
  --output_directory "$OUTPUT_DIR"
  --exp_name "$EXP_NAME"
  --batch_size "$BATCH_SIZE"
  --tot_num_epochs "$EPOCHS"
  --l_r "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --exp_trend "$EXP_TREND"
  --use_snn_backbone
  --backbone_type "$BACKBONE_TYPE"
  --rvt_in_channels "$RVT_IN_CHANNELS"
  --rvt_embed_dim $RVT_EMBED_DIMS
  --rvt_depths $RVT_DEPTHS
  --rvt_dim_head "$RVT_DIM_HEAD"
  --rvt_partition_size $RVT_PARTITION_SIZE
  --dataset_directory "$DATASET_DIR"
  --use_image
  --img_net resnet50
  "${NO_EVAL_FLAG[@]}"
)

# RVT LSTM 时序建模标志
if [[ "${RVT_USE_LSTM:-0}" -eq 1 ]]; then
  COMMON_ARGS+=(--rvt_use_lstm)
fi

# RVT 时序输出标志
if [[ "${RVT_RETURN_TEMPORAL:-0}" -eq 1 ]]; then
  COMMON_ARGS+=(--rvt_return_temporal)
fi

# 仅当 PRETRAINED_WEIGHT 非空时，才添加加载参数
if [[ -n "${PRETRAINED_WEIGHT}" ]]; then
  COMMON_ARGS+=(--load_pretrained_weight "$PRETRAINED_WEIGHT")
fi

# 分布式启动 (DISTRIBUTED=1)
if [[ "${DISTRIBUTED:-0}" -eq 1 ]]; then
  # 自动推断 GPU 数量
  if [[ -z "${NUM_GPUS:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -r -a DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
      NUM_GPUS=${#DEV_ARR[@]}
    else
      NUM_GPUS=1
    fi
  fi
  $TORCHRUN --nproc_per_node="$NUM_GPUS" "$TRAIN_SCRIPT" \
    --distributed \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

# 单卡启动
else
  $PYTHON "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
fi
