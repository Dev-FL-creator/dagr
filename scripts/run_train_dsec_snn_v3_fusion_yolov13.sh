#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DISTRIBUTED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export WANDB_MODE=disabled
export NO_EVAL=${NO_EVAL:-0}

PYTHON=python
TORCHRUN=torchrun
TRAIN_SCRIPT=scripts/train_dsec_snn_v3_2branch_yolov13.py

OUTPUT_DIR=/media/data/hucao/jinkai/dagr/logs_snn_fusion_v3_yolo13
EXP_NAME=fusion_event_image_sdtv3_bs4_2branch_yolo13

BACKBONE_TYPE=sdtv3
SDT_T=4
SDT_IN_CHANNELS=2
SDT_MLP_RATIO=4.0
SDT_NORM=4.0
SDT_CHECKPOINT=0

PRETRAINED_WEIGHT=""
SDT_EMBED_DIMS="64 128 256 512"
SDT_DEPTHS="2 2 6 2"
SDT_NUM_HEADS=8
SDT_SR_RATIO=4

BATCH_SIZE=4
EPOCHS=801
LR=0.0002
WEIGHT_DECAY=0.00001

DATASET=DSEC_Det
EXP_TREND=full
DATASET_DIR=/media/data/hucao/zhenwu/hucao/DSEC

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Training log will be saved to: $LOG_FILE"
echo "Starting training..."
echo "Mode: PRETRAINED_WEIGHT='${PRETRAINED_WEIGHT}'"
echo "Dims: ${SDT_EMBED_DIMS}"

NO_EVAL_FLAG=()
if [[ "${NO_EVAL}" -eq 1 ]]; then
  NO_EVAL_FLAG+=(--no_eval)
fi


COMMON_ARGS=(
  --config config/dagr-s-dsec.yaml
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
  --sdt_T "$SDT_T"
  --sdt_in_channels "$SDT_IN_CHANNELS"
  --sdt_embed_dim $SDT_EMBED_DIMS
  --sdt_depths $SDT_DEPTHS
  --sdt_num_heads "$SDT_NUM_HEADS"
  --sdt_mlp_ratio "$SDT_MLP_RATIO"
  --sdt_norm "$SDT_NORM"
  --sdt_sr_ratio "$SDT_SR_RATIO"
  --dataset_directory "$DATASET_DIR"
  --use_image
  --img_net resnet50
  "${NO_EVAL_FLAG[@]}"
)

if [[ "${SDT_CHECKPOINT:-0}" -eq 1 ]]; then
  COMMON_ARGS+=(--use_checkpointing)
fi

if [[ -n "${PRETRAINED_WEIGHT}" ]]; then
  COMMON_ARGS+=(--load_pretrained_weight "$PRETRAINED_WEIGHT")
fi

USE_YOLOV13_HEAD=${USE_YOLOV13_HEAD:-1}
if [[ "${USE_YOLOV13_HEAD}" -eq 1 ]]; then
  COMMON_ARGS+=(--use_yolov13_head)
  echo "Using YOLOv13 head (requires ultralytics package installed)"
fi

if [[ "${DISTRIBUTED:-0}" -eq 1 ]]; then
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
else
  $PYTHON "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
fi