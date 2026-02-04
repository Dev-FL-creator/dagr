#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=disabled

PYTHON=python
TEST_SCRIPT=scripts/test_dsec_rvt_2branch.py

OUTPUT_DIR=/media/data/hucao/jinkai/dagr/logs_rvt_fusion
EXP_NAME=fusion_event_image_rvt_bs8_2branch

BACKBONE_TYPE=rvt
RVT_IN_CHANNELS=2
RVT_DIM_HEAD=32
RVT_USE_LSTM=0
RVT_RETURN_TEMPORAL=0

RVT_EMBED_DIMS="64 128 256 512"
RVT_DEPTHS="2 2 6 2"
RVT_PARTITION_SIZE="7 5"

PRETRAINED_WEIGHT=""

BATCH_SIZE=4

DATASET=DSEC_Det
DATASET_DIR=/media/data/hucao/zhenwu/hucao/DSEC

#CHECKPOINT="${OUTPUT_DIR}/DSEC_Det/detection/${EXP_NAME}/last_model.pth"
CHECKPOINT="${OUTPUT_DIR}/DSEC_Det/detection/${EXP_NAME}/best_model_mAP_0.42215826130562834.pth"


LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_DIR"

echo "Testing log will be saved to: $LOG_FILE"
echo "Starting testing with RVT 2-branch fusion..."
echo "Checkpoint: $CHECKPOINT"
echo "Dims: ${RVT_EMBED_DIMS}"
echo "Depths: ${RVT_DEPTHS}"

COMMON_ARGS=(
  --config config/dagr-rvt-dsec.yaml
  --dataset "$DATASET"
  --output_directory "$OUTPUT_DIR"
  --exp_name "${EXP_NAME}_test"
  --batch_size "$BATCH_SIZE"
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
  --checkpoint "$CHECKPOINT"
)

if [[ "${RVT_USE_LSTM:-0}" -eq 1 ]]; then
  COMMON_ARGS+=(--rvt_use_lstm)
fi

if [[ "${RVT_RETURN_TEMPORAL:-0}" -eq 1 ]]; then
  COMMON_ARGS+=(--rvt_return_temporal)
fi

if [[ -n "${PRETRAINED_WEIGHT}" ]]; then
  COMMON_ARGS+=(--load_pretrained_weight "$PRETRAINED_WEIGHT")
fi

$PYTHON "$TEST_SCRIPT" \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

