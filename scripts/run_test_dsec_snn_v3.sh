#!/usr/bin/env bash
set -euo pipefail

# GPU selection (optional)
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Disable wandb if needed
export WANDB_MODE=disabled

# Python command
PYTHON=python
TEST_SCRIPT=scripts/test_dsec_snn_v3.py

# Output directory (should match training)
OUTPUT_DIR=/media/data/hucao/jinkai/dagr/logs_snn_v3
EXP_NAME=event_only_sdtv3_4_Timeslices_bs_4

# ------------------------------------------------------------------------------
# Mode Selection: Single Event Branch vs. Dual Branch Fusion
# ------------------------------------------------------------------------------
# - To test SINGLE EVENT BRANCH (event-only): set USE_IMAGE=0
# - To test DUAL BRANCH FUSION (event+image): set USE_IMAGE=1
# ------------------------------------------------------------------------------
USE_IMAGE=0  # 设置为 0 表示单事件分支，设置为 1 表示双分支融合

# ------------------------------------------------------------------------------
# SDT-V3 Model Configuration (Must match training configuration)
# ------------------------------------------------------------------------------

BACKBONE_TYPE=sdtv3
SDT_T=4                     # Time slices
SDT_IN_CHANNELS=2          # Event polarity channels
SDT_MLP_RATIO=4.0
SDT_NORM=4.0
SDT_CHECKPOINT=0           # Checkpointing (not needed for inference, but keep consistent)

# --- Model Architecture (Must match training) ---
# If you trained with pretrained weights (19M with 360 dim):
# SDT_EMBED_DIMS="64 128 256 360"

# If you trained from scratch with standard dims:
SDT_EMBED_DIMS="64 128 256 512"

SDT_DEPTHS="2 2 6 2"
SDT_NUM_HEADS=8
SDT_SR_RATIO=4

# ------------------------------------------------------------------------------
# Test Configuration
# ------------------------------------------------------------------------------

# Batch size for testing (原始分辨率需要更多显存，适当降低 batch size)
BATCH_SIZE=4

# Dataset
DATASET=DSEC_Det
DATASET_DIR=/media/data/hucao/zhenwu/hucao/DSEC

# Checkpoint path - UPDATE THIS to your actual checkpoint
CHECKPOINT="${OUTPUT_DIR}/DSEC_Det/detection/${EXP_NAME}/last_model.pth"

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_test_$(date +%Y%m%d_%H%M%S).log"

echo "================================================"
echo "Testing SDT-V3 Model"
echo "================================================"
echo "Mode: $([ $USE_IMAGE -eq 1 ] && echo 'Dual Branch (Event+Image)' || echo 'Single Event Branch')"
echo "Log file: $LOG_FILE"
echo "Checkpoint: $CHECKPOINT"
echo "Backbone: $BACKBONE_TYPE"
echo "Dims: ${SDT_EMBED_DIMS}"
echo "================================================"

# Check if checkpoint exists
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script."
    exit 1
fi

echo "Starting testing..."

# Build command arguments
CMD_ARGS=(
  --config config/dagr-s-dsec.yaml
  --dataset "$DATASET"
  --output_directory "$OUTPUT_DIR"
  --exp_name "${EXP_NAME}_test"
  --batch_size "$BATCH_SIZE"
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
  --checkpoint "$CHECKPOINT"
)

# Add --use_image flag only if USE_IMAGE=1
if [[ $USE_IMAGE -eq 1 ]]; then
  CMD_ARGS+=(--use_image)
fi

$PYTHON "$TEST_SCRIPT" "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "================================================"
echo "Testing completed. Results saved to: $LOG_FILE"
echo "================================================"
