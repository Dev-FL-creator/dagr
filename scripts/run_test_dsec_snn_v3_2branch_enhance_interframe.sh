#!/usr/bin/env bash
set -euo pipefail

# GPU selection (optional)
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Disable wandb if needed
export WANDB_MODE=disabled

# Python command
PYTHON=python
TEST_SCRIPT=scripts/test_dsec_snn_v3_2branch_enhance_interframe.py

# Output directory
OUTPUT_DIR=/media/data/hucao/jinkai/dagr/logs_snn_fusion_v3
EXP_NAME=fusion_event_image_sdtv3_bs4_2branch_enhance_interframe

# ------------------------------------------------------------------------------
# SDT-V3 Model Configuration 
# ------------------------------------------------------------------------------

BACKBONE_TYPE=sdtv3
SDT_T=4                     # Time slices
SDT_IN_CHANNELS=2          # Event polarity channels
SDT_MLP_RATIO=4.0
SDT_NORM=4.0
SDT_CHECKPOINT=0           # Checkpointing (not needed for inference, but keep consistent)

# --- Model Architecture ---
# If you trained with pretrained weights (19M with 360 dim):
# SDT_EMBED_DIMS="64 128 256 360"

# If you trained from scratch with standard dims:
SDT_EMBED_DIMS="64 128 256 512"

SDT_DEPTHS="2 2 6 2"
SDT_NUM_HEADS=8
SDT_SR_RATIO=4

# ------------------------------------------------------------------------------
# Interframe Test Configuration
# ------------------------------------------------------------------------------

# Batch size for testing
BATCH_SIZE=4

# Number of time window steps to test
NUM_INTERFRAME_STEPS=10

# Dataset
DATASET=DSEC_Det
DATASET_DIR=/media/data/hucao/zhenwu/hucao/DSEC

# Checkpoint path
CHECKPOINT="${OUTPUT_DIR}/DSEC_Det/detection/fusion_event_image_sdtv3_bs4_2branch_enhance/last_model.pth"

# Image backbone 
IMG_NET=resnet50

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_test_$(date +%Y%m%d_%H%M%S).log"

echo "================================================"
echo "Interframe Testing Dual-Branch SDT-V3 Model (Event + Image)"
echo "================================================"
echo "Log file: $LOG_FILE"
echo "Checkpoint: $CHECKPOINT"
echo "Backbone: $BACKBONE_TYPE"
echo "Dims: ${SDT_EMBED_DIMS}"
echo "Image Network: ${IMG_NET}"
echo "Interframe Steps: ${NUM_INTERFRAME_STEPS}"
echo "================================================"

# Check if checkpoint exists
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script."
    exit 1
fi

echo "Starting interframe testing..."

# Build command arguments
CMD_ARGS=(
  --config config/dagr-s-dsec.yaml
  --dataset "$DATASET"
  --output_directory "$OUTPUT_DIR"
  --exp_name "$EXP_NAME"
  --batch_size "$BATCH_SIZE"
  --num_interframe_steps "$NUM_INTERFRAME_STEPS"
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
  --use_image
  --img_net "$IMG_NET"
  --no_eval
)

$PYTHON "$TEST_SCRIPT" "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "================================================"
echo "Interframe testing completed. Results saved to:"
echo "  Log: $LOG_FILE"
echo "  Detections: ${OUTPUT_DIR}/DSEC_Det/detection/${EXP_NAME}/"
echo "================================================"
echo ""
echo "To visualize results, use:"
echo "python scripts/visualize_detections.py \\"
echo "  --detections_folder ${OUTPUT_DIR}/DSEC_Det/detection/${EXP_NAME}/ \\"
echo "  --sequence zurich_city_13_b \\"
echo "  --write_to_output"
echo "================================================"