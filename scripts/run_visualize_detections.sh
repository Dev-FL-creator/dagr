#!/usr/bin/env bash
set -euo pipefail

# Python command
PYTHON=python
VIS_SCRIPT=scripts/visualize_detections.py

# ------------------------------------------------------------------------------
# Visualization Configuration
# ------------------------------------------------------------------------------

# Detections folder (where .npy files are located)
# Update this to match interframe test output directory
DETECTIONS_FOLDER=/media/data/hucao/jinkai/dagr/logs_snn_fusion_v3/DSEC_Det/detection/fusion_event_image_sdtv3_bs4_2branch_enhance_interframe

# DSEC dataset directory (test split)
DATASET_DIR=/media/data/hucao/zhenwu/hucao/DSEC/test

# Sequence to visualize (change as needed)
SEQUENCE=zurich_city_13_b

# Visualization parameters
VIS_TIME_STEP_US=1000       # Time step between frames (microseconds)
EVENT_TIME_WINDOW_US=5000   # Event time window for visualization (microseconds)

# Output mode: set to true to save images, false for real-time display
WRITE_TO_OUTPUT=true

# ------------------------------------------------------------------------------
# Available sequences (uncomment to visualize)
# ------------------------------------------------------------------------------
# SEQUENCE=thun_01_a
# SEQUENCE=thun_01_b
# SEQUENCE=thun_02_a
# SEQUENCE=interlaken_00_a
# SEQUENCE=interlaken_00_b
# SEQUENCE=interlaken_01_a
# SEQUENCE=zurich_city_12_a
# SEQUENCE=zurich_city_13_a
# SEQUENCE=zurich_city_13_b
# SEQUENCE=zurich_city_14_a
# SEQUENCE=zurich_city_14_b
# SEQUENCE=zurich_city_14_c
# SEQUENCE=zurich_city_15_a

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

echo "================================================"
echo "Visualizing DSEC Detection Results"
echo "================================================"
echo "Detections folder: $DETECTIONS_FOLDER"
echo "Dataset directory: $DATASET_DIR"
echo "Sequence: $SEQUENCE"
echo "Time step: ${VIS_TIME_STEP_US} μs"
echo "Event window: ${EVENT_TIME_WINDOW_US} μs"
echo "Write to output: $WRITE_TO_OUTPUT"
echo "================================================"

# Check if detections folder exists
if [[ ! -d "$DETECTIONS_FOLDER" ]]; then
    echo "ERROR: Detections folder not found: $DETECTIONS_FOLDER"
    echo "Please update the DETECTIONS_FOLDER variable or wait for interframe testing to complete."
    exit 1
fi

# Check if detection file exists for the specified sequence
DETECTION_FILE="${DETECTIONS_FOLDER}/detections_${SEQUENCE}.npy"
if [[ ! -f "$DETECTION_FILE" ]]; then
    echo "ERROR: Detection file not found: $DETECTION_FILE"
    echo "Available detection files:"
    find "$DETECTIONS_FOLDER" -name "detections_*.npy" -exec basename {} \; 2>/dev/null || echo "  None found"
    echo "Please check the sequence name or wait for interframe testing to complete."
    exit 1
fi

# Check if dataset directory exists
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    echo "Please update the DATASET_DIR variable."
    exit 1
fi

# Check if sequence directory exists in dataset
SEQUENCE_DIR="${DATASET_DIR}/${SEQUENCE}"
if [[ ! -d "$SEQUENCE_DIR" ]]; then
    echo "ERROR: Sequence directory not found: $SEQUENCE_DIR"
    echo "Available sequences:"
    find "$DATASET_DIR" -maxdepth 1 -type d -exec basename {} \; | grep -v "^test$" | sort || echo "  None found"
    exit 1
fi

echo "Starting visualization..."

# Build command arguments
CMD_ARGS=(
  --detections_folder "$DETECTIONS_FOLDER"
  --dataset_directory "$DATASET_DIR"
  --sequence "$SEQUENCE"
  --vis_time_step_us "$VIS_TIME_STEP_US"
  --event_time_window_us "$EVENT_TIME_WINDOW_US"
)

# Add write_to_output flag if enabled
if [[ "$WRITE_TO_OUTPUT" == "true" ]]; then
    CMD_ARGS+=(--write_to_output)
    OUTPUT_PATH="${DETECTIONS_FOLDER}/visualization"
    echo "Visualization images will be saved to: $OUTPUT_PATH"
else
    echo "Real-time visualization mode (press any key in OpenCV window to advance frames)"
fi

# Run visualization
$PYTHON "$VIS_SCRIPT" "${CMD_ARGS[@]}"

echo "================================================"
if [[ "$WRITE_TO_OUTPUT" == "true" ]]; then
    echo "Visualization completed! Images saved to:"
    echo "  ${DETECTIONS_FOLDER}/visualization/"
    echo ""
    echo "You can create a video using:"
    echo "  cd ${DETECTIONS_FOLDER}/visualization/"
    echo "  ffmpeg -framerate 10 -i %06d.png -c:v libx264 -pix_fmt yuv420p ${SEQUENCE}_visualization.mp4"
else
    echo "Visualization completed!"
fi
echo "================================================"