#!/usr/bin/env bash
# ============================================================
# Run Stage D (Targeted VLM Verification) independently
# ============================================================
# Usage:
#   bash scripts/run_stage_d.sh                          # all videos
#   bash scripts/run_stage_d.sh Abuse028_x264            # single video
#   bash scripts/run_stage_d.sh Abuse028_x264 --no-adv   # without adversarial
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATASET="${DATASET:-ucf_crime}"
BASE="data/${DATASET}"

FLAGGED_DIR="${BASE}/reflection/phase3_flagged"
CONTEXT_DIR="${BASE}/context/phase2"
VIDEO_FOLDER="${BASE}/videos"
ANNO_FILE="${BASE}/annotations/test.txt"
OUTPUT_DIR="${BASE}/captions/phase4_fine"
ROOT_PATH="${BASE}"

VIDEO_NAME="${1:-}"
ADVERSARIAL_FLAG="--adversarial"

shift 2>/dev/null || true
for arg in "$@"; do
    case "$arg" in
        --no-adv)  ADVERSARIAL_FLAG="" ;;
        --adv)     ADVERSARIAL_FLAG="--adversarial" ;;
        --dataset=*) DATASET="${arg#*=}"
                     BASE="data/${DATASET}"
                     FLAGGED_DIR="${BASE}/reflection/phase3_flagged"
                     CONTEXT_DIR="${BASE}/context/phase2"
                     VIDEO_FOLDER="${BASE}/videos"
                     ANNO_FILE="${BASE}/annotations/test.txt"
                     OUTPUT_DIR="${BASE}/captions/phase4_fine"
                     ROOT_PATH="${BASE}"
                     ;;
    esac
done

echo "============================================"
echo "  Stage D: Targeted VLM Verification"
echo "============================================"
echo "  Dataset:    ${DATASET}"
echo "  Adversarial: ${ADVERSARIAL_FLAG:-(none)}"
echo "  Video:      ${VIDEO_NAME:-(all)}"
echo "============================================"

CMD=(
    python src/pipeline/stage_d_targeted_verify.py
    --flagged_dir       "${FLAGGED_DIR}"
    --context_dir       "${CONTEXT_DIR}"
    --video_folder      "${VIDEO_FOLDER}"
    --annotationfile_path "${ANNO_FILE}"
    --output_dir        "${OUTPUT_DIR}"
    --root_path         "${ROOT_PATH}"
    ${ADVERSARIAL_FLAG}
)

if [ -n "${VIDEO_NAME}" ]; then
    CMD+=(--video_name "${VIDEO_NAME}")
fi

echo "  → ${CMD[*]}"
echo ""

"${CMD[@]}"
