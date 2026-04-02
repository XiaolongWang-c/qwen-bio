#!/usr/bin/env sh
set -eu

SCRIPT_PATH="$0"
case "${SCRIPT_PATH}" in
  /*) ;;
  *) SCRIPT_PATH="$(pwd)/${SCRIPT_PATH}" ;;
esac

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${SCRIPT_PATH}")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="${1:-${QWEN_IMAGE_EDIT_MODEL:-/sda/wangxl/ai-bio/qwen-image-edit}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-micro/generated_qwen_test8}"
LIMIT="${LIMIT:-8}"
TRAIN_COUNT="${TRAIN_COUNT:-2400}"
TEST_COUNT="${TEST_COUNT:-600}"
SEED="${SEED:-20260402}"
STEPS="${STEPS:-28}"
CFG_SCALE="${CFG_SCALE:-3.2}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
BLEND_STRENGTH="${BLEND_STRENGTH:-0.42}"
DETAIL_SIGMA="${DETAIL_SIGMA:-1.1}"
MASK_QUANTILE="${MASK_QUANTILE:-94}"

echo "Project root: ${PROJECT_ROOT}"
echo "Model path: ${MODEL_PATH}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Limit per split: ${LIMIT}"

python micro/micro_qwen_dataset_pipeline.py \
  --model-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --subset-train-count "${TRAIN_COUNT}" \
  --subset-test-count "${TEST_COUNT}" \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --num-inference-steps "${STEPS}" \
  --true-cfg-scale "${CFG_SCALE}" \
  --blend-strength "${BLEND_STRENGTH}" \
  --detail-sigma "${DETAIL_SIGMA}" \
  --mask-quantile "${MASK_QUANTILE}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --overwrite

echo
echo "Generated images:"
echo "  ${PROJECT_ROOT}/${OUTPUT_ROOT}/train/Micro/image"
echo "  ${PROJECT_ROOT}/${OUTPUT_ROOT}/test/Micro/image"
echo "Raw model outputs:"
echo "  ${PROJECT_ROOT}/${OUTPUT_ROOT}/raw_model_output/train/Micro/image"
echo "  ${PROJECT_ROOT}/${OUTPUT_ROOT}/raw_model_output/test/Micro/image"
echo "Source subset:"
echo "  ${PROJECT_ROOT}/${OUTPUT_ROOT}/source_subset/train/Micro/image"
echo "  ${PROJECT_ROOT}/${OUTPUT_ROOT}/source_subset/test/Micro/image"
echo "Preview:"
echo "  ${PROJECT_ROOT}/${OUTPUT_ROOT}/records/train_preview.png"
echo "  ${PROJECT_ROOT}/${OUTPUT_ROOT}/records/test_preview.png"
