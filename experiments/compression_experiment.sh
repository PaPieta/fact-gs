#!/bin/bash

set -euo pipefail

DEFAULT_MAIN_ROOT="$(pwd)"
MAIN_ROOT="${MAIN_ROOT:-${DEFAULT_MAIN_ROOT}}"

real_base="${MAIN_ROOT}/data/real_dataset/cone_ntrain_25_angle_360"
synthetic_base="${MAIN_ROOT}/data/synthetic_dataset/cone_ntrain_25_angle_360"
model_save="${MAIN_ROOT}/models/vol_compress"

real_datasets=(
    pine
    seashell
    walnut
)

synthetic_datasets=(
    0_chest_cone
    0_foot_cone
    0_head_cone
    0_jaw_cone
    0_pancreas_cone
    1_beetle_cone
    1_bonsai_cone
    1_broccoli_cone
    1_kingsnake_cone
    1_pepper_cone
    2_backpack_cone
    2_engine_cone
    2_mount_cone
    2_present_cone
    2_teapot_cone
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

run_case() {
    local dataset_name="$1"
    local data_path="$2"

    echo "Evaluating ${dataset_name}"
    python train_volume.py \
        model=model_compress_volume \
        optim=optim_compress_volume \
        eval=eval_500 \
        model.data_source_path="${data_path}" \
        model.model_path="${model_save}/${dataset_name}"
}

for dataset in "${real_datasets[@]}"; do
    run_case "${dataset}" "${real_base}/${dataset}"
done

for dataset in "${synthetic_datasets[@]}"; do
    run_case "${dataset}" "${synthetic_base}/${dataset}"
done

echo "Running baseline compression tests"
python experiments/helpers/test_baseline_compression.py

echo "Combining results"
python experiments/helpers/collect_compression_results.py \
    --results-root "${model_save}" \
    --baseline-csv "${model_save}/baseline_compression.csv" \
    --output-csv "${model_save}/compression_combined.csv"
