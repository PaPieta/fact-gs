#!/bin/bash

set -euo pipefail
shopt -s nullglob

DEFAULT_MAIN_ROOT="$(pwd)"
MAIN_ROOT="${MAIN_ROOT:-${DEFAULT_MAIN_ROOT}}"
CORAL_DATASET_ROOT="${MAIN_ROOT}/data/coral_dataset"
STUDY_ROOT="${MAIN_ROOT}/models/scaling_study_coral_090ssim"
RESULTS_CSV="${STUDY_ROOT}/scaling_study_coral_090ssim_times.csv"

declare -A GAUSSIAN_COUNTS=(
    [coral256_cone]=50000
    [coral384_cone]=75000
    [coral512_cone]=100000
    [coral768_cone]=150000
    [coral1k_cone]=200000
)

ORDERED_DATASETS=(
    coral256_cone
    coral384_cone
    coral512_cone
    coral768_cone
    coral1k_cone
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

run_case() {
    local dataset_name="$1"
    local data_path="$2"
    local gaussian_count="$3"
    local model_path="${STUDY_ROOT}/${dataset_name}"

    mkdir -p "${model_path}"
    echo "Running ${dataset_name} with ${gaussian_count} Gaussians (SSIM early-stop @0.9)"
    python train_recon.py \
        eval=eval_100 \
        eval.extra_eval_iter_num=300 \
        optim=optim_recon_ssim090 \
        model.data_source_path="${data_path}" \
        model.model_path="${model_path}" \
        model.num_gaussians="${gaussian_count}" \
        optim.max_num_gaussians="${gaussian_count}"
}

for dataset_name in "${ORDERED_DATASETS[@]}"; do
    data_path="${CORAL_DATASET_ROOT}/${dataset_name}"
    [[ -d "${data_path}" ]] || continue
    gaussian_count="${GAUSSIAN_COUNTS[${dataset_name}]:-}"
    if [[ -z "${gaussian_count}" ]]; then
        echo "Skipping dataset ${dataset_name}: no gaussian count mapping" >&2
        continue
    fi
    run_case "${dataset_name}" "${data_path}" "${gaussian_count}"
done

echo "Collecting coral SSIM-0.90 scaling study times"
python experiments/helpers/collect_scaling_study_coral_results.py \
    --study-root "${STUDY_ROOT}" \
    --output-csv "${RESULTS_CSV}"
