#!/bin/bash

set -euo pipefail
shopt -s nullglob

DEFAULT_MAIN_ROOT="$(pwd)"
MAIN_ROOT="${MAIN_ROOT:-${DEFAULT_MAIN_ROOT}}"
CORAL_DATASET_ROOT="${MAIN_ROOT}/data/coral_dataset"
GAUSSIAN_STUDY_CORAL_ROOT="${MAIN_ROOT}/models/gaussian_study_coral"

ITERATIONS=400
VIEW_COUNT=50
GAUSSIAN_COUNTS=(
    5000
    10000
    20000
    30000
    40000
    50000
    60000
    70000
    80000
    90000
    100000
    125000
    150000
    175000
    200000
    225000
    250000
    275000
    300000
    325000
    350000
    375000
    400000
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

run_case() {
    local dataset_name="$1"
    local data_path="$2"
    local steps=$((ITERATIONS * VIEW_COUNT))

    for gaussian_count in "${GAUSSIAN_COUNTS[@]}"; do
        local model_path="${GAUSSIAN_STUDY_CORAL_ROOT}/${dataset_name}/${gaussian_count}"
        mkdir -p "${model_path}"

        echo "Running ${dataset_name} (${VIEW_COUNT} views) with ${gaussian_count} Gaussians"
        python train_recon.py \
            model.data_source_path="${data_path}" \
            model.model_path="${model_path}" \
            optim.steps="${steps}" \
            model.num_gaussians="${gaussian_count}" \
            optim.max_num_gaussians="${gaussian_count}"
    done
}

for dataset_dir in "${CORAL_DATASET_ROOT}"/*; do
    [[ -d "${dataset_dir}" ]] || continue
    dataset_name="$(basename "${dataset_dir}")"
    run_case "${dataset_name}" "${dataset_dir}"
done

echo "Collecting coral Gaussian study metrics"
python experiments/helpers/collect_gaussian_study_coraldata_results.py \
    --study-root "${GAUSSIAN_STUDY_CORAL_ROOT}" \
    --output-csv "${GAUSSIAN_STUDY_CORAL_ROOT}/coral_gaussian_study_metrics.csv"
