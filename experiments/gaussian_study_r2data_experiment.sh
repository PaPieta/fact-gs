#!/bin/bash

set -euo pipefail
shopt -s nullglob

DEFAULT_MAIN_ROOT="$(pwd)"
MAIN_ROOT="${MAIN_ROOT:-${DEFAULT_MAIN_ROOT}}"
DATA_ROOT="${MAIN_ROOT}/data"
GAUSSIAN_STUDY_ROOT="${MAIN_ROOT}/models/gaussian_study_r2"

ITERATIONS=400
SPLIT="cone_ntrain_50_angle_360"
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
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STEPS_PER_RUN=$((ITERATIONS * 50))

run_case() {
    local dataset_name="$1"
    local data_path="$2"

    for gaussian_count in "${GAUSSIAN_COUNTS[@]}"; do
        local model_path="${GAUSSIAN_STUDY_ROOT}/${dataset_name}/${gaussian_count}"
        mkdir -p "${model_path}"

        echo "Running ${dataset_name} with ${gaussian_count} Gaussians"
        python train_recon.py \
            model.data_source_path="${data_path}" \
            model.model_path="${model_path}" \
            optim.steps="${STEPS_PER_RUN}" \
            model.num_gaussians="${gaussian_count}"
    done
}

process_domain() {
    local domain="$1"
    local split_path="${DATA_ROOT}/${domain}/${SPLIT}"

    if [[ ! -d "${split_path}" ]]; then
        echo "Skipping missing path: ${split_path}" >&2
        return
    fi

    for dataset_dir in "${split_path}"/*; do
        [[ -d "${dataset_dir}" ]] || continue
        local dataset_name
        dataset_name="$(basename "${dataset_dir}")"
        run_case "${dataset_name}" "${dataset_dir}"
    done
}

process_domain "real_dataset"
process_domain "synthetic_dataset"

echo "Collecting Gaussian study metrics"
python experiments/helpers/collect_gaussian_study_r2data_results.py \
    --study-root "${GAUSSIAN_STUDY_ROOT}" \
    --output-csv "${GAUSSIAN_STUDY_ROOT}/gaussian_study_metrics.csv"
