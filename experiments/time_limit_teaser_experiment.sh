#!/bin/bash

set -euo pipefail
shopt -s nullglob

DEFAULT_MAIN_ROOT="$(pwd)"
MAIN_ROOT="${MAIN_ROOT:-${DEFAULT_MAIN_ROOT}}"
DATA_ROOT="${MAIN_ROOT}/data/teaser_figure_data"
MODEL_ROOT="${MAIN_ROOT}/models/time_limit_teaser"

declare -A GAUSSIAN_COUNTS=(
    [0_chest_cone]=50000
    [walnut_dtu_512_cone]=100000
    [coral1k_cone]=200000
)

declare -A TIME_LIMITS=(
    [0_chest_cone]=30    # seconds
    [walnut_dtu_512_cone]=60   # seconds
    [coral1k_cone]=120   # seconds
)

ORDERED_CASES=(
    0_chest_cone
    walnut_dtu_512_cone
    coral1k_cone
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

run_case() {
    local dataset_name="$1"
    local data_path="$2"
    local gaussian_count="$3"
    local time_limit="$4"
    local model_path="${MODEL_ROOT}/${dataset_name}"

    mkdir -p "${model_path}"
    echo "Running ${dataset_name}: ${gaussian_count} Gaussians, training_time_limit_seconds=${time_limit}"
    python train_recon.py \
        model.data_source_path="${data_path}" \
        model.model_path="${model_path}" \
        model.num_gaussians="${gaussian_count}" \
        optim.steps="20000" \
        optim.max_num_gaussians="${gaussian_count}" \
        optim.training_time_limit_seconds="${time_limit}"
}

for dataset_name in "${ORDERED_CASES[@]}"; do
    data_path="${DATA_ROOT}/${dataset_name}"
    [[ -d "${data_path}" ]] || continue
    gaussian_count="${GAUSSIAN_COUNTS[${dataset_name}]:-}"
    time_limit="${TIME_LIMITS[${dataset_name}]:-}"
    if [[ -z "${gaussian_count}" || -z "${time_limit}" ]]; then
        echo "Skipping ${dataset_name}: missing gaussian count or time limit." >&2
        continue
    fi
    run_case "${dataset_name}" "${data_path}" "${gaussian_count}" "${time_limit}"
done
