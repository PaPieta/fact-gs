#!/bin/bash

set -euo pipefail
shopt -s nullglob

DEFAULT_MAIN_ROOT="$(pwd)"
MAIN_ROOT="${MAIN_ROOT:-${DEFAULT_MAIN_ROOT}}"

MASTER_DATA_PATH="${MAIN_ROOT}/data"
MODEL_SAVE_ROOT="${MAIN_ROOT}/models/recon"
MAIN_RESULTS_CSV="${MODEL_SAVE_ROOT}/main_experiment_metrics.csv"
ITERATIONS=400

# SPLITS=(
#     cone_ntrain_25_angle_360
#     cone_ntrain_50_angle_360
#     cone_ntrain_75_angle_360
# )

# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# cd "${REPO_ROOT}"

# run_case() {
#     local relative_name="$1"
#     local data_path="$2"
#     local steps="$3"
#     local model_path="${MODEL_SAVE_ROOT}/${relative_name}"

#     mkdir -p "${model_path}"
#     echo "Running train_recon.py for ${relative_name}"
#     python train_recon.py \
#         model.data_source_path="${data_path}" \
#         model.model_path="${model_path}" \
#         optim.steps="${steps}"
# }

# process_domain() {
#     local domain="$1"

#     for split in "${SPLITS[@]}"; do
#         local split_path="${MASTER_DATA_PATH}/${domain}/${split}"

#         if [[ ! -d "${split_path}" ]]; then
#             echo "Skipping missing path: ${split_path}" >&2
#             continue
#         fi

#         local views
#         if [[ "${split}" =~ cone_ntrain_([0-9]+)_angle_360 ]]; then
#             views="${BASH_REMATCH[1]}"
#         else
#             echo "Unable to parse view count from split: ${split}" >&2
#             continue
#         fi

#         local steps=$((ITERATIONS * views))

#         for dataset_dir in "${split_path}"/*; do
#             [[ -d "${dataset_dir}" ]] || continue
#             local dataset_name
#             dataset_name="$(basename "${dataset_dir}")"
#             local relative_name="${domain}/${split}/${dataset_name}"
#             run_case "${relative_name}" "${dataset_dir}" "${steps}"
#         done
#     done
# }

# process_domain "real_dataset"
# process_domain "synthetic_dataset"

echo "Collecting results"
python experiments/helpers/collect_main_experiment_results.py \
    --model-root "${MODEL_SAVE_ROOT}" \
    --output-csv "${MAIN_RESULTS_CSV}"
