#!/bin/bash

set -euo pipefail
shopt -s nullglob

DEFAULT_MAIN_ROOT="$(pwd)"
MAIN_ROOT="${MAIN_ROOT:-${DEFAULT_MAIN_ROOT}}"

WARMSTART_DATA_ROOT="${MAIN_ROOT}/data/init_dataset"
INIT_RECON_SAVE_ROOT="${MAIN_ROOT}/models/init_test/recon"
INIT_VOL_SAVE_ROOT="${MAIN_ROOT}/models/init_test/vol_prior"
INIT_RESULTS_CSV="${MAIN_ROOT}/models/init_test/init_comparison_metrics.csv"
RECON_STEPS="10"
VOLUME_STEPS="50"

DATASETS=(
    "crick_1_cone:cricket"
    "brain_cone:brain"
    "pancreas1_cone:pancreas"
    "walnut_1_cone:walnut"
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

run_recon() {
    local dataset_name="$1"
    local data_path="$2"
    local init_mode="$3"
    local model_path="${INIT_RECON_SAVE_ROOT}/${dataset_name}"

    mkdir -p "${model_path}"
    echo "Running train_recon.py (${init_mode}) for ${dataset_name}"
    python train_recon.py \
        optim.steps="${RECON_STEPS}" \
        model.data_source_path="${data_path}" \
        model.model_path="${model_path}" \
        model.init_mode="${init_mode}"
}

run_volume() {
    local dataset_name="$1"
    local data_path="$2"
    local model_path="${INIT_VOL_SAVE_ROOT}/${dataset_name}"

    mkdir -p "${model_path}"
    echo "Running train_volume.py for ${dataset_name}"
    python train_volume.py \
        optim.steps="${VOLUME_STEPS}" \
        model.data_source_path="${data_path}" \
        model.model_path="${model_path}"
}

for dataset_spec in "${DATASETS[@]}"; do
    IFS=":" read -r dataset_folder dataset_name <<< "${dataset_spec}"
    data_path="${WARMSTART_DATA_ROOT}/${dataset_folder}"

    if [[ ! -d "${data_path}" ]]; then
        echo "Skipping missing dataset: ${data_path}" >&2
        continue
    fi

    run_recon "${dataset_name}" "${data_path}" "gradient"
    run_recon "${dataset_name}" "${data_path}" "intensity"
    run_volume "${dataset_name}" "${data_path}"
done

echo "Collecting summary results"
python experiments/helpers/collect_init_results.py \
    --warmstart-data-root "${WARMSTART_DATA_ROOT}" \
    --recon-save-root "${INIT_RECON_SAVE_ROOT}" \
    --vol-save-root "${INIT_VOL_SAVE_ROOT}" \
    --output-csv "${INIT_RESULTS_CSV}"
