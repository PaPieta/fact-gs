#!/bin/bash

set -euo pipefail
shopt -s nullglob

DEFAULT_MAIN_ROOT="$(pwd)"
MAIN_ROOT="${MAIN_ROOT:-${DEFAULT_MAIN_ROOT}}"

DATA_ROOT="${MAIN_ROOT}/data/hyperres_dataset"
VOL_PRIOR_ROOT="${MAIN_ROOT}/models/hyperres/vol_prior"
RECON_GRAD_ROOT="${MAIN_ROOT}/models/hyperres/recon"
RECON_PRIOR_ROOT="${MAIN_ROOT}/models/hyperres/recon_prior"
WARM_START_RESULTS_CSV="${MAIN_ROOT}/models/hyperres/warm_start_metrics.csv"
VOLUME_STEPS=50
RECON_STEPS=20000

CASES=(
    chest_CT_01_1mm
    chest_CT_02_1mm
    chest_CT_03_1mm
    chest_CT_04_1mm
    chest_CT_05_1mm
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

run_case() {
    local case_name="$1"
    local data_path="${DATA_ROOT}/${case_name}_cone"
    local vol_prior_model="${VOL_PRIOR_ROOT}/${case_name}"
    local recon_grad_model="${RECON_GRAD_ROOT}/${case_name}"
    local recon_prior_model="${RECON_PRIOR_ROOT}/${case_name}"
    local prior_point_cloud="${vol_prior_model}/point_cloud/step_50/point_cloud.pickle"

    echo "=== Warm-start case: ${case_name} ==="

    python train_volume.py optim.steps="${VOLUME_STEPS}" \
        model.data_source_path="${data_path}" \
        model.model_path="${vol_prior_model}" \
        model.num_gaussians=100000

    python train_recon.py optim.steps="${RECON_STEPS}" \
        model.data_source_path="${data_path}" \
        model.model_path="${recon_grad_model}" \
        model.init_mode=gradient \
        model.num_gaussians=100000 \
        eval=eval_100_silent

    python train_recon.py optim.steps="${RECON_STEPS}" \
        model.data_source_path="${data_path}" \
        model.model_path="${recon_prior_model}" \
        model.init_mode=prior \
        model.prior_path="${prior_point_cloud}" \
        model.num_gaussians=100000 \
        eval=eval_100_silent
}

for case in "${CASES[@]}"; do
    run_case "${case}"
done

case_list="$(IFS=,; echo "${CASES[*]}")"

echo "Collecting results"
python experiments/helpers/collect_warm_start_result.py \
    --recon-grad-root "${RECON_GRAD_ROOT}" \
    --recon-prior-root "${RECON_PRIOR_ROOT}" \
    --total-steps "${RECON_STEPS}" \
    --cases "${case_list}" \
    --output-csv "${WARM_START_RESULTS_CSV}"
