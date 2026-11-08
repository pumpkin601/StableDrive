#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/final/T1_msts_training.py}"
GPUS="${2:-4}"

bash ./tools/dist_train.sh "${CONFIG}" "${GPUS}" --deterministic
