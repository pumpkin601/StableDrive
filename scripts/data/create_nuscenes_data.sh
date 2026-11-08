#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd):${PYTHONPATH:-}"

# python tools/data_converter/nuscenes_converter.py nuscenes \
#     --root-path ./data/nuscenes \
#     --canbus ./data/nuscenes \
#     --out-dir ./data/infos/ \
#     --extra-tag nuscenes \
#     --version v1.0-mini

# python tools/data_converter/nuscenes_converter.py nuscenes \
#     --root-path ./data/nuscenes \
#     --canbus ./data/nuscenes \
#     --out-dir ./data/infos/ \
#     --extra-tag nuscenes \
#     --version v1.0

python tools/data_converter/nuscenes_converter_6s.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0
