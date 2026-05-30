#!/bin/bash
#SBATCH --job-name=square_toroid_pic_bore_1us
#SBATCH --gres=gpu:gfx942-mi300x:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1T
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "host    : $(hostname)"
echo "job id  : ${SLURM_JOB_ID}"
echo "started : $(date -Is)"

export PYTHONPATH=build/hip-gfx942-release/python

python3 -m quasar.pic.cli run examples/square_toroid_pic/input.yaml \
  --print-config \
  --log-every 100000 \
  --write-every 500000

echo "finished: $(date -Is)"
