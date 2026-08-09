#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16

set -euo pipefail

: "${QUASAR_BUILD_DIR:?set QUASAR_BUILD_DIR to a distributed Quasar build}"
QUASAR_PIC_DECK="${QUASAR_PIC_DECK:-examples/distributed/input_pic.yaml}"
QUASAR_CHECKPOINT_DIR="${QUASAR_CHECKPOINT_DIR:-${PWD}/distributed-checkpoints}"

mkdir -p "${QUASAR_CHECKPOINT_DIR}"
export PYTHONPATH="${QUASAR_BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

srun --cpu-bind=cores --gpu-bind=closest \
  python -m quasar.pic.cli run "${QUASAR_PIC_DECK}" \
  --devices auto \
  --decomposition 2x2 \
  --transport staged \
  --diagnostics-layout sharded \
  --checkpoint "${QUASAR_CHECKPOINT_DIR}/pic.h5" \
  --checkpoint-every 100
