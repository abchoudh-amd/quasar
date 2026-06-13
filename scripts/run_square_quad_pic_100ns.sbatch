#!/bin/bash
#SBATCH --job-name=square_quad_pic_100ns
#SBATCH --partition=defq
#SBATCH --constraint=MARKHAM
#SBATCH --gres=gpu:gfx942-mi300x:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1T
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log

# --constraint=MARKHAM: this cluster splits /home by site -- the login node and
# repo live on the Markham (mkm) fileserver, while AUSTIN-tagged nodes mount a
# different /home where this repo does not exist (an srun/sbatch there fails with
# "couldn't chdir ... No such file or directory"). MARKHAM pins the job to nodes
# that can see the checkout + build tree.

# 100 ns EM-PIC run of examples/square_quad_pic (H+/mu- on the central null of a
# 30 cm square-frame magnetic quadrupole), full resolution: 128^2 grid, 100 ppc,
# two species (~3.28M particles).
#
# The EM solver is explicit, so dt is fixed by the light-crossing CFL
# (dt ~ 2.303e-14 s at dx ~ 19.5 um), NOT particle dynamics. 100 ns of physical
# time is therefore 4,341,459 steps. There is NO checkpoint/resume in the CLI
# (the per-particle 'alive' confinement-loss flag cannot be restored), so this is
# a single run: --write-every flushes a rolling checkpoint and the LAST completed
# flush is the recoverable artifact if Slurm SIGTERMs the job at the 12h cap
# before the full 100 ns is reached. Mirrors scripts/run_square_toroid_pic_1m_*.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "host    : $(hostname)"
echo "job id  : ${SLURM_JOB_ID}"
echo "started : $(date -Is)"

# This run uses the PREBUILT extension at build/hip-gfx942-release (already
# compiled for gfx942). We deliberately do NOT rebuild in-job: the repo's cmake
# preset fails to configure on some MARKHAM nodes whose ROCm layout omits the
# `hip-lang` cmake package (CMakeHIPInformation can't find hip-langConfig.cmake),
# and the prebuilt _core.so imports and runs fine on these nodes (verified). If
# the C++ sources change, rebuild on the login node before submitting.
export PYTHONPATH=build/hip-gfx942-release/python

# 1) Fail fast if HIP can't actually open a GPU on this node. Probe with a real
#    hipGetDeviceCount() -- NOT rocminfo/rocm-smi (rocminfo fails under Slurm on
#    this cluster on the AD-user render-group check even where HIP works, and
#    rocm-smi reads sysfs so it can pass on a node whose KFD ioctl is broken).
cat > "${TMPDIR:-/tmp}/hipprobe_${SLURM_JOB_ID}.cpp" <<'PROBE'
#include <hip/hip_runtime.h>
#include <cstdio>
int main(){int n=0; hipError_t e=hipGetDeviceCount(&n);
  printf("hipGetDeviceCount err=%d(%s) count=%d\n",(int)e,hipGetErrorString(e),n);
  return n>0?0:2;}
PROBE
hipcc "${TMPDIR:-/tmp}/hipprobe_${SLURM_JOB_ID}.cpp" -o "${TMPDIR:-/tmp}/hipprobe_${SLURM_JOB_ID}"
if ! "${TMPDIR:-/tmp}/hipprobe_${SLURM_JOB_ID}"; then
  echo "ERROR: HIP sees no GPU on $(hostname); resubmit to land elsewhere." >&2
  exit 1
fi

# 2) Smoke + gate: 50 steps must complete cleanly (prints resolved dt/steps and a
#    real steps/s for this node). At ~300 step/s the full run is ~4 h, inside cap.
python3 -m quasar.pic.cli run examples/square_quad_pic/input.yaml \
  --print-config --steps-override 50

# 3) The 100 ns run. 4,341,459 steps = 100.0 ns at the deck's auto (CFL) dt.
#    --log-every prints progress + ETA; --write-every flushes a rolling
#    out_<step>.npz checkpoint (~every 250k steps) so a cap-truncated run still
#    leaves a recoverable furthest-reached state.
python3 -m quasar.pic.cli run examples/square_quad_pic/input.yaml \
  --print-config \
  --steps-override 4341459 \
  --log-every 50000 \
  --write-every 250000

echo "finished: $(date -Is)"
