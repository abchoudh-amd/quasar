#!/bin/bash
#SBATCH --job-name=stp1m_gfx942_4096
#SBATCH --partition=defq
#SBATCH --gres=gpu:gfx942-mi300x:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1T
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log

# Full-resolution (4096^2) PIC run of examples/square_toroid_pic_1m on an MI300X
# (gfx942) node -- the MI350/gfx950 queue had no schedulable node, so this targets
# the readily-available gfx942 fleet instead. Builds gfx942 in-job, gates on the
# PIC unit tests + a 50-step smoke, then runs. The deck targets the full 5 us
# (~64.3M steps); the 12h cap ends it partway. There is NO resume (the CLI can't
# restore the per-particle 'alive' flag, which is this sim's confinement-loss
# result), so this is a single partial run whose deliverable is the loss curve up
# to whatever sim-time 12h reaches. Frequent --write-every flushes mean the last
# out_4096_full.npz holds the furthest-reached state when Slurm kills it at the cap.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "host    : $(hostname)"
echo "job id  : ${SLURM_JOB_ID}"
echo "started : $(date -Is)"

# cmake is not on PATH here; the repo presets need cmake >= 3.27 (older rejects the
# preset 'version' field). This venv ships 4.3.2; ctest lives beside it.
CMAKE=/home/AMD/abchoudh/.venvs/rocprofiler-compute-asan_ci_test_integration/bin/cmake
CTEST=/home/AMD/abchoudh/.venvs/rocprofiler-compute-asan_ci_test_integration/bin/ctest

# 1) Fail fast if HIP can't actually open a GPU on this node. We probe with a real
#    hipGetDeviceCount() call -- NOT rocminfo/rocm-smi: rocminfo fails on this
#    cluster inside Slurm with "Failed to get user name to check for render group
#    membership" (AD-user lookup) even on nodes where HIP works fine, and rocm-smi
#    reads sysfs and can pass on a node whose KFD ioctl is broken (e.g.
#    ctr-cx64-mi300x-4). hipGetDeviceCount is exactly what the solver needs.
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

# 2) Configure + build gfx942 (self-contained: rebuilds on the node).
"$CMAKE" --preset hip-gfx942-release
"$CMAKE" --build  --preset hip-gfx942-release -j

# 3) Gate: the PIC unit tests must pass before the long run commits; a 50-step
#    smoke prints the resolved dt/steps + a real steps/s for this node.
"$CTEST" --preset hip-gfx942-release -R '^pic_' --output-on-failure

export PYTHONPATH=build/hip-gfx942-release/python
python3 -m quasar.pic.cli run examples/square_toroid_pic_1m/input_4096.yaml \
  --print-config --steps-override 50

# 4) The run. The deck's `steps` covers the full 5 us; Slurm SIGTERMs it at the
#    12h cap. --write-every flushes a rolling checkpoint (~every 1M steps) to
#    examples/square_toroid_pic_1m/out_4096_full.npz; the last completed flush is
#    the recoverable artifact.
python3 -m quasar.pic.cli run examples/square_toroid_pic_1m/input_4096.yaml \
  --print-config --log-every 200000 --write-every 1000000

echo "finished: $(date -Is)"
