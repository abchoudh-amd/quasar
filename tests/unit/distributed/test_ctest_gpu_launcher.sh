#!/usr/bin/env bash

set -euo pipefail

launcher="${1:?usage: test_ctest_gpu_launcher.sh LAUNCHER}"

env \
  -u HIP_VISIBLE_DEVICES \
  CTEST_RESOURCE_GROUP_COUNT=2 \
  CTEST_RESOURCE_GROUP_0=gpus \
  CTEST_RESOURCE_GROUP_0_GPUS='id:4,slots:1' \
  CTEST_RESOURCE_GROUP_1=gpus \
  CTEST_RESOURCE_GROUP_1_GPUS='id:7,slots:1' \
  "${launcher}" -- \
  bash -c 'test "${HIP_VISIBLE_DEVICES}" = "4,7"'

# With resource allocation disabled, preserve the caller's visibility exactly.
env \
  -u CTEST_RESOURCE_GROUP_COUNT \
  HIP_VISIBLE_DEVICES=3 \
  "${launcher}" -- \
  bash -c 'test "${HIP_VISIBLE_DEVICES}" = "3"'

set +e
error="$(env \
  -u CTEST_RESOURCE_GROUP_0_GPUS \
  CTEST_RESOURCE_GROUP_COUNT=1 \
  CTEST_RESOURCE_GROUP_0=gpus \
  "${launcher}" -- true 2>&1)"
status=$?
set -e
test "${status}" -eq 77
test "${error}" = \
  "run_with_ctest_gpus: CTest GPU group 0 has no allocation"
