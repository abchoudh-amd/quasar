#!/usr/bin/env bash

# Apply CTest's per-test GPU allocation to the launched HIP process. CTest
# exposes allocated resource IDs through CTEST_RESOURCE_GROUP_* variables but
# deliberately does not alter device visibility itself. Tests use rank-local
# HIP ordinals, so masking the allocation here makes ordinal zero mean the
# first GPU assigned to this test and prevents parallel CTest jobs colliding.

set -u

if [[ "${1:-}" != "--" ]]; then
  echo "usage: $0 -- COMMAND [ARG ...]" >&2
  exit 2
fi
shift
if (( $# == 0 )); then
  echo "run_with_ctest_gpus: missing command" >&2
  exit 2
fi

group_count="${CTEST_RESOURCE_GROUP_COUNT:-}"
if [[ -z "${group_count}" || "${group_count}" == "0" ]]; then
  exec "$@"
fi
if [[ ! "${group_count}" =~ ^[0-9]+$ ]]; then
  echo "run_with_ctest_gpus: invalid CTEST_RESOURCE_GROUP_COUNT" >&2
  exit 77
fi

declare -a gpu_ids=()
declare -A seen_gpu_ids=()
for ((group = 0; group < group_count; ++group)); do
  group_types_name="CTEST_RESOURCE_GROUP_${group}"
  group_types="${!group_types_name-}"
  if [[ ",${group_types}," != *,gpus,* ]]; then
    continue
  fi

  allocation_name="CTEST_RESOURCE_GROUP_${group}_GPUS"
  allocation="${!allocation_name-}"
  if [[ -z "${allocation}" ]]; then
    echo "run_with_ctest_gpus: CTest GPU group ${group} has no allocation" >&2
    exit 77
  fi

  IFS=';' read -r -a resources <<< "${allocation}"
  for resource in "${resources[@]}"; do
    if [[ ! "${resource}" =~ (^|,)id:([^,;]+)(,|$) ]]; then
      echo "run_with_ctest_gpus: malformed GPU allocation '${resource}'" >&2
      exit 77
    fi
    gpu_id="${BASH_REMATCH[2]}"
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
      echo "run_with_ctest_gpus: GPU resource ID '${gpu_id}' is not an ordinal" >&2
      exit 77
    fi
    if [[ -n "${seen_gpu_ids[${gpu_id}]+present}" ]]; then
      echo "run_with_ctest_gpus: GPU resource ID '${gpu_id}' was allocated twice" >&2
      exit 77
    fi
    seen_gpu_ids["${gpu_id}"]=1
    gpu_ids+=("${gpu_id}")
  done
done

if (( ${#gpu_ids[@]} == 0 )); then
  echo "run_with_ctest_gpus: CTest supplied no GPU resources" >&2
  exit 77
fi

visible="$(IFS=,; echo "${gpu_ids[*]}")"

# HIP_VISIBLE_DEVICES is layered above any launcher-provided
# ROCR_VISIBLE_DEVICES mask. Do not overwrite the lower-level mask: scheduler
# allocations commonly use it to keep the process inside its job allocation.
export HIP_VISIBLE_DEVICES="${visible}"
exec "$@"
