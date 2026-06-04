#!/usr/bin/env bash
# quasar-review.sh — Deterministic helper for the quasar-review skill.
#
# Reviews a COMMIT RANGE or the FULL REPO of the Quasar tree (HIP/C++20 +
# pybind11 + Python). Unlike deep-review.sh (working tree vs a fixed base
# branch), this scopes by git commit range or by all tracked source files.
#
# Modes (mutually exclusive):
#   --scope                Cache the review scope + changed-file classification.
#   --lint                 Run ruff + static project gates; emit JSON.
#
# Target (mutually exclusive; default --commits HEAD~1..HEAD):
#   --commits <range>      Git range: HEAD~3..HEAD, <sha>..<sha>, or a single <sha>.
#   --full-repo            All tracked source under include/ src/ python/ apps/ tests/.
#
# Common flags:
#   --help, -h             Show this help.
#
# Exit codes:
#   0 success    1 bad args    2 missing dep    3 git failure

set -euo pipefail

CACHE_ROOT="${QUASAR_REVIEW_CACHE:-/tmp}"
SOURCE_DIRS=(include src python apps tests)
MODE=""
TARGET=""          # commits | full-repo
RANGE="HEAD~1..HEAD"

die()  { echo "quasar-review: $*" >&2; exit "${2:-1}"; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1" 2; }

set_mode()   { [[ -n "$MODE" ]]   && die "only one mode allowed"; MODE="$1"; }
set_target() { [[ -n "$TARGET" ]] && die "only one target allowed (--commits or --full-repo)"; TARGET="$1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)     set_mode scope; shift ;;
    --lint)      set_mode lint;  shift ;;
    --commits)   [[ $# -ge 2 ]] || die "missing value for --commits"; set_target commits; RANGE="$2"; shift 2 ;;
    --full-repo) set_target full-repo; shift ;;
    -h|--help)   sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           die "unknown flag: $1" ;;
  esac
done

[[ -z "$MODE" ]] && die "one of --scope / --lint is required"
[[ -z "$TARGET" ]] && TARGET="commits"
need git
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git work tree" 3

# Globals populated by build_scope (called directly, NOT in a subshell, so
# die/exit propagates to the real script).
DIR=""
SLUG=""

target_slug() {
  if [[ "$TARGET" == "full-repo" ]]; then
    echo "full-repo"
  else
    echo "commits-${RANGE//[^A-Za-z0-9]/-}"
  fi
}

# Build diff.patch (commit mode only) and files.txt. Sets DIR/SLUG globals.
# Call directly so die propagates.
build_scope() {
  SLUG="$(target_slug)"
  DIR="$CACHE_ROOT/quasar-review-$SLUG"
  mkdir -p "$DIR"

  if [[ "$TARGET" == "full-repo" ]]; then
    : > "$DIR/diff.patch"   # no diff in full-repo mode; reviewers read whole files
    git ls-files -- "${SOURCE_DIRS[@]}" | sort -u | sed '/^$/d' > "$DIR/files.txt" \
      || die "git ls-files failed" 3
  else
    git diff "$RANGE" > "$DIR/diff.patch" || die "git diff failed for range '$RANGE'" 3
    git diff --name-only "$RANGE" | sort -u | sed '/^$/d' > "$DIR/files.txt" \
      || die "git diff --name-only failed for range '$RANGE'" 3
  fi
}

# ---------------------------------------------------------------------------
# --scope
# ---------------------------------------------------------------------------
cmd_scope() {
  need jq
  build_scope
  local files; files="$(cat "$DIR/files.txt")"

  local t_include=0 t_src=0 t_backend_hip=0 t_physics=0 t_numerics=0 t_boundary=0
  local t_python=0 t_tests_cpp=0 t_tests_py=0 t_changelog=0 t_docs=0 t_examples=0
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    case "$f" in include/*)            t_include=1 ;; esac
    case "$f" in src/*)                t_src=1 ;; esac
    case "$f" in src/backend/hip/*)    t_backend_hip=1 ;; esac
    case "$f" in */physics/*)          t_physics=1 ;; esac
    case "$f" in */numerics/*)         t_numerics=1 ;; esac
    case "$f" in */boundary/*)         t_boundary=1 ;; esac
    case "$f" in python/quasar/*)      t_python=1 ;; esac
    case "$f" in tests/unit/*)         t_tests_cpp=1 ;; esac
    case "$f" in tests/python/*)       t_tests_py=1 ;; esac
    case "$f" in CHANGELOG.md)         t_changelog=1 ;; esac
    case "$f" in docs/*)               t_docs=1 ;; esac
    case "$f" in examples/*)           t_examples=1 ;; esac
  done <<<"$files"

  jq -n \
    --arg dir "$DIR" --arg target "$TARGET" --arg range "$RANGE" \
    --argjson include "$t_include" --argjson src "$t_src" \
    --argjson backend_hip "$t_backend_hip" --argjson physics "$t_physics" \
    --argjson numerics "$t_numerics" --argjson boundary "$t_boundary" \
    --argjson python "$t_python" --argjson tests_cpp "$t_tests_cpp" \
    --argjson tests_py "$t_tests_py" --argjson changelog "$t_changelog" \
    --argjson docs "$t_docs" --argjson examples "$t_examples" \
    '{cache_dir:$dir, target:$target, range:$range,
      touches_include:($include==1), touches_src:($src==1),
      touches_backend_hip:($backend_hip==1), touches_physics:($physics==1),
      touches_numerics:($numerics==1), touches_boundary:($boundary==1),
      touches_python:($python==1), touches_tests_cpp:($tests_cpp==1),
      touches_tests_py:($tests_py==1), touches_changelog:($changelog==1),
      touches_docs:($docs==1), touches_examples:($examples==1)}' \
    | tee "$DIR/classification.json"
}

# ---------------------------------------------------------------------------
# --lint
# ---------------------------------------------------------------------------
# Emits a JSON array. Each gate object:
#   { id, status (pass|warn|fail), message, anchor_file? }
cmd_lint() {
  need jq
  build_scope
  local files; files="$(cat "$DIR/files.txt")"

  local results='[]'
  emit() { results="$(jq --argjson e "$1" '. + [$e]' <<<"$results")"; }

  # Gate: ruff on Python under python/quasar/ and tests/python/.
  local py_files
  py_files="$(grep -E '\.py$' <<<"$files" | grep -E '^(python/quasar/|tests/python/)' || true)"
  if [[ -z "$py_files" ]]; then
    emit '{"id":"ruff","status":"pass","message":"No changed Python files under python/quasar/ or tests/python/."}'
  elif ! command -v ruff >/dev/null 2>&1; then
    emit '{"id":"ruff","status":"warn","message":"ruff not installed; skipped lint gate."}'
  else
    local ruff_out ruff_rc=0
    ruff_out="$(ruff check $py_files 2>&1)" || ruff_rc=$?
    if [[ "$ruff_rc" -eq 0 ]]; then
      emit '{"id":"ruff","status":"pass","message":"ruff check clean on changed Python files."}'
    else
      emit "$(jq -n --arg m "ruff reported issues:\n$ruff_out" '{id:"ruff",status:"fail",message:$m}')"
    fi
  fi

  local code_changed=0 changelog_changed=0 examples_changed=0 example_tests_changed=0
  grep -qE '^(include/|src/|python/quasar/)' <<<"$files" && code_changed=1
  grep -q  '^CHANGELOG.md'                   <<<"$files" && changelog_changed=1
  grep -q  '^examples/'                       <<<"$files" && examples_changed=1
  grep -q  '^tests/python/test_examples.py'  <<<"$files" && example_tests_changed=1

  # Gate: changelog presence for user-facing code changes (commit mode only;
  # in full-repo mode the whole tree is "changed" so the gate is not meaningful).
  if [[ "$TARGET" == "full-repo" ]]; then
    emit '{"id":"changelog","status":"pass","message":"Full-repo mode: changelog presence gate not applicable."}'
  elif [[ "$code_changed" -eq 1 && "$changelog_changed" -eq 0 ]]; then
    emit '{"id":"changelog","status":"warn","message":"include/src/python changes detected but no CHANGELOG.md entry. If user-observable, add an entry (see write-changelog skill)."}'
  else
    emit '{"id":"changelog","status":"pass","message":"CHANGELOG entry present or no code changes."}'
  fi

  # Gate: example/test sync — a new example needs a test_examples.py entry
  # (CLAUDE.md: each example has a matching integration test).
  if [[ "$TARGET" == "full-repo" ]]; then
    emit '{"id":"example_test_sync","status":"pass","message":"Full-repo mode: example/test sync gate not applicable."}'
  elif [[ "$examples_changed" -eq 1 && "$example_tests_changed" -eq 0 ]]; then
    emit '{"id":"example_test_sync","status":"warn","message":"examples/ changed but tests/python/test_examples.py was not. A new example needs a README and a matching test_examples.py entry.","anchor_file":"tests/python/test_examples.py"}'
  else
    emit '{"id":"example_test_sync","status":"pass","message":"No example added without a matching test entry."}'
  fi

  jq -c '.' <<<"$results"
}

case "$MODE" in
  scope) cmd_scope ;;
  lint)  cmd_lint ;;
esac
