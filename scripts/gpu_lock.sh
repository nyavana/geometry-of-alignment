#!/usr/bin/env bash
# gpu_lock.sh — serialize GPU work across parallel agents on this box.
#
# The 4070 Ti Super 16 GB is a single-writer resource: activation extraction
# and abliteration both need to load Gemma 4 E4B at 8-bit simultaneously would
# blow the VRAM budget. This script acquires an exclusive flock on
# ~/.geometry-of-alignment/.gpu.lock before running the wrapped command, so
# that any agent running through it waits its turn instead of racing for VRAM.
#
# Usage:
#   scripts/gpu_lock.sh <command...>
#   scripts/gpu_lock.sh --wait 120 <command...>   # wait up to 120s for lock
#   scripts/gpu_lock.sh --max-hold 3600 <cmd...>  # kill after 1h (default 6h)
#
# Environment:
#   GPU_LOCK_FILE   override the lock file path
#   GPU_LOCK_WAIT   default --wait seconds (default 60)
#   GPU_LOCK_MAX    default --max-hold seconds (default 21600 = 6h)

set -euo pipefail

LOCK_FILE="${GPU_LOCK_FILE:-$HOME/.geometry-of-alignment/.gpu.lock}"
WAIT="${GPU_LOCK_WAIT:-60}"
MAX_HOLD="${GPU_LOCK_MAX:-21600}"

usage() {
    cat >&2 <<EOF
Usage: $0 [--wait SECONDS] [--max-hold SECONDS] <command...>

Serializes GPU work by taking an exclusive flock on:
    $LOCK_FILE

Defaults:
  --wait      $WAIT       (max seconds to block waiting for lock acquisition)
  --max-hold  $MAX_HOLD   (kill the wrapped command after this many seconds)
EOF
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait)
            [[ $# -ge 2 ]] || usage
            WAIT="$2"
            shift 2
            ;;
        --max-hold)
            [[ $# -ge 2 ]] || usage
            MAX_HOLD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    usage
fi

# Ensure the lock file exists (may have been deleted by cleanup).
mkdir -p "$(dirname "$LOCK_FILE")"
: > "$LOCK_FILE" 2>/dev/null || touch "$LOCK_FILE"

# Stamp the lock with acquirer info so humans can see who's holding it.
HOLDER="pid=$$ host=$(hostname) user=${USER:-unknown} started=$(date -Iseconds) cmd=$*"

# Acquire an exclusive lock with a bounded wait, then run the command under
# a watchdog timeout. timeout --foreground forwards signals so Ctrl-C works.
# The EXIT trap is only guaranteed to fire if we don't exec-replace this shell,
# so we run the wrapped command as a child and propagate its exit code.
exec flock --exclusive --wait "$WAIT" "$LOCK_FILE" bash -c '
    MAX_HOLD="$1"
    HOLDER="$2"
    LF="$3"
    shift 3
    printf "%s\n" "$HOLDER" > "${LF}.holder" 2>/dev/null || true
    cleanup() { rm -f "${LF}.holder" 2>/dev/null || true; }
    trap cleanup EXIT INT TERM
    timeout --foreground --kill-after=30 "$MAX_HOLD" "$@"
    rc=$?
    cleanup
    exit "$rc"
' gpu_lock_inner "$MAX_HOLD" "$HOLDER" "$LOCK_FILE" "$@"
