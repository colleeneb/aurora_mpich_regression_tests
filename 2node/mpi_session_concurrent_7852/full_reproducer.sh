#!/bin/bash
#
# full_reproducer.sh -- MPICH deadlock/abort when an MPI Sessions process runs
# concurrently with a classic-world MPI process that shares its PMIx identity.
#
# Per rank, one launcher forks a `session` helper (background) and a `world`
# app (foreground), so the two MPI instances inherit the same PMIx identity
# (PMIX_NAMESPACE / PMIX_RANK). It runs the scenario with the plain `session`
# (reproduces the bug) and with `session_fix` (-DNS_FIX, distinct
# PMIX_NAMESPACE -> works), and prints the exit code of each. A per-run
# `timeout` turns a hang into a non-zero code.
#
# Usage:  ./full_reproducer.sh [NRANKS] [PPN]      (default 128 64, needs >=2 nodes)

set -u

# We will call ourself as some pont. Guard it with --worked.
# This will run first session than hello_world
# We send kill to the session to unpause and exiting nicely
if [ "${1:-}" = "--worker" ]; then
    here="$2"; session_bin="$3"
    "$here/$session_bin" &
    helper=$!
    "$here/world"
    rc=$?
    kill "$helper" 2>/dev/null
    exit $rc
fi

NRANKS="${1:-2}"
PPN="${2:-4}"
TIMEOUT="${TIMEOUT:-20}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SELF="$HERE/$(basename "$0")"

mpicc -O2 -o "$HERE/world"       "$HERE/world.c"            || exit 2
mpicc -O2 -o "$HERE/session"     "$HERE/session.c"          || exit 2
mpicc -O2 -o "$HERE/session_fix" -DNS_FIX "$HERE/session.c" || exit 2

run_case() {
    local session_bin="$1"
    # Sweep stragglers from a previous hung run (one pkill per node).
    mpirun -n "$NRANKS" -- pkill -9 -x 'world|session|session_fix' >/dev/null 2>&1

    # Run ourself
    echo "$session_bin: Running"
    timeout "$TIMEOUT" mpirun -n "$NRANKS" --ppn "$PPN" -- \
        "$SELF" --worker "$HERE" "$session_bin"
    echo "$session_bin: exit code $?"
}

run_case session
run_case session_fix
