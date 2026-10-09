#!/usr/bin/env bash
set -euo pipefail

suite=${1:?usage: run-suite.sh <1node|2node>}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$here/ci/mpich-env.sh"

export BATS_REPORT_FILENAME="${CI_JOB_NAME:-$suite}.xml"
rm -rf "$here/reports"
mkdir -p "$here/reports"
rc=0
bats --timing --print-output-on-failure -r --report-formatter junit --output "$here/reports" "$here/$suite" || rc=$?
test -s "$here/reports/$BATS_REPORT_FILENAME"
exit "$rc"
