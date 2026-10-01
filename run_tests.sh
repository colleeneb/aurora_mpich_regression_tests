#!/usr/bin/env bash
#
# Run the Aurora reproducer tests in one directory.
#
#   ./run_tests.sh 1node
#   ./run_tests.sh 2node
#
# Needs mpiexec and bats on PATH. To get bats:
#   git clone --depth 1 -b v1.11.0 https://github.com/bats-core/bats-core.git
#   export PATH=$PWD/bats-core/bin:$PATH

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
directory=${1:-}

if [ ! -d "${here}/${directory}" ]; then
    echo "usage: $0 <directory>" >&2
    exit 2
fi

exec bats --timing --print-output-on-failure "${here}/${directory}"
