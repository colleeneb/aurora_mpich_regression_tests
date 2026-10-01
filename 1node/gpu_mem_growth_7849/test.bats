#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7849 -- GPU memory grows without bound in a loop of
# allocate / Irecv+Send / Waitall / free.
#
# Reported growth: 0.079 GB at iteration 0, 0.568 GB at 25, 1.154 GB at 55 -- about 0.0195 GB
# per iteration at these sizes. The program prints used GPU memory every 5th iteration;
# ZES_ENABLE_SYSMAN=1 is required or the free-memory query returns nothing useful.
#
# ITERATION COUNT IS DELIBERATELY LOW. The issue runs 2000 iterations, which at the reported
# leak rate is ~39 GB -- enough to exhaust a 64 GB tile and take the node down. 60 iterations
# is past the point where the growth is unambiguous (1.15 GB by iteration 55) and peaks near
# 1.2 GB, which is harmless.
#
# THRESHOLD: the issue states no pass/fail line. 0.5 GB of growth between the first and last
# reading is provisional -- a leaking build passes it by iteration 25, a fixed build should
# stay near its starting value.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fsycl mem_growth.cpp -o mem_growth
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7849 gpu memory does not grow" {
    ZES_ENABLE_SYSMAN=1 mpiexec -n 2 -ppn 2 gpu_tile_compact.sh ./mem_growth 1310710 60 &> output || true
    cat output
    first=$(grep "Used GPU memory" output | head -1 | awk '{print $7}')
    last=$(grep "Used GPU memory" output | tail -1 | awk '{print $7}')
    echo "first=${first} GB  last=${last} GB"
    python3 -c "import sys; sys.exit(0 if ${last} - ${first} < 0.5 else 1)"
}
