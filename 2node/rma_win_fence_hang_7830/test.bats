#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7830 -- RMA broke with SHS 14.0 / libfabric 2.3.1.
#
# Two ranks on two nodes, a window over host memory, two fences. It hangs in the first fence,
# stuck in cxip_cntr_read inside libfabric, so the program prints "calling Win_fence (1st)"
# and never reaches "Win_fence done". MPIR_CVAR_CH4_OFI_ENABLE_RMA=0 is the workaround, so it
# must stay unset here.
#
# A hang, not a crash: the timeout is what fails the test.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicc -O2 -g rma_min.c -o rma_min
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7830 win_fence does not hang" {
    timeout 120 mpiexec -n 2 -ppn 1 ./rma_min &> output || true
    cat output
    grep "DONE" output
}
