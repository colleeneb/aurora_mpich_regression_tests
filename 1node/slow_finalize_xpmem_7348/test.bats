#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7348 -- MPI_Finalize is very slow when many
# shared-memory VMAs have to be torn down. Found in Thunder-SVM.
#
# Each broadcast uses a fresh 32 KB buffer, so the XPMEM mappings accumulate and finalize
# pays for all of them at once. The issue quotes 100000 iterations (~900,000 VMAs); 3000 is
# the reporter's own quick version and is what runs here.
#
# The program prints a "Finalizing ..." line, then a "Complete! (time: ~N secs). Goodbye."
# line whose N is the finalize time. It reports whole seconds, so the number has no decimal
# point; ~0 secs is what a healthy build gives.
#
# THRESHOLD: the issue gives no pass/fail line. 60 s of finalize is provisional.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicc -O2 -fiopenmp xpmem_stress.c -o xpmem_stress
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7348 finalize is not slow" {
    mpiexec -n 12 -ppn 12 --env FI_CXI_DEFAULT_CQ_SIZE=8192 ./xpmem_stress --bcast 32768 --iters 3000 &> output || true
    cat output
    fin=$(grep -oE "time: ~[0-9]+(\.[0-9]+)? secs" output | grep -oE "[0-9]+(\.[0-9]+)?" | tail -1)
    echo "finalize took ${fin} s"
    python3 -c "import sys; sys.exit(0 if ${fin} < 60.0 else 1)"
}
