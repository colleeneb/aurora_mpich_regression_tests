#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7724 -- yaksa leaks ze handles at init.
#
# MPI_Init plus MPI_Finalize under AddressSanitizer. LeakSanitizer reports 96 bytes in 6
# objects from yaksuri_ze_init_hook via yaksa_init. -fsycl is needed because libigc's
# DEEPBIND does not work under ASan; FI_PROVIDER=tcp keeps cxi out of the report.
#
# LeakSanitizer exits nonzero whenever it reports anything, including leaks from libfabric
# and PMIx that are not what this test is about, so the run's own exit code is not the
# signal -- the yaksa frame in the report is.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fsycl -O3 -g -fsanitize=address init_finalize.cpp -o init_finalize
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7724 no ze leak at init" {
    FI_PROVIDER=tcp mpiexec -n 1 -ppn 1 ./init_finalize &> output || true
    cat output
    ! grep "yaksuri_ze_init_hook" output
}
