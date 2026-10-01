#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7043 -- MPI_File_write_at_all from a device buffer
# hangs when GPU pipelining is on.
#
# Two ranks, one per node, each writing 200000 bytes from device memory. Size matters:
# 100000 does not hang, 200000 does. Node count matters too -- the issue states one node does
# not hang.
#
# A hang, not a crash: the timeout is what fails it.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fsycl t.cpp -o mpiio_pipeline
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
    rm -f test
}

@test "mpi_issue7043 write_at_all with pipelining" {
    MPIR_CVAR_CH4_OFI_ENABLE_GPU_PIPELINE=1 timeout 180 mpiexec -n 2 -ppn 1 ./mpiio_pipeline
}
