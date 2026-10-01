#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7927 -- MPI_Put into a window backed by SYCL host
# USM asserts in MPL_gpu_local_to_global_dev_id.
#
# Single rank: sycl::malloc_host for both the origin and the window, MPI_Win_lock_all, then
# an MPI_Put to its own rank. Reported failing on Sunspot and working on Aurora, so it is
# here to catch the Aurora side regressing.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fsycl put_host_usm.cpp -o put_host_usm
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7927 put into host USM window" {
    mpiexec -n 1 ./put_host_usm
}
