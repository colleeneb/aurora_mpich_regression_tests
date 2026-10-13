#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7880 -- AURORA MPI_THREAD_MULTIPLE random hangs:
# two threads per rank, each driving its own duplicated communicator over GPU device memory
# (sycl::malloc_device) with MPI_Alltoall, intermittently hang. Using host memory instead of
# malloc_device removes the hang. A maintainer linked PR #8008 as resolving the reproducer.
#
# The hang is random, so one invocation proves nothing -- the loop runs it 5 times under a
# timeout and any single hang (timeout) or missing verify=OK fails the test.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fsycl -lpthread -O2 thread_multi_alltoall.cpp -o thread_multi_alltoall
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7880 threaded alltoall over dup comms, 5 runs" {
    for i in $(seq 1 5); do
        timeout 300 mpiexec -np 24 --ppn 12 gpu_tile_compact.sh ./thread_multi_alltoall &> output || true
        cat output
        grep "verify=OK" output
    done
}
