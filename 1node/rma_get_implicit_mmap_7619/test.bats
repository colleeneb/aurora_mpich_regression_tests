#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7619 -- mpl/ze fast_memcpy crashes on the mmap in
# implicit scaling mode.
#
# A window over device memory, then one MPI_Get of a single element per rank. Implicit
# scaling (gpu_dev_compact.sh) fails with "mmap failed" and "Abort(15) ... internal_Get";
# explicit scaling (gpu_tile_compact.sh) passes. Both runs are kept so the pair shows which
# one regressed.
#
# The buffer size matters and is not monotonic -- the source lists sizes that work and sizes
# that crash. 101041 is the size the reporter left active.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpif90 -fc=ifx -O2 -g -i4 -r8 -what -fiopenmp -fopenmp-targets=spir64 test-small.F90 -o test_small
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7619 tile compact" {
    mpiexec -np 6 -ppn 6 -envall --no-vni --cpu-bind=list:1-16:17-32:33-48:53-68:69-84:85-100 gpu_tile_compact.sh ./test_small &> output || true
    cat output
    grep "All Done" output
}

@test "mpi_issue7619 dev compact" {
    mpiexec -np 6 -ppn 6 -envall --no-vni --cpu-bind=list:1-16:17-32:33-48:53-68:69-84:85-100 gpu_dev_compact.sh ./test_small &> output || true
    cat output
    grep "All Done" output
}
