#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7600 -- the optimized build prints
# "[WARNING] yaksa: 1 leaked handle pool objects", which should only appear in mpich/dbg.
#
# The program commits an MPI datatype and deliberately does not free it. The leak is the
# point; the warning reaching a non-debug build is the bug. The -DFREE_COMMIT build frees the
# datatype and must also be quiet.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx m.cpp -o leaking_pool
    mpicxx -DFREE_COMMIT m.cpp -o leaking_pool_free
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7600 no leak warning" {
    mpiexec -n 1 ./leaking_pool &> output || true
    cat output
    ! grep "leaked" output
}

@test "mpi_issue7600 no leak warning when freed" {
    mpiexec -n 1 ./leaking_pool_free &> output || true
    cat output
    ! grep "leaked" output
}
