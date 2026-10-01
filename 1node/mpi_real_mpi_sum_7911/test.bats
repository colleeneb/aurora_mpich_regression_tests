#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7911 -- MPI_Reduce with MPI_REAL and MPI_SUM
# returns a wrong answer in Fortran. Reported as Infinity where 3 was expected; MPI_REAL4
# gives the right result.
#
# The program checks its own answer and stops nonzero.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpif90 -O2 reduce_repro.f90 -o reduce_repro
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7911 reduce with MPI_REAL" {
    mpiexec -n 2 ./reduce_repro
}
