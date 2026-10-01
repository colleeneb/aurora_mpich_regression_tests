#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7461 -- MPI_THREAD_MULTIPLE with more than one VCI
# segfaults or hangs.
#
# Eight threads per rank, each with its own duplicated communicator, sending slices of one
# device allocation. VCIS=2 segfaults, VCIS=8 hangs, and the default (cvar unset) passes.
# Both failing values are kept as separate tests so the two symptoms stay distinguishable.
#
# Still intermittent as of 2026-09-24 on mpich/prd/5.0.0.aurora_test.51a9474, despite the
# issue being closed by PR #7462: two hangs and one clean pass on the same module.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fiopenmp -fopenmp-targets=spir64 threaded_gpu_rep.cpp -o threaded_gpu_rep
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7461 vcis 2" {
    MPIR_CVAR_CH4_NUM_VCIS=2 timeout 900 mpiexec -n 2 --ppn 1 gpu_tile_compact.sh ./threaded_gpu_rep &> output || true
    cat output
    grep "Finished" output
}

@test "mpi_issue7461 vcis 8" {
    MPIR_CVAR_CH4_NUM_VCIS=8 timeout 900 mpiexec -n 2 --ppn 1 gpu_tile_compact.sh ./threaded_gpu_rep &> output || true
    cat output
    grep "Finished" output
}
