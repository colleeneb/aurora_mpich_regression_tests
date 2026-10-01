#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7467 -- sending from a host buffer into a GPU
# buffer crashes when MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1.
#
# Rank 0 sends 2^28 ints from host memory, rank 1 receives into an omp_target_alloc_device
# pointer. Without the cvar it passes; with it rank 1 dies in internal_Recv. Both ranks print
# "Starting MPI Point 2 Point" but "Finished" never appears on failure.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fiopenmp -fopenmp-targets=spir64 gpu_rmd_rep.cpp -o gpu_rmd_rep
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7467 hmem enabled" {
    MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1 mpiexec -n 2 -ppn 1 gpu_tile_compact.sh ./gpu_rmd_rep &> output || true
    cat output
    grep "Finished" output
}

@test "mpi_issue7467 hmem default" {
    mpiexec -n 2 -ppn 1 gpu_tile_compact.sh ./gpu_rmd_rep &> output || true
    cat output
    grep "Finished" output
}
