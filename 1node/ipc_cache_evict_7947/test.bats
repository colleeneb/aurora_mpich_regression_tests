#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7947 -- the ch4/ipc GPU handle cache evicts an
# entry while a message using it is still outstanding.
#
# Rank 0 sends two messages from two different device buffers. With the cache limited to one
# entry, the second message evicts the first while the first is still in flight, and the
# receiver aborts in handle_to_fd(). Cache size 2 passes.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpifort -O2 -g -fiopenmp -fopenmp-targets=spir64 cache_evict_repro.f90 -o repro_evict
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7947 cache size 1" {
    MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=1 mpiexec -n 2 --ppn 2 gpu_tile_compact.sh ./repro_evict &> output || true
    cat output
    grep "RESULT: PASS" output
}

@test "mpi_issue7947 cache size 2" {
    MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=2 mpiexec -n 2 --ppn 2 gpu_tile_compact.sh ./repro_evict &> output || true
    cat output
    grep "RESULT: PASS" output
}
