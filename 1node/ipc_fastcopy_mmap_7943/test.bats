#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7943 -- ch4/ipc/gpu and misc/utils disagree about
# when to mmap, so a cached mapping is reused with the wrong kind of pointer.
#
# Rank 0 sends twice from the same base device buffer: 16384 bytes, then 1024. The first send
# caches a device pointer. The second is under MPIR_CVAR_GPU_FAST_COPY_MAX_SIZE (4096) so the
# receiver mmaps its destination, but the cached source is still a device pointer, and
# zeCommandListAppendMemoryCopy fails with ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY. Forcing both
# sends down one path with FAST_COPY_MAX_SIZE=0 passes.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpifort -O2 -g -fiopenmp -fopenmp-targets=spir64 oneway_repro.f90 -o repro_oneway
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7943 default" {
    mpiexec -n 2 --ppn 2 gpu_tile_compact.sh ./repro_oneway &> output || true
    cat output
    grep "RESULT: PASS" output
}

@test "mpi_issue7943 no fast copy" {
    MPIR_CVAR_GPU_FAST_COPY_MAX_SIZE=0 mpiexec -n 2 --ppn 2 gpu_tile_compact.sh ./repro_oneway &> output || true
    cat output
    grep "RESULT: PASS" output
}
