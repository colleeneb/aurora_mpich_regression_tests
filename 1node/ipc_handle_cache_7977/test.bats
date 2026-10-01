#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7977 -- the ch4/ipc GPU handle cache mishandles
# allocations that do not fit in the cache, and requests that outlive a cache entry.
#
# rt_min: 2 ranks, 17 device allocations of 11 x 1536 bytes, one Isend and one Irecv per
# slice, one Waitall. With more allocations than MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE (16 by
# default), the uncached allocations share one GEM handle with no reference count, so the
# first completion closes it and the rest abort in handle_to_fd(). 16 allocations pass and
# 17 abort; the boundary follows the cvar exactly.
#
# rt_remesh: 20 allocations x 12 slices, buffers freed and reallocated between rounds the
# way Kokkos::realloc does. Freeing runs the cache's free hook, which compacts the entry
# array, and requests holding a pointer into it then decrement the wrong entry's in_use
# counter. Needs pidfd handles.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fsycl -O2 -g -std=c++17 rt_min.cpp -o rt_min
    mpicxx -fsycl -O2 -g -std=c++17 rt_remesh.cpp -o rt_remesh
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7977 cache overflow" {
    mpiexec -n 2 -ppn 2 gpu_tile_compact.sh ./rt_min
}

@test "mpi_issue7977 cache overflow small cache" {
    MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=1 mpiexec -n 2 -ppn 2 gpu_tile_compact.sh ./rt_min
}

@test "mpi_issue7977 remesh pidfd" {
    MPIR_CVAR_CH4_IPC_ZE_SHAREABLE_HANDLE=pidfd mpiexec -n 2 -ppn 2 gpu_tile_compact.sh ./rt_remesh
}
