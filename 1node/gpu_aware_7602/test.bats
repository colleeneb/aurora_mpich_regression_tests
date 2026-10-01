#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7602 -- mpl/ze asserts in
# MPL_gpu_query_is_same_dev on a plain GPU-aware exchange.
#
# Two ranks, Isend/Irecv on sycl::malloc_device buffers. The issue was only seen with
# mpich/dbg, not mpich/opt, so this is a smoke test that basic GPU-aware point-to-point
# still works rather than a reproducer of the assert itself.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fsycl t.cpp -o gpu_aware
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7602 gpu aware sendrecv" {
    mpiexec -n 2 gpu_tile_compact.sh ./gpu_aware
}
