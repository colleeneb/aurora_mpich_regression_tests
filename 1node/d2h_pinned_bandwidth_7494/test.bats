#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7494 -- MPI is slower than a plain OMP target
# memcpy for a device-to-pinned-host transfer. Reported 45.03 GB/s for MPI against 53.90 GB/s
# for omp_target_memcpy, intra-node.
#
# The program checks itself: it compares the two bandwidths with a 10% relative tolerance and
# exits nonzero on a mismatch, so no threshold is invented here.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fiopenmp -fopenmp-targets=spir64 d2h_pinned_bw.cpp -o d2h_pinned_bw
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7494 d2h pinned bandwidth" {
    mpiexec --cpu-bind=list:1-8:9-16 --mem-bind=list:2:2 -n 2 --ppn 2 gpu_tile_compact.sh ./d2h_pinned_bw
}
