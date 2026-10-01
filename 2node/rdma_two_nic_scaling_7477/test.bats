#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7477 -- with HMEM and two ranks per tile, using two
# NICs is no faster than one.
#
# Four ranks over two nodes. gpu_dev_compact.sh gives 38.65 GB/s on two NICs against 21.27 on
# one and passes; gpu_tile_compact.sh gives 20.56 against 21.25 and fails. Both are kept.
#
# The program checks itself -- almost_equal(bw_2ppn, 2 * bw_1ppn, 0.20) and exit(1) -- so no
# threshold is invented here.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fiopenmp -fopenmp-targets=spir64 rdma_mpi_contension.cpp -o rdma_contention
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7477 dev compact" {
    NEOReadDebugKeys=1 EnableImplicitScaling=0 MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1 \
        mpiexec --cpu-bind=list:1-8:9-16 --mem-bind=list:2:2 -n 4 --ppn 2 gpu_dev_compact.sh ./rdma_contention
}

@test "mpi_issue7477 tile compact" {
    NEOReadDebugKeys=1 EnableImplicitScaling=0 MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1 \
        mpiexec --cpu-bind=list:1-8:9-16 --mem-bind=list:2:2 -n 4 --ppn 2 gpu_tile_compact.sh ./rdma_contention
}
