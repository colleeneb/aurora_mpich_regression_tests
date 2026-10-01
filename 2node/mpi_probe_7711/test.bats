#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7711 -- MPI_Probe crashes when messages arrive from
# a large number of ranks.
#
# 204 ranks over two nodes. Reported as "Fatal error in internal_Probe: Other MPI error" with
# ranks dying from signal 6, nondeterministically. FI_CXI_RX_MATCH_MODE=software is the
# workaround, so it must stay unset here.
#
# Also tracked as https://github.com/argonne-lcf/AuroraBugTracking/issues/112

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx t.cpp -o mpi_probe
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7711 probe from many ranks" {
    FI_LOG_LEVEL=warn FI_LOG_PROV=cxi mpiexec -n 204 -ppn 102 ./mpi_probe
}
