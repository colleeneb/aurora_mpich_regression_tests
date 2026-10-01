#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7881 -- one large MPI_Get with RMA disabled trips
# "packed_size == seg_sz" in ofi_am_impl.h.
#
# Rank 0 gets rank 1's whole 32 MiB buffer of MPI_C_DOUBLE_COMPLEX in a single call. The
# instrumented values were seg_sz=16344 against packed=16336 -- the segment size is not a
# multiple of the element size, and yaksa's pack rounds down. "rank 0: MPI_Get posted" prints
# but the fence-completion line never does.
#
# N_DOUBLES can be set to sweep the buffer size.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicc -O0 -g repro_bug2.c -o repro_bug2
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7881 one large get with rma disabled" {
    MPIR_CVAR_CH4_OFI_ENABLE_RMA=0 timeout 300 mpiexec -n 2 -ppn 1 ./repro_bug2 &> output || true
    cat output
    grep "All Done" output
}
