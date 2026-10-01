#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7884 -- MPI_Get with RMA disabled segfaults in
# MPIDIG_get_ack_target_msg_cb once there is enough concurrent traffic.
#
# 100000 MPI_Gets across 200 fences. msg_hdr->greq_ptr decodes to something that is not a
# heap pointer, and the assert on rreq->kind faults. The program prints "All Done" on success.
#
# The issue says this is only reachable once issue 7881 is fixed, and as of 2026-09-24 it
# still aborts on 7881's assert (packed_size == seg_sz) before reaching its own path.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicc -O0 -g repro_bug3_min.c -o repro_bug3_min
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7884 many gets with rma disabled" {
    MPIR_CVAR_CH4_OFI_ENABLE_RMA=0 timeout 600 mpiexec -n 2 -ppn 1 ./repro_bug3_min &> output || true
    cat output
    grep "All Done" output
}
