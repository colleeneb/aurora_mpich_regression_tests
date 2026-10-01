#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7944 -- ipc_hdr is read without checking ipc_type,
# so global_dev_id is used uninitialised and MPL_gpu_query_is_same_dev asserts.
#
# A 3x2x2 halo exchange over 12 ranks. The failure is intermittent, about 15% of runs, so one
# invocation proves nothing -- the loop runs it 20 times and any single abort fails the test.
#
# MPIR_CVAR_GPU_FAST_COPY_MAX_SIZE=0 keeps issue 7943 out of the way, as the issue instructs.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpifort -O2 -g -fiopenmp -fopenmp-targets=spir64 min_halo.f90 -o min_halo
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7944 halo exchange, 20 runs" {
    export MPIR_CVAR_GPU_FAST_COPY_MAX_SIZE=0
    for i in $(seq 1 20); do
        mpiexec -n 12 --ppn 12 --cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100 gpu_tile_compact.sh ./min_halo 25 &> output || true
        grep "RESULT: PASS" output
    done
}
