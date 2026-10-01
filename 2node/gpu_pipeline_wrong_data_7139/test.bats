#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7139 -- GPU pipelining returns wrong data when the
# pipeline buffer is smaller than the messages.
#
# 16 concurrent Isend/Irecv pairs of 128k-256k doubles between GPU buffers, repeated, with
# every received element compared against what was sent. With the pipeline buffer forced down
# to 256k it fails regularly on 2 nodes; the default size did not fail up to 8 nodes, so the
# small buffer is the point of the test, not an incidental setting.
#
# The program checks itself exactly (no tolerance), counts mismatches, Allreduces the count
# and prints " N errors." on failure or " done." on success.
#
# The same code later exposed a hang, split off as issue 7373, so a timeout is used here too.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fsycl -qopenmp sendrecvgpu.cc -o sendrecvgpu
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7139 pipelined sendrecv data is correct" {
    export MPIR_CVAR_CH4_OFI_ENABLE_GPU_PIPELINE=1
    export MPIR_CVAR_CH4_OFI_GPU_PIPELINE_BUFFER_SZ=$((256*1024))
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    timeout 900 mpiexec -np 24 --ppn 12 ./sendrecvgpu &> output || true
    tail -20 output
    ! grep "errors." output
}
