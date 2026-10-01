#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7752 -- MPI_Allreduce on a 1 GiB GPU buffer is far
# slower than the same reduction through oneCCL. Reported 29.70 GB/s for MPI against 1467.62
# GB/s for oneCCL, a ratio of about 0.02.
#
# The program runs both and prints "Time ratio oneCCL / MPI". It also verifies the result
# values itself and throws on a mismatch.
#
# The two do the same reduction on the same hardware, so the target is parity: a ratio of 1.0.
# The issue reports 0.0202, and 0.0193 was measured on 2026-09-24 with
# mpich/prd/5.0.0.aurora_test.51a9474.
#
# THRESHOLD: the issue states no pass/fail line -- 0.5 is the line taken here, meaning MPI
# within 2x of oneCCL. It is a judgement call, not something the issue asked for.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -lccl -fsycl allreduce_vs_ccl.cpp -o allreduce_vs_ccl
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7752 allreduce within 2x of oneCCL" {
    ZE_FLAT_DEVICE_HIERARCHY=flat mpiexec -n 12 -ppn 12 ./ccl_rank.sh ./allreduce_vs_ccl &> output || true
    cat output
    ratio=$(grep "Time ratio" output | grep -oE "[0-9.e-]+$")
    echo "oneCCL/MPI ratio = ${ratio} (1.0 is parity)"
    python3 -c "import sys; sys.exit(0 if ${ratio} > 0.5 else 1)"
}
