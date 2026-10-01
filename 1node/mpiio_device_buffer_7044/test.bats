#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7044 -- MPI_File_write_at with a GPU device buffer
# writes nothing. The same code with a host buffer works.
#
# One rank writes 4 bytes from device memory and then checks the file size. Reported as
# "The size of test is 0 bytes." where 4 was expected. The program checks itself and returns
# 1, so there is no threshold here.
#
# The file is opened MPI_MODE_EXCL, so a leftover "test" from a previous run makes the open
# fail.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicxx -fsycl t.cpp -o mpiio_device_buffer
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
    rm -f test
}

@test "mpi_issue7044 write_at from device buffer" {
    mpiexec -n 1 ./mpiio_device_buffer
}
