#!/usr/bin/env bats
#
# https://github.com/pmodels/mpich/issues/7852 -- an MPI Sessions process fails when it runs
# alongside a classic MPI_COMM_WORLD process that shares its PMIx identity.
#
# Each rank forks a "session" helper and then runs "world", so both inherit the same
# PMIX_NAMESPACE. The failure is nondeterministic: either a hang (exit 124 under the driver's
# own timeout) or "Fatal error in internal_Comm_create_from_group: Message truncated".
# Building with -DNS_FIX gives the session a distinct PMIX_NAMESPACE and works, so the driver
# runs both and prints each exit code.
#
# full_reproducer.sh nests its own mpirun, which is how the reproducer is written. Defaults
# are 2 ranks / 4 ppn from the script itself.

setup_file() {
    cd "${BATS_TEST_DIRNAME}"
    mpicc -O2 -o session session.c
    mpicc -O2 -o session_fix -DNS_FIX session.c
    mpicc -O2 -o world world.c
}

setup() {
    cd "${BATS_TEST_DIRNAME}"
}

@test "mpi_issue7852 session alongside world" {
    bash full_reproducer.sh &> output || true
    cat output
    # the plain session must exit 0, same as the -DNS_FIX build
    ! grep -E "^session: exit code [^0]" output
}
