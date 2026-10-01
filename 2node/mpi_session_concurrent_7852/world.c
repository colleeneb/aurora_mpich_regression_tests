/*
 * world.c -- a classic MPI "world model" program.
 *
 * Plain MPI_Init / MPI_COMM_WORLD / MPI_Finalize, exactly like a normal
 * application. On its own it always works.
 */
#include <mpi.h>
#include <stdio.h>

int main(void) {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("world: rank %d / %d\n", rank, size);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}

