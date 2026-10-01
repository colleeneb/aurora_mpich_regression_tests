#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *buf = (int*)aligned_alloc(64, 1024 * sizeof(int));
    for (int i = 0; i < 1024; ++i) buf[i] = rank;

    MPI_Win win;
    if (rank == 0) { printf("calling Win_create\n"); fflush(stdout); }
    MPI_Win_create(buf, 1024 * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    if (rank == 0) { printf("calling Win_fence (1st)\n"); fflush(stdout); }
    MPI_Win_fence(0, win);
    if (rank == 0) { printf("Win_fence done\n"); fflush(stdout); }
    MPI_Win_fence(0, win);
    if (rank == 0) { printf("Win_fence #2 done\n"); fflush(stdout); }
    MPI_Win_free(&win);
    free(buf);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("DONE\n");
    MPI_Finalize();
    return 0;
}
