#include <mpi.h>

int main(void) {
    MPI_Init(NULL, NULL);
    MPI_Finalize();
}
