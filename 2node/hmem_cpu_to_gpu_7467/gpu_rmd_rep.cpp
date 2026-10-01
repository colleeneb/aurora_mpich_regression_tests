#include <cassert>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <vector>

int main(int argc, char **argv) {

   MPI_Init(NULL,NULL);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int N = (1 << 28);
  const int N_byte = N * sizeof(int);
  int device_id = omp_get_default_device();

  std::vector<int> A(N);
  int *A2_gpu = (int *)omp_target_alloc_device(N_byte, device_id);
  std::cout << "Starting MPI Point 2 Point" << std::endl;
    if (world_rank % world_size == 0) {
      MPI_Send(
          /* data         = */ A.data(),
          /* count        = */ N,
          /* datatype     = */ MPI_INT,
          /* destination  = */ world_rank + 1,
          /* tag          = */ 0,
          /* communicator = */ MPI_COMM_WORLD);
    } else {
      MPI_Recv(
          /* data         = */ A2_gpu,
          /* count        = */ N,
          /* datatype     = */ MPI_INT,
          /* source       = */ world_rank - 1,
          /* tag          = */ 0,
          /* communicator = */ MPI_COMM_WORLD,
          /* status       = */ MPI_STATUS_IGNORE);
    }

  std::cout << "Finished " << std::endl;
}
