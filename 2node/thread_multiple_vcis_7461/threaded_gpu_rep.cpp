#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <cassert>

int main(int argc, char **argv) {
  // Initialize the MPI environment
  {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      printf("The threading support level is lesser than that demanded.\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int N = (1 << 28);
  const int N_byte = N * sizeof(int);
  int device_id = omp_get_default_device();

  int n_chunk = 8;
  assert(N % n_chunk == 0);
  int chunk_size = N / n_chunk;

  int *A2_gpu = (int *)omp_target_alloc_device(N_byte, device_id);
  std::vector<int *> A2c_gpu;
  std::vector<MPI_Comm> Acc;
  for (int i = 0; i < n_chunk; ++i) {
    A2c_gpu.push_back(A2_gpu + i * chunk_size);
    MPI_Comm duplicated_communicator;
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_communicator);
    Acc.push_back(duplicated_communicator);
  }

  std::cout << "Starting MPI Point 2 Point" << std::endl;
#pragma omp parallel for
  for (int i = 0; i < n_chunk; i++) {
    if (world_rank < world_size / 2) {
      MPI_Send(
          /* data         = */ A2c_gpu[i],
          /* count        = */ chunk_size,
          /* datatype     = */ MPI_INT,
          /* destination  = */ world_rank + world_size / 2,
          /* tag          = */ i,
          /* communicator = */ Acc[i]);
    } else {
      MPI_Recv(
          /* data         = */ A2c_gpu[i],
          /* count        = */ chunk_size,
          /* datatype     = */ MPI_INT,
          /* source       = */ world_rank - world_size / 2,
          /* tag          = */ i,
          /* communicator = */ Acc[i],
          /* status       = */ MPI_STATUS_IGNORE);
    }
  }

  std::cout << "Finished " << std::endl;
  for (int i = 0; i < n_chunk; ++i)
    MPI_Comm_free(&Acc[i]);
}
