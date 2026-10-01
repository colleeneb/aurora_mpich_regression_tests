#include <iostream>

#include <sycl/sycl.hpp>
#include <mpi.h>

namespace {
  using data_type = double;
  MPI_Datatype mpi_type = MPI_DOUBLE;
  constexpr size_t type_size = sizeof(data_type);
  constexpr size_t data_count = 1;
  constexpr data_type fill_value = 1;
}

int main(int argc, char* argv[]) {
  
  MPI_Init(&argc, &argv);
  
  int mpi_world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); 
  
  sycl::queue sycl_queue{sycl::gpu_selector_v};
  data_type* origin_buffer = sycl::malloc_host<data_type>(data_count, sycl_queue);
  data_type* window_buffer = sycl::malloc_host<data_type>(data_count, sycl_queue);
  
  sycl_queue.fill<data_type>(origin_buffer, fill_value, data_count);sycl_queue.wait_and_throw();

  MPI_Win mpi_window;
  MPI_Win_create(window_buffer, data_count * type_size, type_size, MPI_INFO_NULL, MPI_COMM_WORLD, &mpi_window);

  MPI_Win_lock_all(MPI_MODE_NOCHECK, mpi_window);

  const MPI_Aint target_offset = 0;
  MPI_Put(origin_buffer, data_count, mpi_type, mpi_rank, target_offset, data_count, mpi_type, mpi_window);

  MPI_Win_unlock_all(mpi_window);
  MPI_Win_free(&mpi_window);   

  sycl_queue.wait_and_throw();
  sycl::free(origin_buffer, sycl_queue);
  sycl::free(window_buffer, sycl_queue);
  
  MPI_Finalize();
  return 0;
}
