#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <sycl/sycl.hpp>

constexpr double BYTES_TO_GB = 1.0/(1024.0*1024.0*1024.0);

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  int my_rank, n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  int count_per_peer = (argc > 1) ? std::atoi(argv[1]) : 262166;
  int n_iters        = (argc > 2) ? std::atoi(argv[2]) : 100;

  sycl::queue q;
  sycl::device dev = q.get_device();

  int total = count_per_peer * n_ranks;

  if (my_rank == 0)
    printf( "ranks=%d  count/peer=%d doubles (%.2f MB per send)  iters=%d\n",
            n_ranks, count_per_peer, count_per_peer * 8.0 / 1024.0 / 1024.0, n_iters );

  std::vector<MPI_Request> rreq(n_ranks);

  for (int it = 0; it < n_iters; it++) {

    double* sendbuf = sycl::malloc_device<double>(total, q);
    double* recvbuf = sycl::malloc_device<double>(total, q);

    for (int i = 0; i < n_ranks; i++)
      MPI_Irecv(recvbuf + (long long)i * count_per_peer, count_per_peer,
                MPI_DOUBLE, i, i, MPI_COMM_WORLD, &rreq[i]);
    for (int i = 0; i < n_ranks; i++)
      MPI_Send(sendbuf + (long long)i * count_per_peer, count_per_peer,
               MPI_DOUBLE, i, my_rank, MPI_COMM_WORLD);

    MPI_Waitall(n_ranks, rreq.data(), MPI_STATUSES_IGNORE);

    if( it%5 == 0 && my_rank == 0 ) {
      printf( "iteration: %d\n", it);
      double total_byte = dev.get_info<sycl::info::device::global_mem_size>();
      double free_byte  = dev.get_info<sycl::ext::intel::info::device::free_memory>();
      printf( "Used GPU memory on rank 0: %lf GB\n", (total_byte - free_byte)*BYTES_TO_GB );
      fflush(stdout);
    }
    sycl::free(sendbuf, q);
    sycl::free(recvbuf, q);
  }

  MPI_Finalize();
  return 0;
}
