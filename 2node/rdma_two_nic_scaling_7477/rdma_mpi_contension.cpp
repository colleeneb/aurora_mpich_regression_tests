#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <vector>
#include <cassert>

bool almost_equal(double x, double gold, float rel_tol = 0.20, double abs_tol = 0.0) {
  return std::abs(x - gold) <= std::max(rel_tol * std::max(std::abs(x), std::abs(gold)), abs_tol);
}

// Default template: no mapping
template <typename T> struct mpi_type {
  static_assert(sizeof(T) == 0, "Unsupported type for MPI mapping");
};

template <> struct mpi_type<int> {
  static constexpr MPI_Datatype value = MPI_INT;
};

template <> struct mpi_type<double> {
  static constexpr MPI_Datatype value = MPI_DOUBLE;
};

template <typename Func>
double bench(const std::string &label, size_t N_byte, int num_iteration, Func &&f, MPI_Comm comm) {
  unsigned long min_time = std::numeric_limits<unsigned long>::max();
  for (int r = 0; r < num_iteration; r++) {
    MPI_Barrier(comm);
    const unsigned long l_start =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    f(); // Execute the lambda

    const unsigned long l_end =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    unsigned long start, end;
    MPI_Reduce(&l_start, &start, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, comm);
    MPI_Reduce(&l_end, &end, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, comm);

    const unsigned long time = end - start;
    min_time = std::min(time, min_time);
  }

  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  double bw;
  if (world_rank == 0) {
    bw = .5 * (mpi_size)*N_byte / min_time;
    std::cout << label << " BW " << bw << " GB/s" << std::endl;
  }
  MPI_Bcast(&bw, 1, MPI_DOUBLE, 0, comm);
  return bw;
}

template <typename T>
double run(MPI_Comm comm, uint64_t N, std::string prefix, int num_iteration = 10) {

  MPI_Datatype mtype = mpi_type<T>::value;

  // Find out rank, size
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);

  int device_id = omp_get_default_device();
  int host_id = omp_get_initial_device();

  const uint64_t N_byte = N * sizeof(T);

  std::vector<T> A(N);
  // Fast than std::rand
  {
    std::minstd_rand rng{std::random_device{}()};
    std::uniform_int_distribution<> dist(0, 100);
    std::generate(A.begin(), A.end(), [&]() { return dist(rng); });
  }

  T *A2_gpu = (T *)omp_target_alloc_device(N_byte, device_id);
  omp_target_memcpy(A2_gpu, A.data(), N_byte, 0, 0, device_id, host_id);

  const auto mpi_gpu_gpu = bench(
      prefix + " GPU -> GPU", N_byte, num_iteration,
      [&]() {
        if (mpi_rank < mpi_size / 2) {
          MPI_Send(
              /* data         = */ A2_gpu,
              /* count        = */ N,
              /* datatype     = */ mtype,
              /* destination  = */ mpi_rank + mpi_size / 2,
              /* tag          = */ 0,
              /* communicator = */ comm);
        } else {
          MPI_Recv(
              /* data         = */ A2_gpu,
              /* count        = */ N,
              /* datatype     = */ mtype,
              /* source       = */ mpi_rank - mpi_size / 2,
              /* tag          = */ 0,
              /* communicator = */ comm,
              /* status       = */ MPI_STATUS_IGNORE);
        }
      },
      comm);
  omp_target_free(A2_gpu, device_id);
  return mpi_gpu_gpu;
}

int main() {
  MPI_Init(NULL, NULL);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  assert(world_size == 4);

  auto bw_2ppn = run<double>(MPI_COMM_WORLD, 1 << 28, "2 NIC");

  // Use color = 1 for even ranks, MPI_UNDEFINED for odd
  int color = (world_rank % 2 == 0) ? 1 : MPI_UNDEFINED;
  MPI_Comm even_comm;

  // Split the communicator
  double bw_1ppn;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &even_comm);

  if (color == 1) {
    bw_1ppn = run<double>(even_comm, 1 << 28, "1 NIC");
    MPI_Comm_free(&even_comm);
  }

  MPI_Finalize();
  if (world_rank == 0) {
    if (almost_equal(bw_2ppn, 2 * bw_1ppn, 0.20)) {
      std::cout << "Success: Using 2 NIC, is 2x faster than using 1 NIC!" << std::endl;
    } else {
      std::cout << "Fail: Using 2 NIC, is NOT 2x faster than using 1 NIC!" << std::endl;
      exit(1);
    }
  }
}
