#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <vector>

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
double bench(const std::string &label, size_t N_byte, int num_iteration, Func &&f,
             int message_ratio = 2) {
  unsigned long min_time = std::numeric_limits<unsigned long>::max();
  for (int r = 0; r < num_iteration; r++) {
    MPI_Barrier(MPI_COMM_WORLD);
    const unsigned long l_start =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    f(); // Execute the lambda

    const unsigned long l_end =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    unsigned long start, end;
    MPI_Reduce(&l_start, &start, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&l_end, &end, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    const unsigned long time = end - start;
    min_time = std::min(time, min_time);
  }

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  double bw;
  if (world_rank == 0) {
    bw = 1. * (world_size / message_ratio) * N_byte / min_time;
    std::cout << " " << label << " BW " << bw << " GB/s" << std::endl;
  }
  MPI_Bcast(&bw, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return bw;
}

template <typename T> unsigned run(uint64_t N, int num_iteration = 10) {

  MPI_Datatype mtype = mpi_type<T>::value;

  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int device_id = omp_get_default_device();
  int host_id = omp_get_initial_device();
  unsigned fail = 0;

  const uint64_t N_byte = N * sizeof(T);

  std::vector<T> A(N);

  // Fast than std::rand
  {
    std::minstd_rand rng{std::random_device{}()};
    std::uniform_int_distribution<> dist(0, 100);
    std::generate(A.begin(), A.end(), [&]() { return dist(rng); });
  }

  // Put some valid data to GPU memory. To avoid crazy packing optimization
  T *A2_gpu = (T *)omp_target_alloc_device(N_byte, device_id);
  omp_target_memcpy(A2_gpu, A.data(), N_byte, 0, 0, device_id, host_id);
  // Pinned memory
  T *A2_cpu = (T *)omp_target_alloc_host(N_byte, device_id);

  const auto omp_gpu_cpu =
      bench("OMP Target Memcpy-as-MPI, h0.GPU -> h0.CPU (pinned)", N_byte, num_iteration, [&]() {
        // Only half the rank does a zeMemCopy
        if (world_rank < world_size / 2)
          omp_target_memcpy(A2_cpu, A2_gpu, N_byte, 0, 0, host_id, device_id);
      });

  const auto mpi_gpu_cpu =
      bench("MPI (intra-node), h0.GPU -> h0.CPU (pinned)", N_byte, num_iteration, [&]() {
        if (world_rank < world_size / 2) {
          MPI_Send(
              /* data         = */ A2_gpu,
              /* count        = */ N,
              /* datatype     = */ mtype,
              /* destination  = */ world_rank + world_size / 2,
              /* tag          = */ 0,
              /* communicator = */ MPI_COMM_WORLD);
        } else {
          MPI_Recv(
              /* data         = */ A2_cpu,
              /* count        = */ N,
              /* datatype     = */ mtype,
              /* source       = */ world_rank - world_size / 2,
              /* tag          = */ 0,
              /* communicator = */ MPI_COMM_WORLD,
              /* status       = */ MPI_STATUS_IGNORE);
        }
      });

  if (world_rank == 0) {
    if (!almost_equal(mpi_gpu_cpu, omp_gpu_cpu, 0.10)) {
      std::cout << "Failed: MPI d2h(pinned) slower than OMP" << std::endl;
      fail += 1;
    } else {
      std::cout << "Success: MPI d2h(pinned) as fast than OMP" << std::endl;
    }
  }
  return fail;
}

int main() {
  MPI_Init(NULL, NULL);
  auto fail_count = run<double>(1 << 28);
  MPI_Finalize();
  exit(fail_count);
}
