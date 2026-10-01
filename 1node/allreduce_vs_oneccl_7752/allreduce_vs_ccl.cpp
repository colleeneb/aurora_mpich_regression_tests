#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

#ifndef ITER_MAX
#define ITER_MAX 100
#endif

#ifndef ITER_MIN
#define ITER_MIN 20
#endif

template <typename fp> bool almost_equal(fp x, fp y, int ulp) {
  return std::abs(x - y) <=
             std::numeric_limits<fp>::epsilon() * std::abs(x + y) * ulp ||
         std::abs(x - y) < std::numeric_limits<fp>::min();
}

void bench(int *current_iter, unsigned long *min_time,
           const std::function<void()> &f) {

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  MPI_Barrier(MPI_COMM_WORLD);

  // Save start and end
  const unsigned long l_start =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  f();
  const unsigned long l_end =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();

  unsigned long start, end;
  MPI_Allreduce(&l_start, &start, 1, MPI_UNSIGNED_LONG, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(&l_end, &end, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);

  unsigned long time = end - start;

  if (time >= *min_time) {
    *current_iter = *current_iter + 1;
  } else {

    if (my_rank == 0)
      std::cout << "-New min: " << time << "ns, after " << *current_iter
                << " previous iterations" << std::endl;

    *current_iter = 0;
    *min_time = time;
  }
}

unsigned long bench_loop(const std::function<void()> &f) {

  int current_iter = 0;
  int total_iter = 0;
  unsigned long min_time = std::numeric_limits<unsigned long>::max();

  for (total_iter = 0, current_iter = 0;
       total_iter < ITER_MAX && current_iter < ITER_MIN; total_iter++) {
    bench(&current_iter, &min_time, f);
  }
  return min_time;
}

void verify(sycl::queue Q, float *local_sum_cpu, float *global_sum_gpu,
            float *global_sum_cpu_tmp, int count) {

  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get Result Back
  Q.copy(global_sum_gpu, global_sum_cpu_tmp, count).wait();

  for (int i = 0; i < count; i++) {
    float got = global_sum_cpu_tmp[i];
    float expected = local_sum_cpu[i] * size;
    if (!almost_equal(got, expected, 40)) {
      std::cout << "Got: " << got << ", Expected: " << expected
                << ", Delta: " << got - expected << std::endl;
      throw std::runtime_error("Incorrect value at index " + std::to_string(i));
    }
  }
}

unsigned long bench_oneccl(sycl::queue Q, float *local_sum_gpu,
                           float *local_sum_cpu, float *global_sum_gpu,
                           float *global_sum_cpu_tmp, int count) {

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  /* create kvs */
  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type main_addr;
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    main_addr = kvs->get_address();
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
  } else {
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    kvs = ccl::create_kvs(main_addr);
  }

  /* create communicator */
  auto dev = ccl::create_device(Q.get_device());
  auto ctx = ccl::create_context(Q.get_context());
  auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

  /* create stream */
  auto stream = ccl::create_stream(Q);

  auto time = bench_loop([&]() {
    ccl::allreduce(local_sum_gpu, global_sum_gpu, count, ccl::datatype::float32,
                   ccl::reduction::sum, comm, stream)
        .wait();
  });

  verify(Q, local_sum_cpu, global_sum_gpu, global_sum_cpu_tmp, count);
  return time;
}

unsigned long bench_mpi(sycl::queue Q, float *local_sum_gpu,
                        float *local_sum_cpu, float *global_sum_gpu,
                        float *global_sum_cpu_tmp, int count) {

  auto time = bench_loop([&]() {
    MPI_Allreduce(local_sum_gpu, global_sum_gpu, count, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
  });
  verify(Q, local_sum_cpu, global_sum_gpu, global_sum_cpu_tmp, count);
  return time;
}

using BenchFn = unsigned long (*)(sycl::queue, float *, float *, float *,
                                  float *, int);

unsigned long run(std::string name, BenchFn f, sycl::queue &Q,
                  float *local_sum_gpu, float *local_sum, float *global_sum_gpu,
                  float *global_sum, int count) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if (rank == 0) {
    std::cout << name << ": Starting Bench " << std::endl;
  }

  auto time = f(Q, local_sum_gpu, local_sum, global_sum_gpu, global_sum, count);

  unsigned long size = count * sizeof(float);
  if (rank == 0) {
    std::cout << name << ": Time: " << time * 1e-9 << "s, "
              << "GoodPut: " << 2. * comm_size * size / time << "GB/s"
              << std::endl;
  }

  return time;
}

int main() {

  ccl::init();
  MPI_Init(NULL, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  auto Ds = sycl::platform{}.get_devices();
  const sycl::device D = Ds[rank % Ds.size()];

  const sycl::context C(D);
  sycl::queue Q(C, D);

  // 1 GiB
  unsigned long size = 1 << 30;
  if (rank == 0)
    std::cout << "Problem size: " << size << "bytes, "
              << "Number of rank: " << comm_size << std::endl;
  int count = size / sizeof(float);
  float *global_sum_gpu = sycl::malloc_device<float>(count, Q);
  float *local_sum_gpu = sycl::malloc_device<float>(count, Q);

  std::vector<float> global_sum(count);
  std::vector<float> local_sum(count);

  std::vector<float> global_sum_tmp(count);

  std::mt19937 rng(2320);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::generate(local_sum.begin(), local_sum.end(),
                [&]() { return dist(rng); });

  Q.copy(local_sum.data(), local_sum_gpu, count).wait();

  auto oneccl = run("oneCCL", &bench_oneccl, Q, local_sum_gpu, local_sum.data(),
                    global_sum_gpu, global_sum.data(), count);

  auto mpi = run("MPI", &bench_mpi, Q, local_sum_gpu, local_sum.data(),
                 global_sum_gpu, global_sum.data(), count);

  if (rank == 0)
    std::cout << "Time ratio oneCCL / MPI: " << 1. * oneccl / mpi << std::endl;
}
