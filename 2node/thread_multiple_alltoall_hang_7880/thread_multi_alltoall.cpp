#include <mpi.h>
#include <sycl/sycl.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// User-configurable parameters (set from argv in main).
struct Config {
    int    iterations   = 100;      // number of Alltoall iterations
    size_t count_per_pk = 1048576;  // elements sent to EACH peer per iteration
};

// Pick a GPU device for a given thread index (round-robin over available GPUs).
sycl::device pick_gpu_device(unsigned thread_index) {
    std::vector<sycl::device> gpus;
    for (const auto& plt : sycl::platform::get_platforms()) {
        for (const auto& dev : plt.get_devices()) {
            if (dev.is_gpu()) gpus.push_back(dev);
        }
    }
    if (gpus.empty()) {
        throw std::runtime_error("No SYCL GPU device found.");
    }
    return gpus[thread_index % gpus.size()];
}

// Perform `cfg.iterations` GPU-buffer Alltoall exchanges on `comm`.
// Buffers are allocated on the GPU via sycl::malloc_device.
void run_on_comm(MPI_Comm comm, const char* label,
                 unsigned thread_index, Config cfg) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    using elem_t = int;
    const size_t per_peer    = cfg.count_per_pk;        // elements per peer
    const size_t total_elems = static_cast<size_t>(size) * per_peer;
    const size_t total_bytes = total_elems * sizeof(elem_t);

    // Each thread gets its own SYCL queue on a (possibly distinct) GPU.
    sycl::device dev = pick_gpu_device(thread_index);
    sycl::queue  q(dev);

    // Allocate send/recv buffers on the GPU device.
    elem_t* d_send = sycl::malloc_device<elem_t>(total_elems, q);
    elem_t* d_recv = sycl::malloc_device<elem_t>(total_elems, q);
    if (!d_send || !d_recv) {
        std::ostringstream oss;
        oss << "[" << label << "] rank " << rank
            << " ERROR: sycl::malloc_device failed\n";
        std::cerr << oss.str();
        if (d_send) sycl::free(d_send, q);
        if (d_recv) sycl::free(d_recv, q);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Initialize the send buffer on the device. Element value encodes the
    // destination peer index and the source rank, so results are checkable.
    q.submit([&](sycl::handler& h) {
        elem_t* send = d_send;
        int r = rank;
        size_t pp = per_peer;
        h.parallel_for(sycl::range<1>(total_elems), [=](sycl::id<1> idx) {
            size_t i    = idx[0];
            size_t peer = i / pp;               // which peer this slice targets
            send[i]     = static_cast<elem_t>(r * 1000 + static_cast<int>(peer));
        });
    }).wait();

    // Iterated Alltoall over GPU memory (requires GPU-aware MPI).
    double t0 = MPI_Wtime();
    for (int it = 0; it < cfg.iterations; ++it) {
        MPI_Alltoall(d_send, static_cast<int>(per_peer), MPI_INT,
                     d_recv, static_cast<int>(per_peer), MPI_INT,
                     comm);
    }
    double t1 = MPI_Wtime();

    // Copy back one representative slot per source peer for a quick sanity check.
    // After Alltoall, d_recv's block from source `s` holds values s*1000 + rank.
    std::vector<elem_t> host_check(size);
    for (int s = 0; s < size; ++s) {
        // First element of the block received from source s:
        q.memcpy(&host_check[s], d_recv + static_cast<size_t>(s) * per_peer,
                 sizeof(elem_t)).wait();
    }

    bool ok = true;
    for (int s = 0; s < size; ++s) {
        elem_t expected = static_cast<elem_t>(s * 1000 + rank);
        if (host_check[s] != expected) ok = false;
    }

    std::ostringstream oss;
    oss << "[" << label << "] rank " << rank << "/" << size
        << "  iters=" << cfg.iterations
        << "  count/peer=" << per_peer
        << "  buf=" << total_bytes << " B"
        << "  device=" << dev.get_info<sycl::info::device::name>()
        << "  time=" << (t1 - t0) << " s"
        << "  verify=" << (ok ? "OK" : "FAIL") << "\n";
    std::cout << oss.str();

    sycl::free(d_send, q);
    sycl::free(d_recv, q);
}

Config parse_args(int argc, char** argv, int world_rank) {
    Config cfg;
    // Usage: prog [iterations] [count_per_peer]
    if (argc > 1) cfg.iterations   = std::atoi(argv[1]);
    if (argc > 2) cfg.count_per_pk = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));

    if (cfg.iterations <= 0 || cfg.count_per_pk == 0) {
        if (world_rank == 0) {
            std::cerr << "Invalid parameters.\n"
                      << "Usage: mpirun -np N ./prog [iterations>0] [count_per_peer>0]\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (provided < MPI_THREAD_MULTIPLE) {
        if (world_rank == 0) {
            std::cerr << "ERROR: MPI_THREAD_MULTIPLE not provided (got level "
                      << provided << ").\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    Config cfg = parse_args(argc, argv, world_rank);

    // Two subcommunicators, each containing EVERY rank.
    MPI_Comm comm_a = MPI_COMM_NULL, comm_b = MPI_COMM_NULL;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_a);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_b);

    if (world_rank == 0) {
        std::cout << "World size = " << world_size
                  << "; iterations = " << cfg.iterations
                  << "; count/peer = " << cfg.count_per_pk
                  << " (both subcommunicators contain all ranks)\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Each subcommunicator driven by its own thread, each with its own GPU queue.
    std::thread t_a(run_on_comm, comm_a, "comm_a", 0u, cfg);
    std::thread t_b(run_on_comm, comm_b, "comm_b", 1u, cfg);

    t_a.join();
    t_b.join();

    MPI_Comm_free(&comm_a);
    MPI_Comm_free(&comm_b);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
