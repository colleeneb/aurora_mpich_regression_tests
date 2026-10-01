// Reproducer: MPICH aborts in handle_to_fd() during intra-node GPU IPC on Aurora.
//
//   handle_to_fd: No such file or directory
//   src/gpu/mpl_gpu_ze.c:701: int handle_to_fd(int, int, int *): Assertion `ret != -1' failed.
//
// Rank 0 and rank 1 on one node, one tile each. Each rank makes 2 device allocations of
// 11 * 1536 bytes, then posts one Isend and one Irecv per (allocation, slice) to the other
// rank and waits. The abort happens on the first exchange: no re-mesh, no reallocation,
// no second round.
//
// Run with the IPC handle cache limited to one entry:
//   MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=1
// The failure needs one more allocation than the cache holds. With the default cache of 16
// the same code aborts at 17 allocations and runs clean at 16; the boundary follows the cvar
// exactly (9 allocations: aborts at cache 8, clean at cache 16; 2 allocations: aborts at
// cache 1, clean at cache 1024).
//
// The uncached allocation is what fails. Its 11 concurrent messages all share one GEM handle
// (gem_hash in mpl_gpu_ze.c is keyed on the pointer, with no reference count), so the first
// message to complete closes the handle and the rest convert one that is already gone.
//
// Concurrency is required, not the non-blocking API: MPI_Sendrecv per slice runs clean, and
// so does Isend/Irecv + Waitall per slice. Several messages must also share one allocation --
// 187 allocations of 1 slice run clean, while 17 allocations of 11 slices abort with the same
// 187 messages. Slice size matters too: 1536 bytes aborts, 96 and 24576 run clean.
//
// Still failing with PR #7980 applied (issue #7977).
//
// Build (default Aurora modules plus the aurora_test MPICH):
//   module use ~/modulefiles && module swap mpich aurora_test
//   MPICH_CXX=icpx mpicxx -fsycl -O2 -g -std=c++17 rt_min.cpp -o rt_min
// Run:
//   MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=1 mpiexec -np 2 -ppn 2 ./gpu_tile_compact.sh ./rt_min
//
// Observed with MPICH 5.1.0a1 (aurora_test build) on Intel Data Center GPU Max 1550.
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <cstdio>

static const int NALLOC = 2;
static const int NSLICE = 11;
static const size_t SLICE = 1536;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    if (rank == 0) fprintf(stderr, "need exactly 2 ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  int peer = 0;
  if (rank == 0) {
    peer = 1;
  }

  sycl::queue q{sycl::gpu_selector_v};
  unsigned char *send[NALLOC];
  unsigned char *recv[NALLOC];
  for (int a = 0; a < NALLOC; ++a) {
    send[a] = sycl::malloc_device<unsigned char>(SLICE * NSLICE, q);
    recv[a] = sycl::malloc_device<unsigned char>(SLICE * NSLICE, q);
    if (!send[a] || !recv[a]) {
      fprintf(stderr, "[rank %d] device allocation failed\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
    q.memset(send[a], 1 + a, SLICE * NSLICE);
    q.memset(recv[a], 0, SLICE * NSLICE);
  }
  q.wait();

  MPI_Request reqs[2 * NALLOC * NSLICE];
  int n = 0;
  for (int a = 0; a < NALLOC; ++a) {
    for (int s = 0; s < NSLICE; ++s) {
      MPI_Irecv(recv[a] + s * SLICE, SLICE, MPI_BYTE, peer, a * 64 + s, MPI_COMM_WORLD, &reqs[n++]);
    }
  }
  for (int a = 0; a < NALLOC; ++a) {
    for (int s = 0; s < NSLICE; ++s) {
      MPI_Isend(send[a] + s * SLICE, SLICE, MPI_BYTE, peer, a * 64 + s, MPI_COMM_WORLD, &reqs[n++]);
    }
  }
  MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);

  if (rank == 0) printf("RESULT status=PASS allocations=%d slices=%d bytes=%zu\n", NALLOC, NSLICE, SLICE);
  for (int a = 0; a < NALLOC; ++a) {
    sycl::free(send[a], q);
    sycl::free(recv[a], q);
  }
  MPI_Finalize();
  return 0;
}
