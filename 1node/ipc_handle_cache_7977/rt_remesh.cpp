// Reproducer: MPICH aborts on the IPC handle cache in_use counter during intra-node GPU IPC.
//
//   Assertion failed in file src/mpid/ch4/shm/ipc/gpu/gpu_post.c at line 1095: entry->in_use >= 0
//
// Rank 0 and rank 1 on one node, one tile each. Each round every rank posts one Isend and one
// Irecv per (allocation, slice) to the other rank and waits. Between rounds the buffers are
// freed and allocated again, which is what AthenaK's Kokkos::realloc does at a re-mesh: the
// new allocation is made before the old one is freed, so freed addresses come back for other
// buffers.
//
// Freeing is the point. sycl::free runs MPIDI_GPU_handle_free_hook, which calls
// ipc_track_cache_delete, which compacts ipc_handle_cache with memmove (gpu_post.c:306).
// PR #7980 stores a pointer into that array in the request, so entries that shift down leave
// stored pointers on the wrong slot and the completion decrements a neighbour's counter.
//
// Run with pidfd handles:
//   MPIR_CVAR_CH4_IPC_ZE_SHAREABLE_HANDLE=pidfd mpiexec -np 2 -ppn 2 ./gpu_tile_compact.sh ./rt_remesh
//
// See rt_min.cpp for the other defect, which needs no re-mesh and fails under drmfd only.
//
// Build (default Aurora modules plus the aurora_test_7980 MPICH):
//   module use ~/modulefiles && module swap mpich aurora_test_7980
//   MPICH_CXX=icpx mpicxx -fsycl -O2 -g -std=c++17 rt_remesh.cpp -o rt_remesh
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <cstdio>

static const int NALLOC = 20;
static const int NSLICE = 12;
static const int NROUND = 20;
static const size_t SIZES[] = {1536, 4096, 1536, 24576};

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
  unsigned char *send[NALLOC] = {nullptr};
  unsigned char *recv[NALLOC] = {nullptr};
  MPI_Request reqs[2 * NALLOC * NSLICE];

  for (int round = 0; round < NROUND; ++round) {
    const size_t slice = SIZES[round % 4];

    // Kokkos::realloc order: allocate the new buffer before freeing the old one
    for (int a = 0; a < NALLOC; ++a) {
      unsigned char *ns = sycl::malloc_device<unsigned char>(slice * NSLICE, q);
      unsigned char *nr = sycl::malloc_device<unsigned char>(slice * NSLICE, q);
      if (!ns || !nr) {
        fprintf(stderr, "[rank %d] device allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
      if (send[a]) {
        sycl::free(send[a], q);
      }
      if (recv[a]) {
        sycl::free(recv[a], q);
      }
      send[a] = ns;
      recv[a] = nr;
      q.memset(send[a], 1 + a, slice * NSLICE);
      q.memset(recv[a], 0, slice * NSLICE);
    }
    q.wait();

    int n = 0;
    for (int a = 0; a < NALLOC; ++a) {
      for (int s = 0; s < NSLICE; ++s) {
        MPI_Irecv(recv[a] + s * slice, slice, MPI_BYTE, peer, a * 64 + s, MPI_COMM_WORLD, &reqs[n++]);
      }
    }
    for (int a = 0; a < NALLOC; ++a) {
      for (int s = 0; s < NSLICE; ++s) {
        MPI_Isend(send[a] + s * slice, slice, MPI_BYTE, peer, a * 64 + s, MPI_COMM_WORLD, &reqs[n++]);
      }
    }
    MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);

    if (rank == 0) {
      printf("round %d done (slice=%zu)\n", round, slice);
      fflush(stdout);
    }
  }

  if (rank == 0) printf("RESULT status=PASS allocations=%d slices=%d rounds=%d\n", NALLOC, NSLICE, NROUND);
  for (int a = 0; a < NALLOC; ++a) {
    sycl::free(send[a], q);
    sycl::free(recv[a], q);
  }
  MPI_Finalize();
  return 0;
}
