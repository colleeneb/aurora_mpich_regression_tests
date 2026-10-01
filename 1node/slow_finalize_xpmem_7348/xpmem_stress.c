#define _GNU_SOURCE
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>



static unsigned long *alloc_msg_buffer(unsigned long size, int rank, int initialize);
static void dump_self_maps(void);

int main(int argc, char *argv[])
{

  int  numiters = 1, reuse_buffer = 0, verbose = 0, dump_maps = 0;
  unsigned long bcast_length = 0;
  int  numranks, rank, rc, i;
  unsigned long *bcast_buff = 0;
  struct timeval t0, t1;

  for (i = 1; i < argc; i++) {
    if (strcmp("--verbose", argv[i]) == 0) {
      verbose = 1;
    } else if (strcmp("--iters", argv[i]) == 0) {
      numiters = atoi(argv[++i]);
    } else if (strcmp("--bcast", argv[i]) == 0) {
      bcast_length = strtoul(argv[++i], 0, 0);
    } else if (strcmp("--reuse", argv[i]) == 0) {
      reuse_buffer = 1;
    } else if (strcmp("--dump-maps", argv[i]) == 0) {
      dump_maps = 1;
    } else {
      printf("xpmem_stress [--bcast <size>] [--iters <N>] [--reuse] [--dump-maps]\n");
      return 0;
    }
  }

  rc = MPI_Init(&argc, &argv);

  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "Error starting MPI program. Terminating.\n");
    goto abort;
  }

  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  gettimeofday(&t0, NULL);

  for (i = 0; i < numiters; i++) {

    if ((i % 100 == 0) && (rank == 0 || verbose)) {
      printf("[%d] Starting iteration %d / %d\n", rank, i, numiters);
      fflush(stdout);
    }
    if (bcast_length > 0) {

      if (!bcast_buff || !reuse_buffer) {
        bcast_buff = alloc_msg_buffer(bcast_length, rank, rank == 0);
      }

      if (!bcast_buff) {
        rc = -1;
        goto abort;
      }

      rc = MPI_Bcast(bcast_buff, bcast_length / sizeof(unsigned long), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

      if (rc != MPI_SUCCESS) {
        fprintf(stderr, "(E) [%d] MPI_Bcast failed.\n", rank);
        goto abort;
      }
    }
  }

  gettimeofday(&t1, NULL);

  if (rank == 0 || verbose) {
    printf("[%d] Complete! (time: ~%ld secs)   Finalizing ...\n", rank, t1.tv_sec - t0.tv_sec);
    fflush(stdout);
  }

  if (dump_maps && (rank == 0 || verbose)) {
    dump_self_maps();
  }

  gettimeofday(&t0, NULL);
  MPI_Finalize();
  gettimeofday(&t1, NULL);

  if (rank == 0 || verbose) {
    printf("[%d] Complete! (time: ~%ld secs). Goodbye.\n", rank, t1.tv_sec - t0.tv_sec);
  }

  return 0;

 abort:
  MPI_Abort(MPI_COMM_WORLD, rc);
  return -1;
}


static unsigned long *alloc_msg_buffer(unsigned long size, int rank, int initialize)
{
  unsigned long *buff = (unsigned long *)malloc(size);

  if (!buff) {
    fprintf(stderr, "(E) [%d] Could not allocate message buffer.\n", rank);
    return 0;
  }

  if (initialize) {
    unsigned long i, N;

    for (i = 0, N = size/sizeof(unsigned long); i < N; i++)
      buff[i] = i;
  }

  return buff;
}

static void dump_self_maps(void) {

  char buffer[1024];
  FILE* fp = fopen("/proc/self/maps", "r");

  if (!fp) {
    fprintf(stderr, "(E) could not open /proc/self/maps\n");
    return;
  }

  while (fgets(buffer, 1024, fp))
    printf("%s", buffer);

  fflush(stdout);
}
