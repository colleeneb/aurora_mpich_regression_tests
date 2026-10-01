#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>

//const int nmesg = 2;
const int nmesg = 16;
//const int nmesg = 24;
//const int nmesg = 32;
//const int nrep = 1;
const int nrep = 1000;
//const int nrep = 10000;
//const int nrep = 20000;
const int nmin = 128*1024;
//const int nmax = 128*1024;
//const int nmin = 256*1024;
const int nmax = 256*1024;
//const int nmin = 2*1024*1024;
//const int nmax = 2*1024*1024;

void sendrecv(double *dest[], double *src[], int n) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Request sreq[nmesg], rreq[nmesg];
  for(int i=0; i<nmesg; i++) {
    int k = 1 << i;
    int recv = (rank+k) % size;
    MPI_Irecv(dest[i], n, MPI_DOUBLE, recv, i, MPI_COMM_WORLD, &rreq[i]);
  }
  for(int i=0; i<nmesg; i++) {
    int k = 1 << i;
    int send = (rank+k*size-k) % size;
    MPI_Isend(src[i], n, MPI_DOUBLE, send, i, MPI_COMM_WORLD, &sreq[i]);
  }
  MPI_Waitall(nmesg, sreq, MPI_STATUS_IGNORE);
  MPI_Waitall(nmesg, rreq, MPI_STATUS_IGNORE);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  char name[MPI_MAX_PROCESSOR_NAME];
  int namelen;
  MPI_Get_processor_name(name, &namelen);
  //sycl::queue q{sycl::gpu_selector_v};
  sycl::platform plat{sycl::gpu_selector_v};
  auto devs = plat.get_devices();
  int ndev = devs.size();
  int devid = rank % ndev;
  printf("%s  rank %3i  device %2i\n", name, rank, devid);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  sycl::queue q{devs[devid]};
  double *src[nmesg], *srcg[nmesg], *dest[nmesg], *destg[nmesg];
  for(int i=0; i<nmesg; i++) {
    src[i] = (double*)malloc(nmax*sizeof(double));
    srcg[i] = (double*)sycl::malloc_device<double>(nmax, q);
    dest[i] = (double*)malloc(nmax*sizeof(double));
    destg[i] = (double*)sycl::malloc_device<double>(nmax, q);
#pragma omp parallel for
    for(int j=0; j<nmax; j++) {
      src[i][j] = i + j;
    }
  }

  int error = 0;
  int errort = 0;
  for(int n=nmin; n<=nmax; n*=2) {
    if(rank==0) printf("Testing n = %i ...", n);
    for(int rep=0; rep<nrep; rep++) {
      //sendrecv(dest, src, n);
      for(int i=0; i<nmesg; i++) {
	q.memcpy(srcg[i], src[i], n*sizeof(double));
	q.memset(destg[i], 0, n*sizeof(double));
      }
      q.wait();
      sendrecv(destg, srcg, n);
      for(int i=0; i<nmesg; i++) {
	q.memcpy(dest[i], destg[i], n*sizeof(double));
      }
      q.wait();
      for(int i=0; i<nmesg; i++) {
	for(int j=0; j<n; j++) {
	  if (dest[i][j] != src[i][j]) {
	    printf("\n  error %i dest[%i][%i] = %f expected %f\n", rep, i, j, dest[i][j], src[i][j]);
	    error++;
	    break;
	  }
	}
	if(error>0) break;
      }
      MPI_Allreduce(&error, &errort, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (errort>0) break;
    }
    if(errort>0) {
      if (rank==0) printf(" %i errors.\n", errort);
      break;
    } else {
      if (rank==0) printf(" done.\n");
    }
  }
  MPI_Finalize();
}
