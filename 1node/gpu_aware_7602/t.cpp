#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <mpi.h>
#include <sycl/sycl.hpp>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    // Get the number of processes and check only 2 processes are used
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if(num_ranks != 2)
    {
        printf("This application is meant to be run with 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double * snd_buf_cpu_1d_;
    double * rev_buf_cpu_1d_;
    int size = 1;
    MPI_Request  snd_req_[1];
    MPI_Request  rev_req_[1];
    sycl::queue queue;
    snd_buf_cpu_1d_ = sycl::malloc_device<double>( size, queue );
    rev_buf_cpu_1d_ = sycl::malloc_device<double>( size, queue );

    // this is ok:
       // snd_buf_cpu_1d_=(double*)malloc(size*sizeof(double));
       // rev_buf_cpu_1d_=(double*)malloc(size*sizeof(double));

    for(int i=0; i < 2; i++) {

      if(my_rank == 0)
    {
          MPI_Isend(snd_buf_cpu_1d_,size,MPI_DOUBLE,1,52, MPI_COMM_WORLD, &snd_req_[0]);

          MPI_Irecv(rev_buf_cpu_1d_,size,MPI_DOUBLE,1,52, MPI_COMM_WORLD, &rev_req_[0]);
    }
    else {
          MPI_Isend(snd_buf_cpu_1d_,size,MPI_DOUBLE,0,52, MPI_COMM_WORLD, &snd_req_[0]);

          MPI_Irecv(rev_buf_cpu_1d_,size,MPI_DOUBLE,0,52, MPI_COMM_WORLD, &rev_req_[0]);
    }


    std::vector<MPI_Status> s_stat(1),r_stat(1);
    MPI_Waitall(1,&rev_req_[0],&r_stat[0]);
    MPI_Waitall(1,&snd_req_[0],&s_stat[0]);

    }
    MPI_Finalize();

    return 0;
}
