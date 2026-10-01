#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <sycl/sycl.hpp>
#include <cstring>

#define MESSAGE_SIZE 200000
int main(){
    MPI_Init(NULL, NULL);

    sycl::queue syclQ{sycl::gpu_selector_v };

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    MPI_File outFile;
    MPI_File_open(
        MPI_COMM_WORLD, "test", MPI_MODE_CREATE | MPI_MODE_WRONLY,
        MPI_INFO_NULL, &outFile);

    char *bufToWrite_host = (char*)malloc(sizeof(char)*MESSAGE_SIZE);
    char *bufToWrite_device = (char*)sycl::malloc_device<char>(MESSAGE_SIZE, syclQ);
    char *rank_string = (char*)malloc(sizeof(char)*2);
    sprintf(rank_string,"%d",rank);
    for(int i=0;i<MESSAGE_SIZE;i++)
        bufToWrite_host[i] = rank_string[0];

    //    printf("%s\n", bufToWrite_host);

    syclQ.memcpy( bufToWrite_device, bufToWrite_host, sizeof(char)*MESSAGE_SIZE);
    syclQ.wait();
    MPI_File_write_at_all(
                          outFile, rank * MESSAGE_SIZE,
                          bufToWrite_device, MESSAGE_SIZE, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_File_close(&outFile);
    MPI_Finalize();
    return 0;
}
