#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <sycl/sycl.hpp>
#include <filesystem>
#define MESSAGE_SIZE 4

int main(){
    MPI_Init(NULL, NULL);

    sycl::queue syclQ{sycl::gpu_selector_v };

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    MPI_File outFile;
    MPI_File_open(
        MPI_COMM_WORLD, "test", MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL,
        MPI_INFO_NULL, &outFile);
    MPI_Status status;
    char *bufToWrite_host = (char*)malloc(sizeof(char)*MESSAGE_SIZE);
    char *bufToWrite_device = sycl::malloc_device<char>(MESSAGE_SIZE, syclQ);
    snprintf(bufToWrite_host, MESSAGE_SIZE, "%3d", rank);
    printf("%s\n", bufToWrite_host);

    syclQ.memcpy( bufToWrite_device, bufToWrite_host, sizeof(char)*MESSAGE_SIZE).wait();
    MPI_File_write_at(
                          outFile, rank * MESSAGE_SIZE,
                          bufToWrite_device, MESSAGE_SIZE, MPI_CHAR, &status);

    if(status.MPI_ERROR != MPI_SUCCESS) {
      printf( "FAIL %d\n", status.MPI_ERROR );
      return 1;
    }

    MPI_File_close(&outFile);

    MPI_Barrier(MPI_COMM_WORLD);

    if( rank == 0 ) {

      std::filesystem::path p{"test"};

      std::cout << "The size of " << p.u8string() << " is " <<
        std::filesystem::file_size(p) << " bytes.\n";
      if( std::filesystem::file_size(p) == 0 ) {
        std::cout << "This is incorrect" << std::endl;
        return 1;
      }
    }

    MPI_Finalize();
    return 0;
}
