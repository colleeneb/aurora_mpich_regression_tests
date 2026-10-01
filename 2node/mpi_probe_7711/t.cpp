#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// this triggers for 2 nodes                                                                                                                                                                                                                                 
#define NUM 1000000
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    // Size of the default communicator                                                                                                                                                                                                                      
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get my rank and do the corresponding job                                                                                                                                                                                                              
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    printf("rank, size %d %d\n", my_rank, size);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if( my_rank == 0)
      {
        for(int i=0; i < NUM*(size-1); i++ ) {
          int buffer[3] = {123, 456, 789};
          MPI_Send(buffer, 3, MPI_INT, (i % (size-1))+1, 0, MPI_COMM_WORLD);
        }
      }
    if( my_rank != 0)
      {
        for(int i=0; i < NUM; i++ ) {
          // Retrieve information about the incoming message                                                                                                                                                                                                 
          MPI_Status status;
          MPI_Probe(MPI_ANY_SOURCE , MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          int count;
          MPI_Get_count(&status, MPI_INT, &count);

          // Allocate the buffer now that we know how many elements there are                                                                                                                                                                                
          int* buffer = (int*)malloc(sizeof(int) * count);

          // Finally receive the message                                                                                                                                                                                                                     
          MPI_Recv(buffer, count, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          free(buffer);
        }
      }

    MPI_Finalize();
    return 0;
}
