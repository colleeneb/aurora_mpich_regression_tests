#include "mpi.h"
#include <stdio.h>
int main(int argc, char *argv[])
{
    MPI_Datatype type, type2;

    MPI_Init(&argc, &argv);

    // make a data type of 3 integers and call it type2
    MPI_Type_contiguous(3, MPI_INT, &type2);
    MPI_Type_commit(&type2);

#ifdef FREE_COMMIT
    MPI_Type_free(&type2);
#endif

    MPI_Finalize();
    return 0;
}
