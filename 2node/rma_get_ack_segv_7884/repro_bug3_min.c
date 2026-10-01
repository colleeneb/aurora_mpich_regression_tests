#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

int main(int argc, char **argv)
{
    int iam, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (np < 2) {
        if (iam == 0) fprintf(stderr, "need >=2 ranks\n");
        MPI_Finalize();
        return 1;
    }

    int nelems = getenv("NELEMS") ? atoi(getenv("NELEMS")) : 8192;
    int ngets  = getenv("NGETS")  ? atoi(getenv("NGETS"))  : 100000;
    int nfence = getenv("NFENCE") ? atoi(getenv("NFENCE")) : 200;
    if (nelems <= 0) nelems = 8192;
    if (ngets  <= 0) ngets  = 100000;
    if (nfence <= 0) nfence = 200;

    int dcmplx_size;
    MPI_Type_size(MPI_C_DOUBLE_COMPLEX, &dcmplx_size);

    /* Everyone contributes a window (dcmplx elements). Origin (rank 0)                                                                                                                                    
     * fetches from every non-self target repeatedly. */
    MPI_Aint win_elems = (MPI_Aint) nelems * 4;   /* room for varied offsets */
    MPI_Aint win_bytes = win_elems * dcmplx_size;

    double _Complex *winBuf = NULL;
    MPI_Alloc_mem(win_bytes, MPI_INFO_NULL, &winBuf);
    if (!winBuf) MPI_Abort(MPI_COMM_WORLD, 2);
    /* Fill with a recognizable pattern so if we ever want to inspect,                                                                                                                                     
     * we can. */
    for (MPI_Aint i = 0; i < win_elems; i++) {
        winBuf[i] = (double)i + I*(double)i;
    }

    MPI_Win window;
    MPI_Win_create(winBuf, win_bytes, dcmplx_size, MPI_INFO_NULL,
                   MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);

    /* Local origin buffer big enough for a single get. */
    double _Complex *originBuf = (double _Complex *) malloc((size_t) nelems * dcmplx_size);
    if (!originBuf) MPI_Abort(MPI_COMM_WORLD, 3);

    if (iam == 0) {
        printf("repro_bug3_min: np=%d nelems=%d (%d bytes) ngets=%d nfence=%d\n",
               np, nelems, nelems * dcmplx_size, ngets, nfence);
        fflush(stdout);
    }
    if (iam == 0) {
        int nissued = 0;
        for (int i = 0; i < ngets; i++) {
            int target = 1 + (i % (np - 1));  /* rotate over ranks 1..np-1 */
            /* vary source offset a bit so we don't hit an aggressive                                                                                                                                      
             * cache path */
            MPI_Aint disp = (i * 17) % (win_elems - nelems);

            MPI_Get(originBuf, nelems, MPI_C_DOUBLE_COMPLEX,
                    target, disp, nelems, MPI_C_DOUBLE_COMPLEX, window);
            nissued++;

            if ((i + 1) % nfence == 0) {
                MPI_Win_fence(0, window);
                if ((i + 1) % (nfence * 100) == 0) {
                    printf("  fenced after %d\n", i + 1);
                    fflush(stdout);
                }
            }
        }
        printf("origin issued %d gets\n", nissued);
    } else {
        /* Passive targets: just participate in fences. */
        for (int i = nfence; i <= ngets; i += nfence) {
            MPI_Win_fence(0, window);
        }
    }
    MPI_Win_fence(0, window);
    MPI_Win_free(&window);

    free(originBuf);
    MPI_Free_mem(winBuf);

    MPI_Barrier(MPI_COMM_WORLD);
    if (iam == 0) printf("\nAll Done\n");
    MPI_Finalize();
    return 0;
}
