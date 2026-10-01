#include <mpi.h>
#include <complex.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int iam, nRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    if (nRanks < 2) {
        if (iam == 0) fprintf(stderr, "need >= 2 ranks\n");
        MPI_Finalize(); return 1;
    }

    /* Buffer size chosen to be safely above the AM eager threshold                                                                                                                                                                                          
     * (default ~16 KB, but bump to 4 MiB doubles = 32 MiB to guarantee                                                                                                                                                                                      
     * the pipeline path fires). Env override to sweep. */
    size_t N = 4 * 1024 * 1024;   /* 4M doubles = 32 MiB */
    const char *env_n = getenv("N_DOUBLES");
    if (env_n) { size_t v = (size_t) strtoull(env_n, NULL, 10); if (v) N = v; }

    double _Complex *winBuf = (double _Complex *) malloc(N * sizeof(double _Complex));
    if (!winBuf) { fprintf(stderr, "rank %d: malloc failed\n", iam); MPI_Abort(MPI_COMM_WORLD, 2); }
    for (size_t i = 0; i < N; i++) winBuf[i] = (double)(iam + 1) + (double)i * I;

    int dcmplx_size;
    MPI_Type_size(MPI_C_DOUBLE_COMPLEX, &dcmplx_size);
    MPI_Aint buffer_size = (MPI_Aint) N * dcmplx_size;

    MPI_Win window;
    MPI_Win_create(winBuf, buffer_size, dcmplx_size,
                   MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);

    if (iam == 0)
        printf("repro_bug2: %d ranks, N=%zu doubles (%zu MiB), ENABLE_RMA=%s\n",
               nRanks, N, (N * sizeof(double _Complex)) >> 20,
               getenv("MPIR_CVAR_CH4_OFI_ENABLE_RMA") ?
                    getenv("MPIR_CVAR_CH4_OFI_ENABLE_RMA") : "<unset>");
    fflush(stdout);

    /* Rank 0 does one Get from rank 1's whole buffer. Its own buffer                                                                                                                                                                                        
     * is written into slot [0..N). One op, one large payload. */
    if (iam == 0) {
        double _Complex *dst = (double _Complex *) malloc(N * sizeof(double _Complex));
        if (!dst) { MPI_Abort(MPI_COMM_WORLD, 3); }
        MPI_Get(dst, (int)N, MPI_C_DOUBLE_COMPLEX,
                1, 0, (int)N, MPI_C_DOUBLE_COMPLEX, window);
        printf("rank 0: MPI_Get posted\n"); fflush(stdout);
        MPI_Win_fence(0, window);
        printf("rank 0: MPI_Win_fence returned (got payload OK)\n"); fflush(stdout);
        /* sanity print */
        printf("rank 0: dst[0]=(%g,%g)  dst[N-1]=(%g,%g)\n",
               creal(dst[0]), cimag(dst[0]),
               creal(dst[N-1]), cimag(dst[N-1]));
        free(dst);
    } else {
        MPI_Win_fence(0, window);
    }

    MPI_Win_free(&window);
    free(winBuf);
    MPI_Barrier(MPI_COMM_WORLD);
    if (iam == 0) printf("\nAll Done\n");
    MPI_Finalize();
    return 0;
}
