/*
 * session.c -- an MPI "Sessions model" program.
 *
 * It creates a Session, derives a communicator from the "mpi://WORLD"
 * process set, then idles (pause()) until it receives a signal. 
 * 
 * With -DNS_FIX the program appends a suffix to $PMIX_NAMESPACE before any
 * MPI call, giving its MPI instance an identity distinct from the sibling
 * world process. 
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CK(call)                                                               \
    do {                                                                       \
        int _r = (call);                                                       \
        if (_r != MPI_SUCCESS) {                                               \
            char _es[MPI_MAX_ERROR_STRING];                                    \
            int _el;                                                           \
            MPI_Error_string(_r, _es, &_el);                                   \
            fprintf(stderr, "session: %s -> %d: %s\n", #call, _r, _es);        \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(void) {
#ifdef NS_FIX
    /* Work-around: give this MPI instance its own PMIx namespace, distinct
     * from the sibling world process, so the two do not share a bootstrap
     * identity. Must happen before any MPI call. */
    const char *ns = getenv("PMIX_NAMESPACE");
    if (ns) {
        char new_ns[512];
        snprintf(new_ns, sizeof(new_ns), "%s_session", ns);
        setenv("PMIX_NAMESPACE", new_ns, 1);
    }
#endif

    MPI_Session sh = MPI_SESSION_NULL;
    MPI_Group wgroup = MPI_GROUP_NULL;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Info sinfo = MPI_INFO_NULL;

    MPI_Info_create(&sinfo);
    MPI_Info_set(sinfo, "thread_level", "MPI_THREAD_SINGLE");

    CK(MPI_Session_init(sinfo, MPI_ERRORS_RETURN, &sh));
    CK(MPI_Group_from_session_pset(sh, "mpi://WORLD", &wgroup));
    CK(MPI_Comm_create_from_group(wgroup, "session_comm", MPI_INFO_NULL,
                                  MPI_ERRORS_RETURN, &comm));

    int rank = -1, size = -1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    fprintf(stderr, "session: comm created (rank %d / %d), idling\n", rank, size);

    /* Idle at zero CPU until the launcher signals us once the world process
     * has finished. No timeout, no polling. */
    pause();

    MPI_Comm_free(&comm);
    MPI_Group_free(&wgroup);
    MPI_Info_free(&sinfo);
    MPI_Session_finalize(&sh);
    return 0;
}

