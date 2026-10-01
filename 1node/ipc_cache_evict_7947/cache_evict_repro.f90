program cache_evict_repro
    use mpi_f08
    use iso_fortran_env, only: output_unit
    implicit none

    integer, parameter :: BUFFER_ELEMENTS = 524288                                                                                                                                                         
    integer, parameter :: MSG_TAG = 1703

    integer :: my_rank, ierr
    real, allocatable :: buf_a(:), buf_b(:)
    type(MPI_Request) :: requests(2)

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)

    allocate (buf_a(BUFFER_ELEMENTS))
    allocate (buf_b(BUFFER_ELEMENTS))

    !$omp target data map(alloc: buf_a, buf_b)                                                                                                                                                                                                        
    !$omp target data use_device_addr(buf_a, buf_b)                                                                                                                                                                                                   
    if (my_rank == 0) then
        call MPI_Isend(buf_a(1), BUFFER_ELEMENTS, MPI_REAL, 1, MSG_TAG, &
                       MPI_COMM_WORLD, requests(1), ierr)
        call MPI_Isend(buf_b(1), BUFFER_ELEMENTS, MPI_REAL, 1, MSG_TAG+1, &
                       MPI_COMM_WORLD, requests(2), ierr)
    else
        call MPI_Irecv(buf_a(1), BUFFER_ELEMENTS, MPI_REAL, 0, MSG_TAG, &
                       MPI_COMM_WORLD, requests(1), ierr)
        call MPI_Irecv(buf_b(1), BUFFER_ELEMENTS, MPI_REAL, 0, MSG_TAG+1, &
                       MPI_COMM_WORLD, requests(2), ierr)
    end if
    call MPI_Waitall(2, requests, MPI_STATUSES_IGNORE, ierr)
    !$omp end target data                                                                                                                                                                                                                             
    !$omp end target data                                                                                                                                                                                                                             

    deallocate (buf_a, buf_b)

    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (my_rank == 0) write (*,'(A)') 'RESULT: PASS'
    call MPI_Finalize(ierr)

end program cache_evict_repro
