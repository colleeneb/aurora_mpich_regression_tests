program oneway_repro
    use mpi_f08
    use iso_fortran_env, only: output_unit
    implicit none

    integer, parameter :: SIZES(2) = [4096, 256]        ! elements: 16384 and 1024 bytes                                                                                                                                                   
    integer, parameter :: BUFFER_ELEMENTS = 524288
    integer, parameter :: MSG_TAG = 1702

    integer :: my_rank, ierr, iter
    real, allocatable :: gpu_buf(:)

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)

    ! Allocated and mapped once, so the device address is fixed for the run.                                                                                                                                                               
    allocate (gpu_buf(BUFFER_ELEMENTS))

    !$omp target data map(alloc: gpu_buf)                                                                                                                                                                                                  
    do iter = 1, 2
        !$omp target data use_device_addr(gpu_buf)                                                                                                                                                                                         
        if (my_rank == 0) then
            call MPI_Send(gpu_buf(1), SIZES(iter), MPI_REAL, 1, MSG_TAG, &
                          MPI_COMM_WORLD, ierr)
        else
            call MPI_Recv(gpu_buf(1), SIZES(iter), MPI_REAL, 0, MSG_TAG, &
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
        end if
        !$omp end target data                                                                                                                                                                                                              

        if (my_rank == 0) then
            write (*,'(A,I0,A,I0,A)') 'ITER ', iter, ': ', SIZES(iter)*4, ' bytes'
            flush (output_unit)
        end if
    end do
    !$omp end target data                                                                                                                                                                                                                  

    deallocate (gpu_buf)

    ! Rank 1 aborts inside MPI_Recv on failure, so it never reaches here.                                                                                                                                                                  
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (my_rank == 0) write (*,'(A)') 'RESULT: PASS'
    call MPI_Finalize(ierr)

end program oneway_repro
