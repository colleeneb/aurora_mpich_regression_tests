program min_halo
    use mpi_f08
    implicit none

    integer, parameter :: NELEM = 104*104*3*2      ! 64896 MPI_REAL4, > IPC threshold                                                                                                                                                             
    integer, parameter :: DIMS(3) = [3, 2, 2]

    real, allocatable :: sl(:), sr(:), rl(:), rr(:)
    type(MPI_Request) :: rreq(2), sreq(2)
    integer :: nranks, pid, ierr, niter, it, dir, i, c(3), lft(3), rgt(3)
    integer(8) :: nb
    character(len=32) :: arg

    call MPI_Init(ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, pid, ierr)
    if (nranks /= product(DIMS)) call MPI_Abort(MPI_COMM_WORLD, 2, ierr)

    niter = 25
    call get_command_argument(1, arg)
    if (len_trim(arg) > 0) read (arg, *) niter

    c(1) = mod(pid, DIMS(1))
    c(2) = mod(pid/DIMS(1), DIMS(2))
    c(3) = pid/(DIMS(1)*DIMS(2))
    do dir = 1, 3
        lft(dir) = neighbour(c, dir, -1)
        rgt(dir) = neighbour(c, dir, +1)
    end do

    allocate (sl(NELEM), sr(NELEM), rl(NELEM), rr(NELEM))
    !$omp target enter data map(alloc: sl, sr, rl, rr)          

    do it = 1, niter
        do dir = 1, 3
            !$omp target data use_device_addr(sl, sr, rl, rr)                                                                                                                                                                                     
            call MPI_Irecv(rl, NELEM, MPI_REAL4, lft(dir), lft(dir),     MPI_COMM_WORLD, rreq(1), ierr)
            call MPI_Irecv(rr, NELEM, MPI_REAL4, rgt(dir), rgt(dir) + 1, MPI_COMM_WORLD, rreq(2), ierr)
            call MPI_Isend(sl, NELEM, MPI_REAL4, lft(dir), pid + 1,      MPI_COMM_WORLD, sreq(1), ierr)
            call MPI_Isend(sr, NELEM, MPI_REAL4, rgt(dir), pid,          MPI_COMM_WORLD, sreq(2), ierr)
            call MPI_Wait(rreq(1), MPI_STATUS_IGNORE, ierr)
            call MPI_Wait(sreq(1), MPI_STATUS_IGNORE, ierr)
            call MPI_Wait(rreq(2), MPI_STATUS_IGNORE, ierr)
            call MPI_Wait(sreq(2), MPI_STATUS_IGNORE, ierr)
            !$omp end target data                                                                                                                                                                                                                 

            nb = 0
            !$omp target teams distribute parallel do map(alloc: rl) reduction(+: nb)                                                                                                                                                             
            do i = 1, NELEM
                if (rl(i) /= real(i)) nb = nb + 1
            end do
            !$omp end target teams distribute parallel do                                                                                                                                                                                         
            nb = 0
            !$omp target teams distribute parallel do map(alloc: rr) reduction(+: nb)                                                                                                                                                             
            do i = 1, NELEM
                if (rr(i) /= real(i)) nb = nb + 1
            end do
            !$omp end target teams distribute parallel do                                                                                                                                                                                         
        end do
    end do

    !$omp target exit data map(delete: sl, sr, rl, rr)                                                                                                                                                                                            
    deallocate (sl, sr, rl, rr)

    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (pid == 0) write (*,'(A,I0,A)') 'RESULT: PASS (', niter, ' iterations)'
    call MPI_Finalize(ierr)

contains

    integer function neighbour(c0, d, step)
        integer, intent(in) :: c0(3), d, step
        integer :: q(3)
        q = c0
        q(d) = q(d) + step
        neighbour = modulo(q(1), DIMS(1)) + modulo(q(2), DIMS(2))*DIMS(1) &
                    + modulo(q(3), DIMS(3))*DIMS(1)*DIMS(2)
    end function neighbour

end program min_halo
