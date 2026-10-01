! MPI_Reduce with MPI_REAL + MPI_SUM returns garbage on Aurora MPICH 5.x
!
!   Environment:  Aurora PE 26.26.0, oneapi/release/2025.3.1 (ifx 2025.3.2)
!   Fails on:     mpich/opt/5.0.0.aurora_test.3c70a61  (current default)
!                 mpich/opt/develop-git.6037a7a
!   Correct on:   mpich/opt/4.2.3-intel   -> regression between 4.2.3 and 5.0
!
!   Build:  mpif90 -O2 reduce_repro.f90 -o reduce_repro
!   Run:    mpiexec -n 2 ./reduce_repro          (compute node, 2+ ranks)
!
! Each rank contributes 1/nranks, so every reduced element should be 1.0.
! Pass 1 (MPI_REAL) instead returns ~4.25e+37 in every element at every
! count. Pass 2 (MPI_REAL4, byte-identical to MPI_REAL) is correct, as is
! pass 3 (MPI_DOUBLE_PRECISION): the fault is specific to the MPI_REAL
! named constant, not to 4-byte reductions in general.
!
! Also verified: MPI_Allreduce fails the same way; compiling with or
! without GPU offload flags makes no difference, as do
! MPIR_CVAR_ENABLE_GPU=0 and MPICH_GPU_SUPPORT_ENABLED=0; single-rank runs
! are unaffected (no reduction actually performed).
!
! Application impact: fp32 histogram output files full of huge()/NaN.
! Workaround: use MPI_REAL4, or gather and sum on the root rank.

program reduce_repro
    use mpi
    implicit none
    integer :: rank, nranks, ierr

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
    if (rank == 0) print '(a,i0,a)', 'running on ', nranks, &
        ' ranks; every element should reduce to 1.0'

    call test_real4('MPI_REAL4 (control)', MPI_REAL4)
    call test_real4('MPI_REAL', MPI_REAL)
    call test_real8('MPI_DOUBLE_PRECISION (control)')

    call MPI_Finalize(ierr)

contains

    subroutine test_real4(label, dtype)
        character(*), intent(in) :: label
        integer, intent(in) :: dtype
        integer, parameter :: counts(4) = [1024, 16384, 131072, 524288]
        real(4), allocatable :: send(:), recv(:)
        integer :: k, n, nbad
        if (rank == 0) print '(/a)', label
        do k = 1, size(counts)
            n = counts(k)
            allocate(send(n), recv(n))
            send = 1.0_4 / real(nranks, 4)
            recv = 0.0_4
            call MPI_Reduce(send, recv, n, dtype, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
            if (rank == 0) then
                nbad = count(abs(recv - 1.0_4) > 1.0e-5_4)
                write(*,'(a,i7,a,i7,a,es10.3,2a)') '  count ', n, ': wrong ', nbad, &
                    ', max|recv| ', maxval(abs(recv)), '  ', merge('OK  ', 'FAIL', nbad == 0)
                if( nbad > 0 ) stop 1
             end if
            deallocate(send, recv)
        end do
    end subroutine test_real4

    subroutine test_real8(label)
        character(*), intent(in) :: label
        integer, parameter :: counts(4) = [1024, 16384, 131072, 524288]
        real(8), allocatable :: send(:), recv(:)
        integer :: k, n, nbad
        if (rank == 0) print '(/a)', label
        do k = 1, size(counts)
            n = counts(k)
            allocate(send(n), recv(n))
            send = 1.0_8 / real(nranks, 8)
            recv = 0.0_8
            call MPI_Reduce(send, recv, n, MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                            MPI_COMM_WORLD, ierr)
            if (rank == 0) then
                nbad = count(abs(recv - 1.0_8) > 1.0e-12_8)
                write(*,'(a,i7,a,i7,a,es10.3,2a)') '  count ', n, ': wrong ', nbad, &
                    ', max|recv| ', maxval(abs(recv)), '  ', merge('OK  ', 'FAIL', nbad == 0)
            end if
            deallocate(send, recv)
        end do
    end subroutine test_real8

end program reduce_repro
