program test
!$ use omp_lib
  use mpi_f08
  implicit none
  integer                             :: i, iam, nRanks, allocStat
  integer                             :: mpiError, dcmplx_size
  type(MPI_Win)                       :: window
  integer (kind=MPI_ADDRESS_KIND)     :: buffer_size
  double complex, target, allocatable :: buffer(:)
  integer (kind=8)                    :: bufferSize, address
  double precision                    :: timerStart

  call MPI_Init(mpiError)
  call MPI_Comm_rank(MPI_COMM_WORLD, iam, mpiError)
  call MPI_Comm_size(MPI_COMM_WORLD, nRanks, mpiError)

  timerStart = MPI_WTime()

  if(iam == 0) then
    write(*,'(/a,1x,i15)') "Total Number of MPI ranks:                 ", nRanks
    write(*,'( a,1x,i15)') "Number of OMP threads per node:            ", omp_get_max_threads()
    write(*,'( a,1x,i15)') "Number of GPU devices visible to the rank: ", omp_get_num_devices()
    write(*,*) ""
  endif

!  bufferSize = 1024 * 63                  ! works
!  bufferSize = 1024 * 64                  ! works
!  bufferSize = 1024 * 65                  ! crashes
!  bufferSize = 1024 * 96                  ! crashes
!  bufferSize = 1024 * 128                 ! works
!  bufferSize = 100352                     ! crashes
   bufferSize = 101041                     ! crashes
!  bufferSize = 60624600_8                 ! crashes
!  bufferSize = 60624600_8 * 60_8          ! works
!  bufferSize = 1024_8 * 1024_8 * 1024_8   ! works
!  bufferSize = 1024_8 * 1024_8 * 1024_8 / 16_8 * 120_8 ! works in the implicit mode

  ! allocate array on the device
  !$omp allocators allocate (allocator(omp_target_device_mem_alloc):buffer)
  allocate(buffer(0:bufferSize-1), stat=allocStat)

  ! initialize array on the device
  !$omp target teams distribute parallel do has_device_addr(buffer)
  do i = 0, nRanks-1
    buffer(i) = dcmplx(dble(i), dble(i))
  enddo

  ! make sure all ranks reach this point
  call MPI_Barrier(MPI_COMM_WORLD, mpiError)

  ! create one-sided window
  call MPI_Type_size(MPI_DOUBLE_COMPLEX, dcmplx_size)
  buffer_size = bufferSize * dcmplx_size            ! size in bytes
  call MPI_Win_Create(buffer, buffer_size, dcmplx_size, MPI_INFO_NULL, MPI_COMM_WORLD, window, mpiError)
  call MPI_Win_fence(0, window)

  if(iam /= 0) then
    address = iam  ! convert to int_8
    ! each ranks reads one array element from rank 0
    call MPI_Get(buffer(address), 1, MPI_DOUBLE_COMPLEX, 0, &
                 address, 1, MPI_DOUBLE_COMPLEX, window, mpiError)
  endif

  call MPI_Win_fence(0, window)
  call MPI_Win_free(window, mpiError)

  ! deallocate the device-resident array
  deallocate(buffer)

  ! make sure all ranks reach this point
  call MPI_Barrier(MPI_COMM_WORLD, mpiError)

  if(iam == 0) then
    write(*,'(a,f7.1,a)') "Wall time: ", MPI_WTime() - timerStart, "s"
    write(*,'(/a)') "All Done"
  endif
  call MPI_Finalize(mpiError)

end program test
