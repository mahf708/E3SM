module RtmHorizRemap
  !-------------------------------------------------------------------------------------------
  !
  ! MOSART wrapper for shared horizontal remapping infrastructure.
  !
  ! Uses shr_horiz_remap_mod for the core algorithm (reading map files,
  ! building MPI communication patterns, and applying sparse matrix-vector
  ! multiply). This module adds MOSART-specific logic: decomposition mapping
  ! via rtmCTL%gindex, local cell index translation, and PIO output
  ! with cached decompositions.
  !
  ! MOSART is 1D (no vertical levels), so the write routine always produces
  ! 2D (lon, lat) output on the target grid.
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,        only: r8 => shr_kind_r8
  use shr_horiz_remap_mod, only: shr_horiz_remap_t, shr_horiz_remap_read_mapfile, &
                                  shr_horiz_remap_build_comm, shr_horiz_remap_apply
  use RtmSpmd,             only: masterproc, iam, npes, mpicom_rof, &
                                  MPI_INTEGER, MPI_REAL8
  use RtmVar,              only: iulog
  use RunoffMod,           only: rtmCTL
  use shr_sys_mod,         only: shr_sys_abort

  implicit none
  private
  save

  integer, parameter :: max_tapes = 3

  public :: rtm_horiz_remap_init
  public :: rtm_horiz_remap_field
  public :: rtm_horiz_remap_write
  public :: rtm_horiz_remap_is_active

  type rtm_horiz_remap_t
    type(shr_horiz_remap_t) :: shared
    integer, allocatable    :: send_cell_local(:)  ! local cell index per send entry
    logical  :: iodesc_2d_valid = .false.
  end type rtm_horiz_remap_t

  type(rtm_horiz_remap_t), target :: remap_data(max_tapes)

CONTAINS

  !-------------------------------------------------------------------------------------------
  logical function rtm_horiz_remap_is_active(t)
    integer, intent(in) :: t
    rtm_horiz_remap_is_active = remap_data(t)%shared%initialized
  end function rtm_horiz_remap_is_active

  !-------------------------------------------------------------------------------------------
  subroutine rtm_horiz_remap_init(t, mapfile)
    !
    ! Initialize horizontal remapping for history tape t by reading a
    ! mapping file and building the MPI communication pattern.
    !
    ! Uses the shared module for map file I/O and comm pattern building.
    ! MOSART-specific: builds gcol_to_rank using rtmCTL%gindex + MPI_Allgatherv,
    ! then converts send_gcol_list to local cell indices.
    !
    use pio,            only: iosystem_desc_t
    use shr_pio_mod,    only: shr_pio_getiosys
    use RtmVar,         only: inst_name

    ! Arguments
    integer,          intent(in) :: t        ! tape index
    character(len=*), intent(in) :: mapfile  ! path to ESMF mapping file

    ! Local variables
    type(iosystem_desc_t), pointer :: pio_subsystem
    integer           :: ierr
    integer           :: i, cnt, gcol, idx
    integer           :: n_my_cells, n_a
    integer           :: begr, endr, numr
    integer, allocatable :: my_gcol_list(:)
    integer, allocatable :: gcol_to_myidx(:)
    integer, allocatable :: gcol_to_rank(:)
    integer, allocatable :: send_gcol_list(:)
    character(len=*), parameter :: subname = 'rtm_horiz_remap_init'

    if (masterproc) then
      write(iulog,*) trim(subname), ': Reading mapping file for tape ', t, ': ', trim(mapfile)
    end if

    ! --- Phase 1: Read the mapping file using shared infrastructure ---
    pio_subsystem => shr_pio_getiosys(inst_name)

    call shr_horiz_remap_read_mapfile(remap_data(t)%shared, mapfile, mpicom_rof, iam, npes, &
         pio_subsystem, ierr)
    if (ierr /= 0) then
      call shr_sys_abort(trim(subname)//': Error reading mapping file')
    end if

    n_a = remap_data(t)%shared%n_a

    if (masterproc) then
      write(iulog,*) trim(subname), ': n_a=', n_a, &
           ' n_b=', remap_data(t)%shared%n_b, &
           ' target grid nlat=', remap_data(t)%shared%nlat, &
           ' nlon=', remap_data(t)%shared%nlon
    end if

    ! --- Build gcol_to_rank(1:n_a) using MOSART's decomposition ---
    ! MOSART cells are indexed begr:endr locally. The global cell index
    ! is rtmCTL%gindex(r). numr is the total global cell count.
    begr = rtmCTL%begr
    endr = rtmCTL%endr
    numr = rtmCTL%numr
    n_my_cells = endr - begr + 1

    ! Build list of global cell IDs owned by this rank
    allocate(my_gcol_list(n_my_cells))
    cnt = 0
    do i = begr, endr
      cnt = cnt + 1
      my_gcol_list(cnt) = rtmCTL%gindex(i)
    end do

    ! Build reverse map: global gcol -> local index in my_gcol_list (for owned cells)
    allocate(gcol_to_myidx(n_a))
    gcol_to_myidx(:) = 0
    do i = 1, n_my_cells
      if (my_gcol_list(i) >= 1 .and. my_gcol_list(i) <= n_a) then
        gcol_to_myidx(my_gcol_list(i)) = i
      end if
    end do

    ! Build gcol_to_rank via MPI_Allgatherv of gindex segments
    block
      integer, allocatable :: all_begr(:), all_endr(:)
      integer, allocatable :: all_gindex(:)
      integer, allocatable :: all_counts(:), all_displs(:)
      integer :: r_rank

      allocate(gcol_to_rank(n_a))
      gcol_to_rank(:) = -1

      allocate(all_begr(0:npes-1), all_endr(0:npes-1))
      allocate(all_counts(0:npes-1), all_displs(0:npes-1))

      call mpi_allgather(begr, 1, MPI_INTEGER, all_begr, 1, MPI_INTEGER, mpicom_rof, ierr)
      call mpi_allgather(endr, 1, MPI_INTEGER, all_endr, 1, MPI_INTEGER, mpicom_rof, ierr)

      do r_rank = 0, npes-1
        all_counts(r_rank) = max(0, all_endr(r_rank) - all_begr(r_rank) + 1)
        if (r_rank == 0) then
          all_displs(r_rank) = 0
        else
          all_displs(r_rank) = all_displs(r_rank-1) + all_counts(r_rank-1)
        end if
      end do

      allocate(all_gindex(sum(all_counts)))
      call mpi_allgatherv(rtmCTL%gindex(begr), n_my_cells, MPI_INTEGER, &
                          all_gindex, all_counts, all_displs, MPI_INTEGER, &
                          mpicom_rof, ierr)

      do r_rank = 0, npes-1
        do i = 1, all_counts(r_rank)
          gcol = all_gindex(all_displs(r_rank) + i)
          if (gcol >= 1 .and. gcol <= n_a) then
            gcol_to_rank(gcol) = r_rank
          end if
        end do
      end do

      deallocate(all_begr, all_endr, all_gindex, all_counts, all_displs)
    end block

    ! --- Phase 2: Build MPI communication pattern using shared infrastructure ---
    call shr_horiz_remap_build_comm(remap_data(t)%shared, gcol_to_rank, mpicom_rof, iam, npes, &
         send_gcol_list, ierr)
    if (ierr /= 0) then
      call shr_sys_abort(trim(subname)//': Error building communication pattern')
    end if

    deallocate(gcol_to_rank)

    ! --- Convert send_gcol_list to local cell indices ---
    allocate(remap_data(t)%send_cell_local(remap_data(t)%shared%n_send_total))
    do i = 1, remap_data(t)%shared%n_send_total
      idx = gcol_to_myidx(send_gcol_list(i))
      if (idx == 0) then
        call shr_sys_abort(trim(subname)//': Requested cell not owned by this rank')
      end if
      remap_data(t)%send_cell_local(i) = idx
    end do

    deallocate(send_gcol_list, my_gcol_list, gcol_to_myidx)

    if (masterproc) then
      write(iulog,*) trim(subname), ': Horizontal remapping initialized for tape ', t
      write(iulog,*) trim(subname), ':   n_b_local=', remap_data(t)%shared%n_b_local
    end if

  end subroutine rtm_horiz_remap_init

  !-------------------------------------------------------------------------------------------
  subroutine rtm_horiz_remap_field(t, field_in, fld_out)
    !
    ! Remap a field from the MOSART cell decomposition to the target lat-lon grid.
    !
    ! Input: field_in(begr:endr) - field on MOSART cells
    ! Output: fld_out(n_b_local) - remapped field (allocated here)
    !
    integer,  intent(in)    :: t        ! tape index
    real(r8), intent(in)    :: field_in(:)
    real(r8), allocatable, intent(out) :: fld_out(:)

    ! Local variables
    type(rtm_horiz_remap_t), pointer :: rd
    real(r8), allocatable :: send_buf(:)
    real(r8), allocatable :: fld_out_2d(:,:)
    integer :: i, ierr
    character(len=*), parameter :: subname = 'rtm_horiz_remap_field'

    rd => remap_data(t)

    if (.not. rd%shared%initialized) then
      call shr_sys_abort(trim(subname)//': Remapping not initialized for this tape')
    end if

    ! Pack send buffer from MOSART cell data using local indices
    allocate(send_buf(rd%shared%n_send_total))
    do i = 1, rd%shared%n_send_total
      send_buf(i) = field_in(rd%send_cell_local(i))
    end do

    ! Apply remapping via shared infrastructure (numlev=1 for MOSART)
    allocate(fld_out_2d(rd%shared%n_b_local, 1))
    call shr_horiz_remap_apply(rd%shared, send_buf, 1, fld_out_2d, mpicom_rof, npes, ierr)

    ! Copy to 1D output
    allocate(fld_out(rd%shared%n_b_local))
    fld_out(:) = fld_out_2d(:, 1)

    deallocate(send_buf, fld_out_2d)

  end subroutine rtm_horiz_remap_field

  !-------------------------------------------------------------------------------------------
  subroutine rtm_horiz_remap_write(t, File, varid, fld_out, data_type)
    !
    ! Write a remapped field to a PIO file using the target lat-lon decomposition.
    ! MOSART fields are always 2D (lon, lat) since there are no vertical levels.
    !
    use pio,              only: file_desc_t, var_desc_t, io_desc_t, &
                                pio_initdecomp, pio_freedecomp, &
                                pio_write_darray, iosystem_desc_t, &
                                PIO_OFFSET_KIND
    use shr_pio_mod,      only: shr_pio_getiosys
    use RtmVar,           only: inst_name

    ! Arguments
    integer,           intent(in)    :: t         ! tape index
    type(file_desc_t), intent(inout) :: File
    type(var_desc_t),  intent(inout) :: varid
    real(r8),          intent(in)    :: fld_out(:)  ! (n_b_local)
    integer,           intent(in)    :: data_type

    ! Local variables
    type(rtm_horiz_remap_t), pointer :: rd
    type(io_desc_t), save :: iodesc_2d_cache(max_tapes)
    type(iosystem_desc_t), pointer :: pio_subsystem
    integer(PIO_OFFSET_KIND), allocatable :: idof(:)
    integer :: i, global_row, ilat, ilon, ierr
    integer :: nlat, nlon
    character(len=*), parameter :: subname = 'rtm_horiz_remap_write'

    rd => remap_data(t)
    nlat = rd%shared%nlat
    nlon = rd%shared%nlon

    pio_subsystem => shr_pio_getiosys(inst_name)

    ! 2D field: use cached decomposition
    if (.not. rd%iodesc_2d_valid) then
      allocate(idof(rd%shared%n_b_local))
      do i = 1, rd%shared%n_b_local
        global_row = rd%shared%row_start + i - 1
        ilon = mod(global_row - 1, nlon) + 1
        ilat = (global_row - 1) / nlon + 1
        idof(i) = int(ilon + nlon * (ilat - 1), PIO_OFFSET_KIND)
      end do
      call pio_initdecomp(pio_subsystem, data_type, (/nlon, nlat/), idof, iodesc_2d_cache(t))
      deallocate(idof)
      rd%iodesc_2d_valid = .true.
    end if
    call pio_write_darray(File, varid, iodesc_2d_cache(t), fld_out, ierr)

  end subroutine rtm_horiz_remap_write

end module RtmHorizRemap
