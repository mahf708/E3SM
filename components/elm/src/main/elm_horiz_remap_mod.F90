module elm_horiz_remap_mod
  !-------------------------------------------------------------------------------------------
  !
  ! ELM wrapper for shared horizontal remapping infrastructure.
  !
  ! Uses shr_horiz_remap_mod for the core algorithm (reading map files,
  ! building MPI communication patterns, and applying sparse matrix-vector
  ! multiply). This module adds ELM-specific logic: decomposition mapping
  ! via ldecomp%gdc2glo, local gridcell index translation, and PIO output
  ! with cached decompositions.
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,        only: r8 => shr_kind_r8
  use shr_horiz_remap_mod, only: shr_horiz_remap_t, shr_horiz_remap_read_mapfile, &
                                  shr_horiz_remap_build_comm, shr_horiz_remap_apply
  use elm_varctl,          only: iulog
  use abortutils,          only: endrun
  use spmdMod,             only: masterproc, iam, npes, mpicom, &
                                  MPI_INTEGER, MPI_REAL8
  use decompMod,           only: get_proc_bounds, bounds_type, ldecomp, &
                                  get_proc_global

  implicit none
  private
  save

  integer, parameter :: max_tapes = 6

  public :: elm_horiz_remap_init
  public :: elm_horiz_remap_field
  public :: elm_horiz_remap_write
  public :: elm_horiz_remap_is_active

  type elm_horiz_remap_t
    type(shr_horiz_remap_t) :: shared
    integer, allocatable    :: send_gridcell(:)  ! local gridcell index per send entry
    logical  :: iodesc_2d_valid = .false.
    logical  :: iodesc_3d_valid = .false.
    integer  :: iodesc_3d_nlev = 0
  end type elm_horiz_remap_t

  type(elm_horiz_remap_t), target :: remap_data(max_tapes)

CONTAINS

  !-------------------------------------------------------------------------------------------
  logical function elm_horiz_remap_is_active(t)
    integer, intent(in) :: t
    elm_horiz_remap_is_active = remap_data(t)%shared%initialized
  end function elm_horiz_remap_is_active

  !-------------------------------------------------------------------------------------------
  subroutine elm_horiz_remap_init(t, mapfile)
    !
    ! Initialize horizontal remapping for history tape t by reading a
    ! mapping file and building the MPI communication pattern.
    !
    ! Uses the shared module for map file I/O and comm pattern building.
    ! ELM-specific: builds gcol_to_rank using ldecomp%gdc2glo + MPI_Allgatherv,
    ! then converts send_gcol_list to local gridcell indices.
    !
    use pio,            only: iosystem_desc_t
    use shr_pio_mod,    only: shr_pio_getiosys
    use elm_varctl,     only: inst_name

    ! Arguments
    integer,          intent(in) :: t        ! tape index
    character(len=*), intent(in) :: mapfile  ! path to ESMF mapping file

    ! Local variables
    type(iosystem_desc_t), pointer :: pio_subsystem
    integer           :: ierr
    integer           :: i, cnt, gcol, idx, g
    integer           :: n_my_cols, n_a
    type(bounds_type) :: bounds
    integer           :: numg
    integer, allocatable :: my_gcol_list(:)
    integer, allocatable :: gcol_to_myidx(:)
    integer, allocatable :: gcol_to_rank(:)
    integer, allocatable :: send_gcol_list(:)
    character(len=*), parameter :: subname = 'elm_horiz_remap_init'

    if (masterproc) then
      write(iulog,*) trim(subname), ': Reading mapping file for tape ', t, ': ', trim(mapfile)
    end if

    ! --- Phase 1: Read the mapping file using shared infrastructure ---
    pio_subsystem => shr_pio_getiosys(inst_name)

    call shr_horiz_remap_read_mapfile(remap_data(t)%shared, mapfile, mpicom, iam, npes, &
         pio_subsystem, ierr)
    if (ierr /= 0) then
      call endrun(trim(subname)//': Error reading mapping file, ierr='// &
           char(ichar('0') + ierr))
    end if

    n_a = remap_data(t)%shared%n_a

    if (masterproc) then
      write(iulog,*) trim(subname), ': n_a=', n_a, &
           ' n_b=', remap_data(t)%shared%n_b, &
           ' target grid nlat=', remap_data(t)%shared%nlat, &
           ' nlon=', remap_data(t)%shared%nlon
    end if

    ! --- Build gcol_to_rank(1:n_a) using ELM's decomposition ---
    ! ELM gridcells are numbered 1..numg. Each rank owns begg:endg.
    ! The global gridcell index is ldecomp%gdc2glo(g).
    call get_proc_bounds(bounds)
    call get_proc_global(numg=numg)
    n_my_cols = bounds%endg - bounds%begg + 1

    ! Build list of global gridcell IDs owned by this rank
    allocate(my_gcol_list(n_my_cols))
    cnt = 0
    do g = bounds%begg, bounds%endg
      cnt = cnt + 1
      my_gcol_list(cnt) = ldecomp%gdc2glo(g)
    end do

    ! Build reverse map: global gcol -> local index in my_gcol_list (for owned gridcells)
    allocate(gcol_to_myidx(n_a))
    gcol_to_myidx(:) = 0
    do i = 1, n_my_cols
      gcol_to_myidx(my_gcol_list(i)) = i
    end do

    ! Build gcol_to_rank via MPI_Allgatherv of gdc2glo segments
    block
      integer, allocatable :: all_begg(:), all_endg(:)
      integer, allocatable :: all_gdc2glo(:)
      integer, allocatable :: all_counts(:), all_displs(:)
      integer :: r_rank

      allocate(gcol_to_rank(n_a))
      gcol_to_rank(:) = -1

      allocate(all_begg(0:npes-1), all_endg(0:npes-1))
      allocate(all_counts(0:npes-1), all_displs(0:npes-1))

      call mpi_allgather(bounds%begg, 1, MPI_INTEGER, all_begg, 1, MPI_INTEGER, mpicom, ierr)
      call mpi_allgather(bounds%endg, 1, MPI_INTEGER, all_endg, 1, MPI_INTEGER, mpicom, ierr)

      do r_rank = 0, npes-1
        all_counts(r_rank) = max(0, all_endg(r_rank) - all_begg(r_rank) + 1)
        if (r_rank == 0) then
          all_displs(r_rank) = 0
        else
          all_displs(r_rank) = all_displs(r_rank-1) + all_counts(r_rank-1)
        end if
      end do

      allocate(all_gdc2glo(sum(all_counts)))
      call mpi_allgatherv(ldecomp%gdc2glo(bounds%begg), n_my_cols, MPI_INTEGER, &
                          all_gdc2glo, all_counts, all_displs, MPI_INTEGER, &
                          mpicom, ierr)

      do r_rank = 0, npes-1
        do i = 1, all_counts(r_rank)
          gcol = all_gdc2glo(all_displs(r_rank) + i)
          if (gcol >= 1 .and. gcol <= n_a) then
            gcol_to_rank(gcol) = r_rank
          end if
        end do
      end do

      deallocate(all_begg, all_endg, all_gdc2glo, all_counts, all_displs)
    end block

    ! --- Phase 2: Build MPI communication pattern using shared infrastructure ---
    call shr_horiz_remap_build_comm(remap_data(t)%shared, gcol_to_rank, mpicom, iam, npes, &
         send_gcol_list, ierr)
    if (ierr /= 0) then
      call endrun(trim(subname)//': Error building communication pattern')
    end if

    deallocate(gcol_to_rank)

    ! --- Convert send_gcol_list to local gridcell indices ---
    allocate(remap_data(t)%send_gridcell(remap_data(t)%shared%n_send_total))
    do i = 1, remap_data(t)%shared%n_send_total
      idx = gcol_to_myidx(send_gcol_list(i))
      if (idx == 0) then
        call endrun(trim(subname)//': Requested gridcell not owned by this rank')
      end if
      remap_data(t)%send_gridcell(i) = idx
    end do

    deallocate(send_gcol_list, my_gcol_list, gcol_to_myidx)

    if (masterproc) then
      write(iulog,*) trim(subname), ': Horizontal remapping initialized for tape ', t
      write(iulog,*) trim(subname), ':   n_b_local=', remap_data(t)%shared%n_b_local
    end if

  end subroutine elm_horiz_remap_init

  !-------------------------------------------------------------------------------------------
  subroutine elm_horiz_remap_field(t, field_in, numlev, fld_out)
    !
    ! Remap a field from the ELM gridcell decomposition to the target lat-lon grid.
    !
    ! Input: field_in(begg:endg, numlev) - field on ELM gridcells
    ! Output: fld_out(n_b_local, numlev) - remapped field (allocated here)
    !
    integer,  intent(in)    :: t        ! tape index
    real(r8), intent(in)    :: field_in(:,:)
    integer,  intent(in)    :: numlev
    real(r8), allocatable, intent(out) :: fld_out(:,:)

    ! Local variables
    type(elm_horiz_remap_t), pointer :: rd
    real(r8), allocatable :: send_buf(:)
    integer :: i, k, ierr
    character(len=*), parameter :: subname = 'elm_horiz_remap_field'

    rd => remap_data(t)

    if (.not. rd%shared%initialized) then
      call endrun(trim(subname)//': Remapping not initialized for this tape')
    end if

    ! Allocate output
    allocate(fld_out(rd%shared%n_b_local, numlev))

    ! Pack send buffer from ELM gridcell data using local indices
    allocate(send_buf(rd%shared%n_send_total * numlev))
    do i = 1, rd%shared%n_send_total
      do k = 1, numlev
        send_buf((i-1)*numlev + k) = field_in(rd%send_gridcell(i), k)
      end do
    end do

    ! Apply remapping via shared infrastructure
    call shr_horiz_remap_apply(rd%shared, send_buf, numlev, fld_out, mpicom, npes, ierr)

    deallocate(send_buf)

  end subroutine elm_horiz_remap_field

  !-------------------------------------------------------------------------------------------
  subroutine elm_horiz_remap_write(t, File, varid, fld_out, numlev, data_type)
    !
    ! Write a remapped field to a PIO file using the target lat-lon decomposition.
    !
    use pio,              only: file_desc_t, var_desc_t, io_desc_t, &
                                pio_initdecomp, pio_freedecomp, &
                                pio_write_darray, iosystem_desc_t, &
                                PIO_OFFSET_KIND
    use shr_pio_mod,      only: shr_pio_getiosys
    use elm_varctl,       only: inst_name

    ! Arguments
    integer,           intent(in)    :: t         ! tape index
    type(file_desc_t), intent(inout) :: File
    type(var_desc_t),  intent(inout) :: varid
    real(r8),          intent(in)    :: fld_out(:,:)  ! (n_b_local, numlev)
    integer,           intent(in)    :: numlev
    integer,           intent(in)    :: data_type

    ! Local variables
    type(elm_horiz_remap_t), pointer :: rd
    type(io_desc_t), save :: iodesc_2d_cache(max_tapes)
    type(io_desc_t), save :: iodesc_3d_cache(max_tapes)
    type(iosystem_desc_t), pointer :: pio_subsystem
    integer(PIO_OFFSET_KIND), allocatable :: idof(:)
    integer :: i, k, global_row, ilat, ilon, ierr
    integer :: nlat, nlon
    character(len=*), parameter :: subname = 'elm_horiz_remap_write'

    rd => remap_data(t)
    nlat = rd%shared%nlat
    nlon = rd%shared%nlon

    pio_subsystem => shr_pio_getiosys(inst_name)

    if (numlev <= 1) then
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
      call pio_write_darray(File, varid, iodesc_2d_cache(t), fld_out(:,1), ierr)
    else
      ! 3D field: cache decomposition, rebuild if numlev changes
      if (.not. rd%iodesc_3d_valid .or. rd%iodesc_3d_nlev /= numlev) then
        if (rd%iodesc_3d_valid) then
          call pio_freedecomp(File, iodesc_3d_cache(t))
        end if
        allocate(idof(rd%shared%n_b_local * numlev))
        do i = 1, rd%shared%n_b_local
          global_row = rd%shared%row_start + i - 1
          ilon = mod(global_row - 1, nlon) + 1
          ilat = (global_row - 1) / nlon + 1
          do k = 1, numlev
            idof((k-1)*rd%shared%n_b_local + i) = int(ilon + nlon*(ilat-1) + nlon*nlat*(k-1), PIO_OFFSET_KIND)
          end do
        end do
        call pio_initdecomp(pio_subsystem, data_type, (/nlon, nlat, numlev/), idof, iodesc_3d_cache(t))
        deallocate(idof)
        rd%iodesc_3d_valid = .true.
        rd%iodesc_3d_nlev = numlev
      end if
      call pio_write_darray(File, varid, iodesc_3d_cache(t), fld_out, ierr)
    end if

  end subroutine elm_horiz_remap_write

end module elm_horiz_remap_mod
