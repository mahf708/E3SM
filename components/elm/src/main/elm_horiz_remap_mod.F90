module elm_horiz_remap_mod
  !-------------------------------------------------------------------------------------------
  !
  ! Module for online horizontal remapping of ELM history output fields.
  ! Reads a precomputed ESMF/TempestRemap mapping file (standard NetCDF format
  ! with n_s, n_a, n_b dimensions and row, col, S variables) and applies the
  ! sparse matrix-vector multiply to remap fields from the ELM grid to a target
  ! lat-lon grid (e.g., 180x360) before writing history output.
  !
  ! Ported from EAM's horiz_remap_mod.F90, adapted for ELM's 1D gridcell
  ! decomposition (no chunks).
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,  only: r8 => shr_kind_r8
  use elm_varctl,    only: iulog
  use abortutils,    only: endrun
  use spmdMod,       only: masterproc, iam, npes, mpicom, &
                            MPI_INTEGER, MPI_REAL8
  use decompMod,     only: get_proc_bounds, bounds_type, ldecomp, &
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
    logical  :: initialized = .false.
    integer  :: n_a = 0          ! source grid size
    integer  :: n_b = 0          ! target grid size
    integer  :: nlat = 0         ! target latitude dimension
    integer  :: nlon = 0         ! target longitude dimension
    integer  :: n_b_local = 0    ! number of target points owned by this rank
    integer  :: row_start = 0    ! global starting row (1-indexed) for this rank

    ! Local sparse matrix (triplet form, only rows owned by this rank)
    integer  :: nnz_local = 0    ! number of local nonzeros
    integer,  allocatable :: dst_local(:)  ! local target index (1..n_b_local)
    integer,  allocatable :: src_gid(:)    ! global source column ID (1-indexed)
    real(r8), allocatable :: wgt(:)        ! interpolation weight

    ! Source data gathering: src_need_gids is sorted for binary search
    integer  :: n_src_need = 0
    integer,  allocatable :: src_need_gids(:)    ! sorted unique global IDs of needed src gridcells
    integer,  allocatable :: src_need_recvidx(:) ! recv buffer index for each src_need_gids entry

    ! MPI communication pattern for Alltoallv
    integer,  allocatable :: send_counts(:)  ! npes
    integer,  allocatable :: send_displs(:)  ! npes
    integer,  allocatable :: recv_counts(:)  ! npes
    integer,  allocatable :: recv_displs(:)  ! npes
    integer,  allocatable :: send_gridcell(:)  ! local gridcell index for each send entry
    integer  :: n_send_total = 0
    integer  :: n_recv_total = 0

    ! Target grid coordinates
    real(r8), allocatable :: lat(:)   ! nlat
    real(r8), allocatable :: lon(:)   ! nlon

    ! Cached PIO decompositions (avoid repeated init/free per field)
    logical  :: iodesc_2d_valid = .false.
    logical  :: iodesc_3d_valid = .false.
    integer  :: iodesc_3d_nlev = 0
  end type elm_horiz_remap_t

  type(elm_horiz_remap_t), target :: remap_data(max_tapes)

CONTAINS

  !-------------------------------------------------------------------------------------------
  pure integer function bsearch(arr, n, val)
    ! Binary search for val in sorted array arr(1:n). Returns index or 0.
    integer, intent(in) :: n, val
    integer, intent(in) :: arr(n)
    integer :: lo, hi, mid
    lo = 1; hi = n
    bsearch = 0
    do while (lo <= hi)
      mid = (lo + hi) / 2
      if (arr(mid) == val) then
        bsearch = mid
        return
      else if (arr(mid) < val) then
        lo = mid + 1
      else
        hi = mid - 1
      end if
    end do
  end function bsearch

  !-------------------------------------------------------------------------------------------
  pure integer function src_gid_to_recvidx(rd, gid)
    ! Map a global source gridcell ID to its position in the recv buffer,
    ! using binary search on the sorted src_need_gids array.
    type(elm_horiz_remap_t), intent(in) :: rd
    integer, intent(in) :: gid
    integer :: idx
    idx = bsearch(rd%src_need_gids, rd%n_src_need, gid)
    src_gid_to_recvidx = rd%src_need_recvidx(idx)
  end function src_gid_to_recvidx

  !-------------------------------------------------------------------------------------------
  logical function elm_horiz_remap_is_active(t)
    integer, intent(in) :: t
    elm_horiz_remap_is_active = remap_data(t)%initialized
  end function elm_horiz_remap_is_active

  !-------------------------------------------------------------------------------------------
  subroutine elm_horiz_remap_init(t, mapfile)
    !
    ! Initialize horizontal remapping for history tape t by reading a
    ! mapping file and building the MPI communication pattern.
    !
    ! ELM gridcells are numbered 1..numg globally. Each rank owns
    ! gridcells begg:endg. The global gridcell index is obtained from
    ! ldecomp%gdc2glo(g).
    !
    use pio,            only: file_desc_t, pio_closefile, &
                              pio_nowrite, pio_inq_dimid, pio_inq_dimlen, &
                              pio_inq_varid, pio_get_var, var_desc_t, &
                              pio_noerr, pio_int, pio_double, &
                              PIO_OFFSET_KIND, &
                              pio_seterrorhandling, pio_bcast_error, &
                              pio_internal_error
    use ncdio_pio,      only: ncd_pio_openfile

    ! Arguments
    integer,          intent(in) :: t        ! tape index
    character(len=*), intent(in) :: mapfile  ! path to ESMF mapping file

    ! Local variables
    type(file_desc_t) :: pioid
    type(var_desc_t)  :: vid
    integer           :: ierr, dimid
    integer           :: n_s, n_a, n_b
    integer           :: dst_grid_rank
    integer, allocatable :: dst_grid_dims(:)
    integer, allocatable :: row_all(:), col_all(:)
    real(r8), allocatable :: S_all(:)
    real(r8), allocatable :: xc_b(:), yc_b(:)
    integer           :: i, gcol, owner_rank
    integer           :: row_start, row_end, nlat, nlon
    integer           :: cnt, n_unique
    integer, allocatable :: need_from_rank(:)
    integer, allocatable :: recv_gcols(:)
    integer, allocatable :: my_requests(:)
    integer             :: g, n_my_cols
    integer, allocatable :: my_gcol_list(:)   ! global gridcell IDs owned by this rank
    integer, allocatable :: gcol_to_myidx(:)  ! global gcol -> index in my_gcol_list
    integer             :: r, idx
    type(bounds_type)   :: bounds
    integer             :: numg
    character(len=*), parameter :: subname = 'elm_horiz_remap_init'

    if (masterproc) then
      write(iulog,*) trim(subname), ': Reading mapping file for tape ', t, ': ', trim(mapfile)
    end if

    ! Open the mapping file using PIO
    call ncd_pio_openfile(pioid, trim(mapfile), pio_nowrite)

    ! Read dimensions
    ierr = pio_inq_dimid(pioid, 'n_s', dimid)
    ierr = pio_inq_dimlen(pioid, dimid, n_s)
    ierr = pio_inq_dimid(pioid, 'n_a', dimid)
    ierr = pio_inq_dimlen(pioid, dimid, n_a)
    ierr = pio_inq_dimid(pioid, 'n_b', dimid)
    ierr = pio_inq_dimlen(pioid, dimid, n_b)

    remap_data(t)%n_a = n_a
    remap_data(t)%n_b = n_b

    ! Read dst_grid_rank and dst_grid_dims to determine nlat/nlon
    call pio_seterrorhandling(pioid, pio_bcast_error)
    ierr = pio_inq_dimid(pioid, 'dst_grid_rank', dimid)
    call pio_seterrorhandling(pioid, pio_internal_error)
    if (ierr /= pio_noerr) then
      call endrun(trim(subname)//': Map file missing dst_grid_rank dimension. '// &
           'Cannot determine target grid lat/lon dimensions.')
    end if
    ierr = pio_inq_dimlen(pioid, dimid, dst_grid_rank)
    if (dst_grid_rank /= 2) then
      call endrun(trim(subname)//': dst_grid_rank must be 2 for lat-lon target grid')
    end if
    allocate(dst_grid_dims(dst_grid_rank))
    ierr = pio_inq_varid(pioid, 'dst_grid_dims', vid)
    ierr = pio_get_var(pioid, vid, dst_grid_dims)
    nlon = dst_grid_dims(1)
    nlat = dst_grid_dims(2)
    deallocate(dst_grid_dims)

    if (nlat * nlon /= n_b) then
      call endrun(trim(subname)//': nlat*nlon does not equal n_b in mapping file')
    end if
    remap_data(t)%nlat = nlat
    remap_data(t)%nlon = nlon

    if (masterproc) then
      write(iulog,*) trim(subname), ': n_a=', n_a, ' n_b=', n_b, ' n_s=', n_s
      write(iulog,*) trim(subname), ': target grid nlat=', nlat, ' nlon=', nlon
    end if

    ! Read the full sparse matrix (all ranks read all -- simple approach).
    if (masterproc .and. n_s > 10000000) then
      write(iulog,*) 'WARNING: ', trim(subname), ': Large sparse matrix n_s=', n_s
      write(iulog,*) '  All ranks read full matrix. Memory per rank: ~', &
           n_s * 20 / 1000000, ' MB'
    end if
    allocate(row_all(n_s), col_all(n_s), S_all(n_s))
    ierr = pio_inq_varid(pioid, 'row', vid)
    ierr = pio_get_var(pioid, vid, row_all)
    ierr = pio_inq_varid(pioid, 'col', vid)
    ierr = pio_get_var(pioid, vid, col_all)
    ierr = pio_inq_varid(pioid, 'S', vid)
    ierr = pio_get_var(pioid, vid, S_all)

    ! Read target grid coordinates
    allocate(xc_b(n_b), yc_b(n_b))
    ierr = pio_inq_varid(pioid, 'xc_b', vid)
    ierr = pio_get_var(pioid, vid, xc_b)
    ierr = pio_inq_varid(pioid, 'yc_b', vid)
    ierr = pio_get_var(pioid, vid, yc_b)

    call pio_closefile(pioid)

    ! Extract unique lat/lon arrays from the target grid coordinates.
    ! For a lat-lon grid with n_b = nlon*nlat, points are ordered
    ! lon-major: point(i) has ilon = mod(i-1, nlon)+1, ilat = (i-1)/nlon + 1
    allocate(remap_data(t)%lon(nlon), remap_data(t)%lat(nlat))
    do i = 1, nlon
      remap_data(t)%lon(i) = xc_b(i)
    end do
    do i = 1, nlat
      remap_data(t)%lat(i) = yc_b((i-1)*nlon + 1)
    end do
    deallocate(xc_b, yc_b)

    ! Partition target grid rows across MPI ranks (contiguous blocks)
    row_start = iam * n_b / npes + 1
    row_end   = (iam + 1) * n_b / npes
    remap_data(t)%n_b_local = max(0, row_end - row_start + 1)
    remap_data(t)%row_start = row_start

    ! Extract local sparse matrix entries (rows belonging to this rank)
    cnt = 0
    do i = 1, n_s
      if (row_all(i) >= row_start .and. row_all(i) <= row_end) then
        cnt = cnt + 1
      end if
    end do
    remap_data(t)%nnz_local = cnt

    allocate(remap_data(t)%dst_local(cnt))
    allocate(remap_data(t)%src_gid(cnt))
    allocate(remap_data(t)%wgt(cnt))

    cnt = 0
    do i = 1, n_s
      if (row_all(i) >= row_start .and. row_all(i) <= row_end) then
        cnt = cnt + 1
        remap_data(t)%dst_local(cnt) = row_all(i) - row_start + 1
        remap_data(t)%src_gid(cnt) = col_all(i)
        remap_data(t)%wgt(cnt) = S_all(i)
      end if
    end do

    deallocate(row_all, col_all, S_all)

    ! Find unique source gridcells needed by this rank.
    allocate(gcol_to_myidx(n_a))
    gcol_to_myidx(:) = 0

    ! Mark which source gridcells are needed
    do i = 1, remap_data(t)%nnz_local
      gcol_to_myidx(remap_data(t)%src_gid(i)) = 1
    end do

    ! Count unique
    n_unique = 0
    do i = 1, n_a
      if (gcol_to_myidx(i) > 0) then
        n_unique = n_unique + 1
      end if
    end do
    remap_data(t)%n_src_need = n_unique

    ! Build sorted list of unique source gridcell IDs
    allocate(remap_data(t)%src_need_gids(n_unique))
    allocate(remap_data(t)%src_need_recvidx(n_unique))
    cnt = 0
    do i = 1, n_a
      if (gcol_to_myidx(i) > 0) then
        cnt = cnt + 1
        remap_data(t)%src_need_gids(cnt) = i
      end if
    end do
    deallocate(gcol_to_myidx)

    ! Build the communication pattern.
    ! In ELM, gridcells are numbered 1..numg. Each rank owns begg:endg.
    ! The global gridcell index is ldecomp%gdc2glo(g).
    ! We need a reverse mapping: for any global gridcell ID, find which rank owns it.
    !
    ! Step 1: Build list of all global gridcell IDs owned by this rank
    call get_proc_bounds(bounds)
    call get_proc_global(numg=numg)
    n_my_cols = bounds%endg - bounds%begg + 1

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

    ! Step 2: For each source gridcell we need, find which rank owns it.
    ! Since ELM doesn't have a chunk ownership structure like EAM, we need
    ! each rank to know which global gridcells are owned by which rank.
    ! We use MPI_Allgather of the begg/endg ranges and gdc2glo mapping.
    ! For simplicity, gather the (glo_min, glo_max) range per rank and use
    ! the full gdc2glo array which is contiguous.
    !
    ! In practice, ELM's decomposition assigns contiguous local gridcell
    ! indices, and ldecomp%gdc2glo maps local to global. We build a global
    ! array that maps global gridcell ID -> owning rank.
    block
      integer, allocatable :: gcol_owner(:)  ! n_a: global gridcell -> owning rank
      integer, allocatable :: all_begg(:), all_endg(:)
      integer, allocatable :: all_gdc2glo(:)  ! gathered gdc2glo segments
      integer, allocatable :: all_counts(:), all_displs(:)
      integer :: r_rank

      allocate(gcol_owner(n_a))
      gcol_owner(:) = -1

      ! Gather begg/endg from all ranks
      allocate(all_begg(0:npes-1), all_endg(0:npes-1))
      allocate(all_counts(0:npes-1), all_displs(0:npes-1))

      call mpi_allgather(bounds%begg, 1, MPI_INTEGER, all_begg, 1, MPI_INTEGER, mpicom, ierr)
      call mpi_allgather(bounds%endg, 1, MPI_INTEGER, all_endg, 1, MPI_INTEGER, mpicom, ierr)

      ! Gather all gdc2glo mappings
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

      ! Build the owner map
      do r_rank = 0, npes-1
        do i = 1, all_counts(r_rank)
          gcol = all_gdc2glo(all_displs(r_rank) + i)
          if (gcol >= 1 .and. gcol <= n_a) then
            gcol_owner(gcol) = r_rank
          end if
        end do
      end do

      deallocate(all_begg, all_endg, all_gdc2glo, all_counts, all_displs)

      ! Now determine how many columns we need from each rank
      allocate(need_from_rank(0:npes-1))
      need_from_rank = 0
      do i = 1, n_unique
        gcol = remap_data(t)%src_need_gids(i)
        owner_rank = gcol_owner(gcol)
        if (owner_rank < 0) then
          call endrun(trim(subname)//': Source gridcell has no owner in ELM decomposition')
        end if
        need_from_rank(owner_rank) = need_from_rank(owner_rank) + 1
      end do

      ! recv_counts = how many gridcells I need from each rank
      allocate(remap_data(t)%recv_counts(0:npes-1))
      allocate(remap_data(t)%recv_displs(0:npes-1))
      remap_data(t)%recv_counts = need_from_rank

      remap_data(t)%recv_displs(0) = 0
      do r = 1, npes-1
        remap_data(t)%recv_displs(r) = remap_data(t)%recv_displs(r-1) + remap_data(t)%recv_counts(r-1)
      end do
      remap_data(t)%n_recv_total = sum(remap_data(t)%recv_counts)

      ! Build ordered list of gcols to receive (grouped by source rank)
      allocate(recv_gcols(remap_data(t)%n_recv_total))
      deallocate(need_from_rank)
      allocate(need_from_rank(0:npes-1))
      need_from_rank = remap_data(t)%recv_displs
      do i = 1, n_unique
        gcol = remap_data(t)%src_need_gids(i)
        owner_rank = gcol_owner(gcol)
        need_from_rank(owner_rank) = need_from_rank(owner_rank) + 1
        recv_gcols(need_from_rank(owner_rank)) = gcol
      end do

      ! Build src_need_recvidx: for each entry in src_need_gids, store its
      ! position in the recv buffer.
      do i = 1, n_unique
        gcol = remap_data(t)%src_need_gids(i)
        do cnt = 1, remap_data(t)%n_recv_total
          if (recv_gcols(cnt) == gcol) then
            remap_data(t)%src_need_recvidx(i) = cnt
            exit
          end if
        end do
      end do
      deallocate(need_from_rank)

      ! Step 3: Exchange recv_counts so each rank knows what to send
      allocate(remap_data(t)%send_counts(0:npes-1))
      allocate(remap_data(t)%send_displs(0:npes-1))

      call mpi_alltoall(remap_data(t)%recv_counts, 1, MPI_INTEGER, &
                        remap_data(t)%send_counts, 1, MPI_INTEGER, &
                        mpicom, ierr)

      remap_data(t)%send_displs(0) = 0
      do r = 1, npes-1
        remap_data(t)%send_displs(r) = remap_data(t)%send_displs(r-1) + remap_data(t)%send_counts(r-1)
      end do
      remap_data(t)%n_send_total = sum(remap_data(t)%send_counts)

      ! Step 4: Exchange the actual gcol IDs so senders know what to send
      allocate(remap_data(t)%send_gridcell(remap_data(t)%n_send_total))
      allocate(my_requests(remap_data(t)%n_send_total))

      call mpi_alltoallv(recv_gcols, remap_data(t)%recv_counts, remap_data(t)%recv_displs, MPI_INTEGER, &
                         my_requests, remap_data(t)%send_counts, remap_data(t)%send_displs, MPI_INTEGER, &
                         mpicom, ierr)

      deallocate(recv_gcols)

      ! Convert requested global gcols to local gridcell indices
      do i = 1, remap_data(t)%n_send_total
        idx = gcol_to_myidx(my_requests(i))
        if (idx == 0) then
          call endrun(trim(subname)//': Requested gridcell not owned by this rank')
        end if
        ! Store the local gridcell index (offset from begg)
        remap_data(t)%send_gridcell(i) = idx
      end do

      deallocate(my_requests, gcol_owner)
    end block

    deallocate(my_gcol_list, gcol_to_myidx)

    remap_data(t)%initialized = .true.

    if (masterproc) then
      write(iulog,*) trim(subname), ': Horizontal remapping initialized for tape ', t
      write(iulog,*) trim(subname), ':   n_b_local=', remap_data(t)%n_b_local
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
    real(r8), allocatable :: send_buf(:)  ! packed send data (n_send_total * numlev)
    real(r8), allocatable :: recv_buf(:)  ! received source data (n_recv_total * numlev)
    integer :: i, k, src_local, dst_local
    integer :: ierr
    integer, allocatable :: send_counts_lev(:), send_displs_lev(:)
    integer, allocatable :: recv_counts_lev(:), recv_displs_lev(:)
    character(len=*), parameter :: subname = 'elm_horiz_remap_field'

    rd => remap_data(t)

    if (.not. rd%initialized) then
      call endrun(trim(subname)//': Remapping not initialized for this tape')
    end if

    ! Allocate output
    allocate(fld_out(rd%n_b_local, numlev))
    fld_out = 0.0_r8

    ! Pack send buffer: gather local gridcell data requested by other ranks
    allocate(send_buf(rd%n_send_total * numlev))
    do i = 1, rd%n_send_total
      do k = 1, numlev
        send_buf((i-1)*numlev + k) = field_in(rd%send_gridcell(i), k)
      end do
    end do

    ! Receive buffer
    allocate(recv_buf(rd%n_recv_total * numlev))

    ! Alltoallv with level-scaled counts/displacements
    allocate(send_counts_lev(0:npes-1), send_displs_lev(0:npes-1))
    allocate(recv_counts_lev(0:npes-1), recv_displs_lev(0:npes-1))
    do i = 0, npes-1
      send_counts_lev(i) = rd%send_counts(i) * numlev
      send_displs_lev(i) = rd%send_displs(i) * numlev
      recv_counts_lev(i) = rd%recv_counts(i) * numlev
      recv_displs_lev(i) = rd%recv_displs(i) * numlev
    end do

    call mpi_alltoallv(send_buf, send_counts_lev, send_displs_lev, MPI_REAL8, &
                       recv_buf, recv_counts_lev, recv_displs_lev, MPI_REAL8, &
                       mpicom, ierr)

    deallocate(send_buf, send_counts_lev, send_displs_lev, recv_counts_lev, recv_displs_lev)

    ! Apply sparse matrix-vector multiply
    do i = 1, rd%nnz_local
      dst_local = rd%dst_local(i)
      src_local = src_gid_to_recvidx(rd, rd%src_gid(i))
      do k = 1, numlev
        fld_out(dst_local, k) = fld_out(dst_local, k) + &
             rd%wgt(i) * recv_buf((src_local-1)*numlev + k)
      end do
    end do

    deallocate(recv_buf)

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
    nlat = rd%nlat
    nlon = rd%nlon

    pio_subsystem => shr_pio_getiosys(inst_name)

    if (numlev <= 1) then
      ! 2D field: use cached decomposition
      if (.not. rd%iodesc_2d_valid) then
        allocate(idof(rd%n_b_local))
        do i = 1, rd%n_b_local
          global_row = rd%row_start + i - 1
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
        allocate(idof(rd%n_b_local * numlev))
        do i = 1, rd%n_b_local
          global_row = rd%row_start + i - 1
          ilon = mod(global_row - 1, nlon) + 1
          ilat = (global_row - 1) / nlon + 1
          do k = 1, numlev
            idof((k-1)*rd%n_b_local + i) = int(ilon + nlon*(ilat-1) + nlon*nlat*(k-1), PIO_OFFSET_KIND)
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
