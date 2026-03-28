! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_horiz_remap
!
!> \brief MPAS sea-ice horizontal remapping for ACE training data
!> \author ACE team
!> \date   March 2026
!> \details
!>  This module provides online horizontal remapping of MPAS-Sea Ice fields
!>  from the unstructured mesh to a regular lat-lon grid via ESMF/TempestRemap
!>  mapping files. Output is written directly via PIO to separate NetCDF
!>  files with lat/lon dimensions, bypassing the MPAS stream manager.
!>
!>  Ported from EAM's horiz_remap_mod.F90, adapted for MPAS block
!>  decomposition and cell-based indexing.
!
!-----------------------------------------------------------------------

module seaice_horiz_remap

   use mpas_kind_types, only: RKIND, StrKIND
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_log, only: mpas_log_write
   use pio

   implicit none
   private
   save

   type, public :: seaice_horiz_remap_t
      logical :: initialized = .false.
      integer :: n_a = 0          ! source grid size (nCells global)
      integer :: n_b = 0          ! target grid size (nlat*nlon)
      integer :: nlat = 0
      integer :: nlon = 0
      integer :: n_b_local = 0    ! target points owned by this rank
      integer :: row_start = 0    ! global starting row for this rank

      ! Local sparse matrix (triplet form, only rows owned by this rank)
      integer :: nnz_local = 0
      integer, allocatable :: dst_local(:)   ! local target index (1..n_b_local)
      integer, allocatable :: src_gid(:)     ! global source cell ID (1-indexed)
      real(RKIND), allocatable :: wgt(:)     ! interpolation weight

      ! Source data gathering
      integer :: n_src_need = 0
      integer, allocatable :: src_need_gids(:)    ! sorted unique needed global IDs
      integer, allocatable :: src_need_recvidx(:) ! recv buffer index

      ! MPI communication pattern
      integer, allocatable :: send_counts(:), send_displs(:)
      integer, allocatable :: recv_counts(:), recv_displs(:)
      integer, allocatable :: send_cell_local(:) ! local cell index for each send
      integer :: n_send_total = 0
      integer :: n_recv_total = 0

      ! Target grid coordinates
      real(RKIND), allocatable :: lat(:), lon(:)
   end type seaice_horiz_remap_t

   public :: seaice_horiz_remap_init
   public :: seaice_horiz_remap_field_1d
   public :: seaice_horiz_remap_field_2d
   public :: seaice_horiz_remap_is_active

contains

!***********************************************************************
!
!  routine seaice_horiz_remap_is_active
!
!> \brief Check if horizontal remapping is active
!
!-----------------------------------------------------------------------
   logical function seaice_horiz_remap_is_active(rd)!{{{
      type(seaice_horiz_remap_t), intent(in) :: rd
      seaice_horiz_remap_is_active = rd%initialized
   end function seaice_horiz_remap_is_active!}}}

!***********************************************************************
!
!  routine bsearch
!
!> \brief Binary search in sorted array
!
!-----------------------------------------------------------------------
   pure integer function bsearch(arr, n, val)!{{{
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
   end function bsearch!}}}

!***********************************************************************
!
!  routine src_gid_to_recvidx
!
!> \brief Map global source cell ID to recv buffer position
!
!-----------------------------------------------------------------------
   pure integer function src_gid_to_recvidx(rd, gid)!{{{
      type(seaice_horiz_remap_t), intent(in) :: rd
      integer, intent(in) :: gid
      integer :: idx
      idx = bsearch(rd%src_need_gids, rd%n_src_need, gid)
      src_gid_to_recvidx = rd%src_need_recvidx(idx)
   end function src_gid_to_recvidx!}}}

!***********************************************************************
!
!  routine seaice_horiz_remap_init
!
!> \brief Initialize horizontal remapping from ESMF/TempestRemap map file
!> \details
!>  Reads the mapping file, partitions the target grid, extracts local
!>  sparse matrix entries, and builds MPI communication pattern.
!
!-----------------------------------------------------------------------
   subroutine seaice_horiz_remap_init(rd, domain, mapfile, err)!{{{

      type(seaice_horiz_remap_t), intent(inout) :: rd
      type(domain_type), intent(inout) :: domain
      character(len=*), intent(in) :: mapfile
      integer, intent(out) :: err

      ! Local variables
      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: meshPool
      type(file_desc_t) :: pioid
      type(var_desc_t) :: vid
      type(iosystem_desc_t), pointer :: pio_iosystem
      integer :: ierr, dimid
      integer :: n_s, n_a, n_b, dst_grid_rank
      integer, allocatable :: dst_grid_dims(:)
      integer, allocatable :: row_all(:), col_all(:)
      real(RKIND), allocatable :: S_all(:)
      real(RKIND), allocatable :: xc_b(:), yc_b(:)
      integer :: i, gcol, owner_rank
      integer :: row_start, row_end, nlat, nlon
      integer :: cnt, n_unique
      integer :: nCellsSolve, nProcs, myProc, comm
      integer, dimension(:), pointer :: indexToCellID
      integer, allocatable :: gcol_to_rank(:), gcol_to_myidx(:)
      integer, allocatable :: need_from_rank(:), recv_gcols(:), my_requests(:)
      integer :: r

      err = 0
      nProcs = domain%dminfo%nProcs
      myProc = domain%dminfo%myProc
      comm   = domain%dminfo%comm

      call mpas_log_write('seaice_horiz_remap_init: Reading mapping file: ' // trim(mapfile))

      ! Get PIO iosystem from domain
      pio_iosystem => domain%ioContext%pio_iosystem

      ! Open map file
      ierr = pio_openfile(pio_iosystem, pioid, PIO_IOTYPE_NETCDF, trim(mapfile), PIO_NOWRITE)
      if (ierr /= PIO_NOERR) then
         call mpas_log_write('seaice_horiz_remap_init: ERROR opening map file', MPAS_LOG_ERR)
         err = 1
         return
      end if

      ! Read dimensions
      ierr = pio_inq_dimid(pioid, 'n_s', dimid)
      ierr = pio_inq_dimlen(pioid, dimid, n_s)
      ierr = pio_inq_dimid(pioid, 'n_a', dimid)
      ierr = pio_inq_dimlen(pioid, dimid, n_a)
      ierr = pio_inq_dimid(pioid, 'n_b', dimid)
      ierr = pio_inq_dimlen(pioid, dimid, n_b)

      rd%n_a = n_a
      rd%n_b = n_b

      ! Read dst_grid_dims
      ierr = pio_inq_dimid(pioid, 'dst_grid_rank', dimid)
      ierr = pio_inq_dimlen(pioid, dimid, dst_grid_rank)
      if (dst_grid_rank /= 2) then
         call mpas_log_write('seaice_horiz_remap_init: dst_grid_rank must be 2', MPAS_LOG_ERR)
         err = 1
         return
      end if
      allocate(dst_grid_dims(dst_grid_rank))
      ierr = pio_inq_varid(pioid, 'dst_grid_dims', vid)
      ierr = pio_get_var(pioid, vid, dst_grid_dims)
      nlon = dst_grid_dims(1)
      nlat = dst_grid_dims(2)
      deallocate(dst_grid_dims)

      if (nlat * nlon /= n_b) then
         call mpas_log_write('seaice_horiz_remap_init: nlat*nlon /= n_b', MPAS_LOG_ERR)
         err = 1
         return
      end if
      rd%nlat = nlat
      rd%nlon = nlon

      call mpas_log_write('seaice_horiz_remap_init: n_a=$i n_b=$i nlat=$i nlon=$i', &
           intArgs=(/n_a, n_b, nlat, nlon/))

      ! Read full sparse matrix (all ranks, simple approach)
      allocate(row_all(n_s), col_all(n_s), S_all(n_s))
      ierr = pio_inq_varid(pioid, 'row', vid)
      ierr = pio_get_var(pioid, vid, row_all)
      ierr = pio_inq_varid(pioid, 'col', vid)
      ierr = pio_get_var(pioid, vid, col_all)
      ierr = pio_inq_varid(pioid, 'S', vid)
      ierr = pio_get_var(pioid, vid, S_all)

      ! Read target coordinates
      allocate(xc_b(n_b), yc_b(n_b))
      ierr = pio_inq_varid(pioid, 'xc_b', vid)
      ierr = pio_get_var(pioid, vid, xc_b)
      ierr = pio_inq_varid(pioid, 'yc_b', vid)
      ierr = pio_get_var(pioid, vid, yc_b)

      call pio_closefile(pioid)

      ! Extract lat/lon arrays
      allocate(rd%lat(nlat), rd%lon(nlon))
      do i = 1, nlon
         rd%lon(i) = xc_b(i)
      end do
      do i = 1, nlat
         rd%lat(i) = yc_b((i-1)*nlon + 1)
      end do
      deallocate(xc_b, yc_b)

      ! Partition target grid across MPI ranks
      row_start = myProc * n_b / nProcs + 1
      row_end   = (myProc + 1) * n_b / nProcs
      rd%n_b_local = max(0, row_end - row_start + 1)
      rd%row_start = row_start

      ! Extract local sparse matrix entries
      cnt = 0
      do i = 1, n_s
         if (row_all(i) >= row_start .and. row_all(i) <= row_end) cnt = cnt + 1
      end do
      rd%nnz_local = cnt

      allocate(rd%dst_local(cnt), rd%src_gid(cnt), rd%wgt(cnt))
      cnt = 0
      do i = 1, n_s
         if (row_all(i) >= row_start .and. row_all(i) <= row_end) then
            cnt = cnt + 1
            rd%dst_local(cnt) = row_all(i) - row_start + 1
            rd%src_gid(cnt) = col_all(i)
            rd%wgt(cnt) = S_all(i)
         end if
      end do
      deallocate(row_all, col_all, S_all)

      ! Build global cell ID to rank mapping using MPI_Allreduce
      ! Each rank marks cells it owns, then allreduce to share
      allocate(gcol_to_rank(n_a))
      gcol_to_rank(:) = -1

      ! Get indexToCellID and nCellsSolve from first block
      block => domain%blocklist
      call mpas_pool_get_subpool(block%structs, 'mesh', meshPool)
      call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)
      call mpas_pool_get_dimension(block%dimensions, 'nCellsSolve', nCellsSolve)

      do i = 1, nCellsSolve
         if (indexToCellID(i) >= 1 .and. indexToCellID(i) <= n_a) then
            gcol_to_rank(indexToCellID(i)) = myProc
         end if
      end do

      call MPI_Allreduce(MPI_IN_PLACE, gcol_to_rank, n_a, MPI_INTEGER, MPI_MAX, comm, ierr)

      ! Find unique source columns needed
      allocate(gcol_to_myidx(n_a))
      gcol_to_myidx(:) = 0
      do i = 1, rd%nnz_local
         gcol_to_myidx(rd%src_gid(i)) = 1
      end do

      n_unique = 0
      do i = 1, n_a
         if (gcol_to_myidx(i) > 0) n_unique = n_unique + 1
      end do
      rd%n_src_need = n_unique

      allocate(rd%src_need_gids(n_unique), rd%src_need_recvidx(n_unique))
      cnt = 0
      do i = 1, n_a
         if (gcol_to_myidx(i) > 0) then
            cnt = cnt + 1
            rd%src_need_gids(cnt) = i
         end if
      end do
      deallocate(gcol_to_myidx)

      ! Build recv_counts
      allocate(need_from_rank(0:nProcs-1))
      need_from_rank = 0
      do i = 1, n_unique
         gcol = rd%src_need_gids(i)
         owner_rank = gcol_to_rank(gcol)
         if (owner_rank >= 0) then
            need_from_rank(owner_rank) = need_from_rank(owner_rank) + 1
         end if
      end do

      allocate(rd%recv_counts(0:nProcs-1), rd%recv_displs(0:nProcs-1))
      rd%recv_counts = need_from_rank

      rd%recv_displs(0) = 0
      do r = 1, nProcs-1
         rd%recv_displs(r) = rd%recv_displs(r-1) + rd%recv_counts(r-1)
      end do
      rd%n_recv_total = sum(rd%recv_counts)

      ! Build ordered recv_gcols grouped by source rank
      allocate(recv_gcols(rd%n_recv_total))
      need_from_rank = rd%recv_displs
      do i = 1, n_unique
         gcol = rd%src_need_gids(i)
         owner_rank = gcol_to_rank(gcol)
         if (owner_rank >= 0) then
            need_from_rank(owner_rank) = need_from_rank(owner_rank) + 1
            recv_gcols(need_from_rank(owner_rank)) = gcol
         end if
      end do

      ! Build src_need_recvidx
      do i = 1, n_unique
         gcol = rd%src_need_gids(i)
         do cnt = 1, rd%n_recv_total
            if (recv_gcols(cnt) == gcol) then
               rd%src_need_recvidx(i) = cnt
               exit
            end if
         end do
      end do
      deallocate(need_from_rank)

      ! Exchange send_counts
      allocate(rd%send_counts(0:nProcs-1), rd%send_displs(0:nProcs-1))
      call MPI_Alltoall(rd%recv_counts, 1, MPI_INTEGER, &
                        rd%send_counts, 1, MPI_INTEGER, comm, ierr)

      rd%send_displs(0) = 0
      do r = 1, nProcs-1
         rd%send_displs(r) = rd%send_displs(r-1) + rd%send_counts(r-1)
      end do
      rd%n_send_total = sum(rd%send_counts)

      ! Exchange gcol IDs
      allocate(rd%send_cell_local(rd%n_send_total))
      allocate(my_requests(rd%n_send_total))

      call MPI_Alltoallv(recv_gcols, rd%recv_counts, rd%recv_displs, MPI_INTEGER, &
                         my_requests, rd%send_counts, rd%send_displs, MPI_INTEGER, &
                         comm, ierr)
      deallocate(recv_gcols)

      ! Build reverse map: global cell ID -> local cell index
      allocate(gcol_to_myidx(n_a))
      gcol_to_myidx(:) = 0
      do i = 1, nCellsSolve
         gcol_to_myidx(indexToCellID(i)) = i
      end do

      ! Convert requested gcols to local indices
      do i = 1, rd%n_send_total
         rd%send_cell_local(i) = gcol_to_myidx(my_requests(i))
         if (rd%send_cell_local(i) == 0) then
            call mpas_log_write('seaice_horiz_remap_init: requested cell not owned', MPAS_LOG_ERR)
            err = 1
            return
         end if
      end do

      deallocate(my_requests, gcol_to_myidx, gcol_to_rank)

      rd%initialized = .true.

      call mpas_log_write('seaice_horiz_remap_init: remapping initialized, n_b_local=$i', &
           intArgs=(/rd%n_b_local/))

   end subroutine seaice_horiz_remap_init!}}}

!***********************************************************************
!
!  routine seaice_horiz_remap_field_1d
!
!> \brief Remap a 1D field (nCells) to lat-lon
!
!-----------------------------------------------------------------------
   subroutine seaice_horiz_remap_field_1d(rd, field, nCellsSolve, fld_out, comm, err)!{{{

      type(seaice_horiz_remap_t), intent(in) :: rd
      real(RKIND), intent(in) :: field(:)  ! (nCellsSolve)
      integer, intent(in) :: nCellsSolve
      real(RKIND), allocatable, intent(out) :: fld_out(:) ! (n_b_local)
      integer, intent(in) :: comm
      integer, intent(out) :: err

      real(RKIND), allocatable :: send_buf(:), recv_buf(:)
      integer :: i, src_local, dst_local, ierr
      integer :: nProcs

      err = 0
      call MPI_Comm_size(comm, nProcs, ierr)

      allocate(fld_out(rd%n_b_local))
      fld_out = 0.0_RKIND

      ! Pack send buffer
      allocate(send_buf(rd%n_send_total))
      do i = 1, rd%n_send_total
         send_buf(i) = field(rd%send_cell_local(i))
      end do

      allocate(recv_buf(rd%n_recv_total))

      call MPI_Alltoallv(send_buf, rd%send_counts, rd%send_displs, MPI_DOUBLE_PRECISION, &
                         recv_buf, rd%recv_counts, rd%recv_displs, MPI_DOUBLE_PRECISION, &
                         comm, ierr)
      deallocate(send_buf)

      ! Sparse matrix multiply
      do i = 1, rd%nnz_local
         dst_local = rd%dst_local(i)
         src_local = src_gid_to_recvidx(rd, rd%src_gid(i))
         fld_out(dst_local) = fld_out(dst_local) + rd%wgt(i) * recv_buf(src_local)
      end do

      deallocate(recv_buf)

   end subroutine seaice_horiz_remap_field_1d!}}}

!***********************************************************************
!
!  routine seaice_horiz_remap_field_2d
!
!> \brief Remap a 2D field (nLevels x nCells) to lat-lon grid
!
!-----------------------------------------------------------------------
   subroutine seaice_horiz_remap_field_2d(rd, field, nLevels, nCellsSolve, fld_out, comm, err)!{{{

      type(seaice_horiz_remap_t), intent(in) :: rd
      real(RKIND), intent(in) :: field(:,:)  ! (nLevels, nCellsSolve)
      integer, intent(in) :: nLevels, nCellsSolve
      real(RKIND), allocatable, intent(out) :: fld_out(:,:) ! (nLevels, n_b_local)
      integer, intent(in) :: comm
      integer, intent(out) :: err

      real(RKIND), allocatable :: send_buf(:), recv_buf(:)
      integer, allocatable :: send_counts_lev(:), send_displs_lev(:)
      integer, allocatable :: recv_counts_lev(:), recv_displs_lev(:)
      integer :: i, k, src_local, dst_local, ierr
      integer :: nProcs

      err = 0
      call MPI_Comm_size(comm, nProcs, ierr)

      allocate(fld_out(nLevels, rd%n_b_local))
      fld_out = 0.0_RKIND

      ! Pack send buffer (all levels for each cell together)
      allocate(send_buf(rd%n_send_total * nLevels))
      do i = 1, rd%n_send_total
         do k = 1, nLevels
            send_buf((i-1)*nLevels + k) = field(k, rd%send_cell_local(i))
         end do
      end do

      allocate(recv_buf(rd%n_recv_total * nLevels))

      ! Scale counts/displs by nLevels
      allocate(send_counts_lev(0:nProcs-1), send_displs_lev(0:nProcs-1))
      allocate(recv_counts_lev(0:nProcs-1), recv_displs_lev(0:nProcs-1))
      do i = 0, nProcs-1
         send_counts_lev(i) = rd%send_counts(i) * nLevels
         send_displs_lev(i) = rd%send_displs(i) * nLevels
         recv_counts_lev(i) = rd%recv_counts(i) * nLevels
         recv_displs_lev(i) = rd%recv_displs(i) * nLevels
      end do

      call MPI_Alltoallv(send_buf, send_counts_lev, send_displs_lev, MPI_DOUBLE_PRECISION, &
                         recv_buf, recv_counts_lev, recv_displs_lev, MPI_DOUBLE_PRECISION, &
                         comm, ierr)

      deallocate(send_buf, send_counts_lev, send_displs_lev, recv_counts_lev, recv_displs_lev)

      ! Sparse matrix multiply
      do i = 1, rd%nnz_local
         dst_local = rd%dst_local(i)
         src_local = src_gid_to_recvidx(rd, rd%src_gid(i))
         do k = 1, nLevels
            fld_out(k, dst_local) = fld_out(k, dst_local) + &
                 rd%wgt(i) * recv_buf((src_local-1)*nLevels + k)
         end do
      end do

      deallocate(recv_buf)

   end subroutine seaice_horiz_remap_field_2d!}}}

end module seaice_horiz_remap
