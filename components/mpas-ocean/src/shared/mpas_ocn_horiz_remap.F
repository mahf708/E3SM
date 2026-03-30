! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_horiz_remap
!
!> \brief MPAS ocean horizontal remapping for ACE training data
!> \author ACE team
!> \date   March 2026
!> \details
!>  This module provides online horizontal remapping of MPAS-Ocean fields
!>  from the unstructured mesh to a regular lat-lon grid via ESMF/TempestRemap
!>  mapping files. It is a thin wrapper around shr_horiz_remap_mod, handling
!>  MPAS-specific decomposition and data packing.
!
!-----------------------------------------------------------------------

module ocn_horiz_remap

   use mpas_kind_types, only: RKIND, StrKIND
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_log, only: mpas_log_write
   use shr_horiz_remap_mod, only: shr_horiz_remap_t, &
        shr_horiz_remap_read_mapfile, shr_horiz_remap_build_comm, &
        shr_horiz_remap_apply

   implicit none
   private
   save

   type, public :: ocn_horiz_remap_t
      type(shr_horiz_remap_t) :: shared
      integer, allocatable :: send_cell_local(:)  ! local cell index per send entry
   end type ocn_horiz_remap_t

   public :: ocn_horiz_remap_init
   public :: ocn_horiz_remap_field_1d
   public :: ocn_horiz_remap_field_2d
   public :: ocn_horiz_remap_is_active

contains

!***********************************************************************
!
!  routine ocn_horiz_remap_is_active
!
!> \brief Check if horizontal remapping is active
!
!-----------------------------------------------------------------------
   logical function ocn_horiz_remap_is_active(rd)!{{{
      type(ocn_horiz_remap_t), intent(in) :: rd
      ocn_horiz_remap_is_active = rd%shared%initialized
   end function ocn_horiz_remap_is_active!}}}

!***********************************************************************
!
!  routine ocn_horiz_remap_init
!
!> \brief Initialize horizontal remapping from ESMF/TempestRemap map file
!> \details
!>  Uses shr_horiz_remap_mod to read the mapping file and build the MPI
!>  communication pattern.  MPAS-specific work: building gcol_to_rank
!>  from indexToCellID and converting send_gcol_list to local indices.
!
!-----------------------------------------------------------------------
   subroutine ocn_horiz_remap_init(rd, domain, mapfile, err)!{{{

      type(ocn_horiz_remap_t), intent(inout) :: rd
      type(domain_type), intent(inout) :: domain
      character(len=*), intent(in) :: mapfile
      integer, intent(out) :: err

      ! Local variables
      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: meshPool
      integer :: ierr, i
      integer :: nCellsSolve, nProcs, myProc, comm
      integer, dimension(:), pointer :: indexToCellID
      integer, allocatable :: gcol_to_rank(:), gcol_to_myidx(:)
      integer, allocatable :: send_gcol_list(:)

      err = 0
      nProcs = domain%dminfo%nProcs
      myProc = domain%dminfo%myProc
      comm   = domain%dminfo%comm

      call mpas_log_write('ocn_horiz_remap_init: Reading mapping file: ' // trim(mapfile))

      ! Phase 1: Read map file via shared module
      call shr_horiz_remap_read_mapfile(rd%shared, mapfile, comm, myProc, nProcs, &
           domain%ioContext%pio_iosystem, ierr)
      if (ierr /= 0) then
         call mpas_log_write('ocn_horiz_remap_init: ERROR reading map file', MPAS_LOG_ERR)
         err = ierr
         return
      end if

      call mpas_log_write('ocn_horiz_remap_init: n_a=$i n_b=$i nlat=$i nlon=$i', &
           intArgs=(/rd%shared%n_a, rd%shared%n_b, rd%shared%nlat, rd%shared%nlon/))

      ! Build gcol_to_rank using MPAS indexToCellID
      allocate(gcol_to_rank(rd%shared%n_a))
      gcol_to_rank(:) = -1

      block => domain%blocklist
      call mpas_pool_get_subpool(block%structs, 'mesh', meshPool)
      call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)
      call mpas_pool_get_dimension(block%dimensions, 'nCellsSolve', nCellsSolve)

      do i = 1, nCellsSolve
         if (indexToCellID(i) >= 1 .and. indexToCellID(i) <= rd%shared%n_a) then
            gcol_to_rank(indexToCellID(i)) = myProc
         end if
      end do

      call MPI_Allreduce(MPI_IN_PLACE, gcol_to_rank, rd%shared%n_a, MPI_INTEGER, &
                         MPI_MAX, comm, ierr)

      ! Phase 2: Build communication pattern via shared module
      call shr_horiz_remap_build_comm(rd%shared, gcol_to_rank, comm, myProc, nProcs, &
           send_gcol_list, ierr)
      if (ierr /= 0) then
         call mpas_log_write('ocn_horiz_remap_init: ERROR building comm pattern', MPAS_LOG_ERR)
         err = ierr
         deallocate(gcol_to_rank)
         return
      end if

      deallocate(gcol_to_rank)

      ! Convert send_gcol_list (global cell IDs) to local cell indices
      allocate(gcol_to_myidx(rd%shared%n_a))
      gcol_to_myidx(:) = 0
      do i = 1, nCellsSolve
         gcol_to_myidx(indexToCellID(i)) = i
      end do

      allocate(rd%send_cell_local(rd%shared%n_send_total))
      do i = 1, rd%shared%n_send_total
         rd%send_cell_local(i) = gcol_to_myidx(send_gcol_list(i))
         if (rd%send_cell_local(i) == 0) then
            call mpas_log_write('ocn_horiz_remap_init: requested cell not owned', MPAS_LOG_ERR)
            err = 1
            deallocate(gcol_to_myidx, send_gcol_list)
            return
         end if
      end do

      deallocate(gcol_to_myidx, send_gcol_list)

      call mpas_log_write('ocn_horiz_remap_init: remapping initialized, n_b_local=$i', &
           intArgs=(/rd%shared%n_b_local/))

   end subroutine ocn_horiz_remap_init!}}}

!***********************************************************************
!
!  routine ocn_horiz_remap_field_1d
!
!> \brief Remap a 1D field (nCells) to lat-lon
!
!-----------------------------------------------------------------------
   subroutine ocn_horiz_remap_field_1d(rd, field, nCellsSolve, fld_out, comm, err)!{{{

      use shr_kind_mod, only: r8 => shr_kind_r8

      type(ocn_horiz_remap_t), intent(in) :: rd
      real(RKIND), intent(in) :: field(:)  ! (nCellsSolve)
      integer, intent(in) :: nCellsSolve
      real(RKIND), allocatable, intent(out) :: fld_out(:) ! (n_b_local)
      integer, intent(in) :: comm
      integer, intent(out) :: err

      real(r8), allocatable :: send_buf(:)
      real(r8), allocatable :: fld_out_2d(:,:)
      integer :: i, ierr, nProcs

      err = 0
      call MPI_Comm_size(comm, nProcs, ierr)

      ! Pack send buffer from MPAS data using local cell indices
      allocate(send_buf(rd%shared%n_send_total))
      do i = 1, rd%shared%n_send_total
         send_buf(i) = field(rd%send_cell_local(i))
      end do

      ! Apply remapping via shared module (numlev=1)
      allocate(fld_out_2d(rd%shared%n_b_local, 1))
      call shr_horiz_remap_apply(rd%shared, send_buf, 1, fld_out_2d, comm, nProcs, ierr)
      deallocate(send_buf)

      if (ierr /= 0) then
         err = ierr
         deallocate(fld_out_2d)
         return
      end if

      ! Copy to 1D output
      allocate(fld_out(rd%shared%n_b_local))
      fld_out(:) = fld_out_2d(:, 1)
      deallocate(fld_out_2d)

   end subroutine ocn_horiz_remap_field_1d!}}}

!***********************************************************************
!
!  routine ocn_horiz_remap_field_2d
!
!> \brief Remap a 2D field (nLevels x nCells) to lat-lon grid
!
!-----------------------------------------------------------------------
   subroutine ocn_horiz_remap_field_2d(rd, field, nLevels, nCellsSolve, fld_out, comm, err)!{{{

      use shr_kind_mod, only: r8 => shr_kind_r8

      type(ocn_horiz_remap_t), intent(in) :: rd
      real(RKIND), intent(in) :: field(:,:)  ! (nLevels, nCellsSolve)
      integer, intent(in) :: nLevels, nCellsSolve
      real(RKIND), allocatable, intent(out) :: fld_out(:,:) ! (nLevels, n_b_local)
      integer, intent(in) :: comm
      integer, intent(out) :: err

      real(r8), allocatable :: send_buf(:)
      real(r8), allocatable :: fld_out_shared(:,:)
      integer :: i, k, ierr, nProcs

      err = 0
      call MPI_Comm_size(comm, nProcs, ierr)

      ! Pack send buffer (all levels for each cell together)
      allocate(send_buf(rd%shared%n_send_total * nLevels))
      do i = 1, rd%shared%n_send_total
         do k = 1, nLevels
            send_buf((i-1)*nLevels + k) = field(k, rd%send_cell_local(i))
         end do
      end do

      ! Apply remapping via shared module
      allocate(fld_out_shared(rd%shared%n_b_local, nLevels))
      call shr_horiz_remap_apply(rd%shared, send_buf, nLevels, fld_out_shared, &
                                 comm, nProcs, ierr)
      deallocate(send_buf)

      if (ierr /= 0) then
         err = ierr
         deallocate(fld_out_shared)
         return
      end if

      ! Transpose from shared layout (n_b_local, nLevels) to MPAS layout (nLevels, n_b_local)
      allocate(fld_out(nLevels, rd%shared%n_b_local))
      do i = 1, rd%shared%n_b_local
         do k = 1, nLevels
            fld_out(k, i) = fld_out_shared(i, k)
         end do
      end do
      deallocate(fld_out_shared)

   end subroutine ocn_horiz_remap_field_2d!}}}

end module ocn_horiz_remap
