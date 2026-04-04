! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_fme_horiz_remap
!
!> \brief MPAS ocean FME horizontal remapping wrapper
!> \author FME Training Data Support
!> \date   April 2026
!> \details
!>  Thin wrapper around shr_horiz_remap_t for MPAS-Ocean. Handles
!>  MPAS cell decomposition mapping, send buffer packing from MPAS
!>  block data structures, and applies the shared CRS SpMV remapping.
!>
!>  Usage:
!>    call ocn_fme_horiz_remap_init(domain, mapfile, err)
!>    call ocn_fme_horiz_remap_field(domain, field2d, numlev, fld_out, err)
!>    logical = ocn_fme_horiz_remap_active()
!>
!-----------------------------------------------------------------------

module ocn_fme_horiz_remap

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_kind_types
   use mpas_log, only: mpas_log_write

   use shr_kind_mod, only: r8 => shr_kind_r8
   use shr_horiz_remap_mod, only: shr_horiz_remap_t

   implicit none
   private
   save

   public :: ocn_fme_horiz_remap_init
   public :: ocn_fme_horiz_remap_field
   public :: ocn_fme_horiz_remap_active

   type(shr_horiz_remap_t) :: remap_data

   ! Send-side cell mapping: local cell index for each entry in send list
   integer, allocatable :: send_local_cell(:)

   ! Persistent send buffer
   real(r8), allocatable :: ws_send_buf(:)

   logical :: is_initialized = .false.

!***********************************************************************

contains

!***********************************************************************

   subroutine ocn_fme_horiz_remap_init(domain, mapfile, err)!{{{

      type(domain_type), intent(inout) :: domain
      character(len=*), intent(in) :: mapfile
      integer, intent(out) :: err

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: meshPool
      integer, pointer :: nCellsSolve
      integer, dimension(:), pointer :: indexToCellID

      integer :: myrank, nprocs, comm, ierr
      integer :: iCell, gcol
      integer, allocatable :: gcol_to_rank(:)
      integer, allocatable :: send_gcol_list(:)
      integer :: n_send, i

      err = 0

      if (len_trim(mapfile) == 0 .or. trim(mapfile) == 'none') then
         is_initialized = .false.
         return
      end if

      myrank = domain % dminfo % my_proc_id
      nprocs = domain % dminfo % nprocs
      comm   = domain % dminfo % comm

      ! Phase 1: Read map file
      call remap_data%read_mapfile(trim(mapfile), comm, myrank, nprocs, &
           domain % ioContext % pio_iosystem, ierr)
      if (ierr /= 0) then
         call mpas_log_write('ocn_fme_horiz_remap_init: failed to read map file: ' &
              // trim(mapfile), messageType=MPAS_LOG_ERR)
         err = ierr
         return
      end if

      ! Build gcol_to_rank mapping from MPAS cell decomposition
      allocate(gcol_to_rank(remap_data%n_a))
      gcol_to_rank(:) = -1

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)

         do iCell = 1, nCellsSolve
            gcol = indexToCellID(iCell)
            if (gcol >= 1 .and. gcol <= remap_data%n_a) then
               gcol_to_rank(gcol) = myrank
            end if
         end do

         block => block % next
      end do

      ! Phase 2: Build MPI communication pattern
      call remap_data%build_comm(gcol_to_rank, comm, myrank, nprocs, &
           send_gcol_list, n_send, ierr)
      deallocate(gcol_to_rank)

      if (ierr /= 0) then
         call mpas_log_write('ocn_fme_horiz_remap_init: failed to build comm pattern', &
              messageType=MPAS_LOG_ERR)
         err = ierr
         return
      end if

      ! Convert send_gcol_list (global cell IDs) to local cell indices
      allocate(send_local_cell(n_send))
      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)

         do i = 1, n_send
            do iCell = 1, nCellsSolve
               if (indexToCellID(iCell) == send_gcol_list(i)) then
                  send_local_cell(i) = iCell
                  exit
               end if
            end do
         end do

         block => block % next
      end do

      if (allocated(send_gcol_list)) deallocate(send_gcol_list)

      is_initialized = .true.

      if (myrank == 0) then
         call mpas_log_write('ocn_fme_horiz_remap_init: initialized with ' &
              // 'n_a=$i, n_b=$i, nlat=$i, nlon=$i', &
              intArgs=(/remap_data%n_a, remap_data%n_b, &
                        remap_data%nlat, remap_data%nlon/))
      end if

   end subroutine ocn_fme_horiz_remap_init!}}}

!***********************************************************************

   subroutine ocn_fme_horiz_remap_field(domain, field2d, numlev, fld_out, err)!{{{
      !-----------------------------------------------------------------
      ! Remap a 2D field (nVertLevels, nCells) or 1D field (nCells)
      ! from MPAS native grid to the target lat-lon grid.
      !
      ! For 1D fields, pass numlev=1 and field2d(1, nCells).
      !-----------------------------------------------------------------

      type(domain_type), intent(inout) :: domain
      integer, intent(in) :: numlev
      real(kind=RKIND), intent(in) :: field2d(numlev, *)
      real(r8), intent(out) :: fld_out(remap_data%n_b_local, numlev)
      integer, intent(out) :: err

      integer :: i, k, needed, comm, nprocs
      integer :: myrank

      err = 0

      if (.not. is_initialized) then
         err = 1
         return
      end if

      myrank = domain % dminfo % my_proc_id
      nprocs = domain % dminfo % nprocs
      comm   = domain % dminfo % comm

      ! Pack send buffer from MPAS cell data
      needed = remap_data%n_send_total * numlev
      if (needed > 0) then
         if (.not. allocated(ws_send_buf) .or. size(ws_send_buf) < needed) then
            if (allocated(ws_send_buf)) deallocate(ws_send_buf)
            allocate(ws_send_buf(needed))
         end if

         do i = 1, remap_data%n_send_total
            do k = 1, numlev
               ws_send_buf((i-1)*numlev + k) = &
                    real(field2d(k, send_local_cell(i)), r8)
            end do
         end do
      end if

      ! Apply shared SpMV remapping
      call remap_data%apply(ws_send_buf, numlev, fld_out, comm, nprocs, err)

   end subroutine ocn_fme_horiz_remap_field!}}}

!***********************************************************************

   logical function ocn_fme_horiz_remap_active()!{{{
      ocn_fme_horiz_remap_active = is_initialized
   end function ocn_fme_horiz_remap_active!}}}

end module ocn_fme_horiz_remap

! vim: foldmethod=marker
