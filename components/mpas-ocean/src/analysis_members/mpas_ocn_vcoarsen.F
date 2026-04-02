! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_vcoarsen
!
!> \brief MPAS ocean analysis mode member: vertical coarsening
!> \author E3SM Vertical Coarsening Support
!> \date   April 2026
!> \details
!>  Computes layer-thickness-weighted vertical averages of ocean fields
!>  onto a coarsened set of depth levels using the shared shr_vcoarsen_mod.
!>  Also supports level selection by index or nearest depth value.
!>
!-----------------------------------------------------------------------

module ocn_vcoarsen

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_timekeeping
   use mpas_stream_manager
   use mpas_kind_types

   use ocn_constants
   use ocn_diagnostics_variables

   use shr_kind_mod, only: r8 => shr_kind_r8

   implicit none
   private
   save

   public :: ocn_init_vcoarsen, &
             ocn_compute_vcoarsen, &
             ocn_restart_vcoarsen, &
             ocn_finalize_vcoarsen

   integer :: nCoarsenLevels = 0
   integer :: nCoarsenFields = 0
   integer :: nSelectLevels  = 0
   integer :: nSelectDepths  = 0
   real(kind=RKIND), allocatable :: depthBounds(:)
   character(len=64), allocatable :: coarsenFieldNames(:)
   integer, allocatable :: selectLevelIndices(:)
   real(kind=RKIND), allocatable :: selectDepthValues(:)

   real(kind=RKIND), parameter :: fillValue = 1.0E34_RKIND

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_vcoarsen
!
!> \brief   Initialize vertical coarsening
!> \details
!>  Parses configuration for depth bounds, field names, and selection
!>  parameters from the MPAS config pool.
!
!-----------------------------------------------------------------------

   subroutine ocn_init_vcoarsen(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      character(len=StrKIND), pointer :: config_AM_vcoarsen_depth_bounds
      character(len=StrKIND), pointer :: config_AM_vcoarsen_fields
      character(len=StrKIND), pointer :: config_AM_vcoarsen_select_levels
      character(len=StrKIND), pointer :: config_AM_vcoarsen_select_depths
      character(len=StrKIND) :: tempStr
      integer :: nVals, i, iStart, iEnd, strLen

      err = 0

      call mpas_pool_get_config(domain % configs, &
           'config_AM_vcoarsen_depth_bounds', config_AM_vcoarsen_depth_bounds)
      call mpas_pool_get_config(domain % configs, &
           'config_AM_vcoarsen_fields', config_AM_vcoarsen_fields)
      call mpas_pool_get_config(domain % configs, &
           'config_AM_vcoarsen_select_levels', config_AM_vcoarsen_select_levels)
      call mpas_pool_get_config(domain % configs, &
           'config_AM_vcoarsen_select_depths', config_AM_vcoarsen_select_depths)

      ! Parse depth bounds for averaging
      tempStr = trim(config_AM_vcoarsen_depth_bounds)
      if (trim(tempStr) == 'none') then
         nCoarsenLevels = 0
      else
         call count_csv(tempStr, nVals)
         allocate(depthBounds(nVals))
         call parse_csv_reals(tempStr, depthBounds, nVals)
         nCoarsenLevels = nVals - 1
      end if

      ! Parse field names
      tempStr = trim(config_AM_vcoarsen_fields)
      if (trim(tempStr) == 'none') then
         nCoarsenFields = 0
      else
         call count_csv(tempStr, nVals)
         nCoarsenFields = nVals
         allocate(coarsenFieldNames(nVals))
         call parse_csv_strings(tempStr, coarsenFieldNames, nVals)
      end if

      ! Parse select level indices
      tempStr = trim(config_AM_vcoarsen_select_levels)
      if (trim(tempStr) == 'none') then
         nSelectLevels = 0
      else
         call count_csv(tempStr, nVals)
         nSelectLevels = nVals
         allocate(selectLevelIndices(nVals))
         call parse_csv_ints(tempStr, selectLevelIndices, nVals)
      end if

      ! Parse select depth values
      tempStr = trim(config_AM_vcoarsen_select_depths)
      if (trim(tempStr) == 'none') then
         nSelectDepths = 0
      else
         call count_csv(tempStr, nVals)
         nSelectDepths = nVals
         allocate(selectDepthValues(nVals))
         call parse_csv_reals(tempStr, selectDepthValues, nVals)
      end if

   end subroutine ocn_init_vcoarsen!}}}

!***********************************************************************
!
!  routine ocn_compute_vcoarsen
!
!> \brief   Compute vertical coarsening using shared utility
!> \details
!>  For each cell, builds depth interface array from accumulated
!>  layerThickness and delegates to shr_vcoarsen_avg (single column).
!
!-----------------------------------------------------------------------

   subroutine ocn_compute_vcoarsen(domain, timeLevel, err)!{{{

      use shr_vcoarsen_mod, only: shr_vcoarsen_avg

      integer, intent(in) :: timeLevel
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: statePool, meshPool, tracersPool, amPool
      type(mpas_pool_type), pointer :: diagPool

      integer, pointer :: nCellsSolve, nVertLevels, nVcoarsenLevels
      integer :: iCell, k, kOut, iField, tracerIdx
      integer, dimension(:), pointer :: maxLevelCell

      real(kind=RKIND), dimension(:, :, :), pointer :: tracers
      real(kind=RKIND), dimension(:, :), pointer :: layerThickness
      real(kind=RKIND), dimension(:, :), pointer :: srcField2d
      real(kind=RKIND), dimension(:, :), pointer :: outField2d

      ! Working arrays for single-column calls
      real(r8), allocatable :: coord_iface(:)   ! (nVertLevels+1)
      real(r8), allocatable :: field_1d(:)      ! (nVertLevels)
      real(r8), allocatable :: out_1d(:)        ! (nCoarsenLevels)
      character(len=64) :: fieldName, outFieldName
      logical :: fieldFound
      integer, pointer :: tracerIdxPtr

      err = 0

      if (nCoarsenLevels == 0 .or. nCoarsenFields == 0) return

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'vcoarsenAM', amPool)
         call mpas_pool_get_subpool(block % structs, 'diagnostics', diagPool)

         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_dimension(block % dimensions, 'nVertLevels', nVertLevels)
         call mpas_pool_get_dimension(block % dimensions, 'nVcoarsenLevels', nVcoarsenLevels)

         call mpas_pool_get_array(tracersPool, 'activeTracers', tracers, timeLevel)
         call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, timeLevel)
         call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)

         allocate(coord_iface(nVertLevels + 1))
         allocate(field_1d(nVertLevels))
         allocate(out_1d(nCoarsenLevels))

         ! Process each requested field
         do iField = 1, nCoarsenFields
            fieldName = trim(coarsenFieldNames(iField))
            outFieldName = trim(fieldName) // 'Coarsened'

            ! Look up the output array in the AM pool
            nullify(outField2d)
            call mpas_pool_get_array(amPool, trim(outFieldName), outField2d)
            if (.not. associated(outField2d)) cycle

            ! Try to find the source field:
            ! 1) Check tracer indices (temperature, salinity, etc.)
            ! 2) Check state pool (layerThickness, normalVelocity, etc.)
            ! 3) Check diagnostics pool (velocityZonal, velocityMeridional, etc.)
            fieldFound = .false.

            ! Check tracers by index name
            nullify(tracerIdxPtr)
            call mpas_pool_get_dimension(tracersPool, 'index_' // trim(fieldName), tracerIdxPtr)

            if (associated(tracerIdxPtr)) then
               ! Field is a tracer
               fieldFound = .true.
               tracerIdx = tracerIdxPtr

               do iCell = 1, nCellsSolve
                  call build_coord_iface(coord_iface, layerThickness, iCell, &
                       maxLevelCell(iCell), nVertLevels)

                  field_1d(1:maxLevelCell(iCell)) = tracers(tracerIdx, 1:maxLevelCell(iCell), iCell)
                  field_1d(maxLevelCell(iCell)+1:nVertLevels) = 0.0_r8
                  call shr_vcoarsen_avg(field_1d, coord_iface, nVertLevels, &
                       depthBounds(1:nCoarsenLevels+1), nCoarsenLevels, &
                       real(fillValue, r8), out_1d)
                  outField2d(1:nCoarsenLevels, iCell) = out_1d(1:nCoarsenLevels)
                  outField2d(nCoarsenLevels+1:nVcoarsenLevels, iCell) = fillValue
               end do
            end if

            if (.not. fieldFound) then
               ! Try state pool, then diagnostics pool for 2D fields
               nullify(srcField2d)
               call mpas_pool_get_array(statePool, trim(fieldName), srcField2d, timeLevel)
               if (.not. associated(srcField2d)) then
                  call mpas_pool_get_array(diagPool, trim(fieldName), srcField2d)
               end if

               if (associated(srcField2d)) then
                  fieldFound = .true.
                  do iCell = 1, nCellsSolve
                     call build_coord_iface(coord_iface, layerThickness, iCell, &
                          maxLevelCell(iCell), nVertLevels)

                     field_1d(1:maxLevelCell(iCell)) = srcField2d(1:maxLevelCell(iCell), iCell)
                     field_1d(maxLevelCell(iCell)+1:nVertLevels) = 0.0_r8
                     call shr_vcoarsen_avg(field_1d, coord_iface, nVertLevels, &
                          depthBounds(1:nCoarsenLevels+1), nCoarsenLevels, &
                          real(fillValue, r8), out_1d)
                     outField2d(1:nCoarsenLevels, iCell) = out_1d(1:nCoarsenLevels)
                     outField2d(nCoarsenLevels+1:nVcoarsenLevels, iCell) = fillValue
                  end do
               end if
            end if

         end do

         deallocate(coord_iface, field_1d, out_1d)

         block => block % next
      end do

   end subroutine ocn_compute_vcoarsen!}}}

!***********************************************************************

   subroutine build_coord_iface(coord_iface, layerThickness, iCell, maxLevel, nVertLevels)!{{{
      real(kind=RKIND), intent(out) :: coord_iface(:)
      real(kind=RKIND), dimension(:,:), intent(in) :: layerThickness
      integer, intent(in) :: iCell, maxLevel, nVertLevels
      integer :: k

      coord_iface(1) = 0.0_RKIND
      do k = 1, maxLevel
         coord_iface(k + 1) = coord_iface(k) + layerThickness(k, iCell)
      end do
      do k = maxLevel + 1, nVertLevels
         coord_iface(k + 1) = coord_iface(k)
      end do
   end subroutine build_coord_iface!}}}

!***********************************************************************

   subroutine ocn_restart_vcoarsen(domain, err)!{{{
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err
      err = 0
   end subroutine ocn_restart_vcoarsen!}}}

!***********************************************************************

   subroutine ocn_finalize_vcoarsen(domain, err)!{{{
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err
      err = 0
      if (allocated(depthBounds)) deallocate(depthBounds)
      if (allocated(coarsenFieldNames)) deallocate(coarsenFieldNames)
      if (allocated(selectLevelIndices)) deallocate(selectLevelIndices)
      if (allocated(selectDepthValues)) deallocate(selectDepthValues)
   end subroutine ocn_finalize_vcoarsen!}}}

!***********************************************************************
! Private CSV parsing helpers
!***********************************************************************

   subroutine count_csv(str, n)!{{{
      character(len=*), intent(in) :: str
      integer, intent(out) :: n
      integer :: i, slen
      n = 1
      slen = len_trim(str)
      do i = 1, slen
         if (str(i:i) == ',') n = n + 1
      end do
   end subroutine count_csv!}}}

   subroutine parse_csv_reals(str, vals, n)!{{{
      character(len=*), intent(in) :: str
      integer, intent(in) :: n
      real(kind=RKIND), intent(out) :: vals(n)
      integer :: i, iStart, iEnd, slen
      slen = len_trim(str)
      iStart = 1
      do i = 1, n
         iEnd = index(str(iStart:slen), ',')
         if (iEnd == 0) then
            iEnd = slen + 1
         else
            iEnd = iEnd + iStart - 1
         end if
         read(str(iStart:iEnd-1), *) vals(i)
         iStart = iEnd + 1
      end do
   end subroutine parse_csv_reals!}}}

   subroutine parse_csv_ints(str, vals, n)!{{{
      character(len=*), intent(in) :: str
      integer, intent(in) :: n
      integer, intent(out) :: vals(n)
      integer :: i, iStart, iEnd, slen
      slen = len_trim(str)
      iStart = 1
      do i = 1, n
         iEnd = index(str(iStart:slen), ',')
         if (iEnd == 0) then
            iEnd = slen + 1
         else
            iEnd = iEnd + iStart - 1
         end if
         read(str(iStart:iEnd-1), *) vals(i)
         iStart = iEnd + 1
      end do
   end subroutine parse_csv_ints!}}}

   subroutine parse_csv_strings(str, vals, n)!{{{
      character(len=*), intent(in) :: str
      integer, intent(in) :: n
      character(len=64), intent(out) :: vals(n)
      integer :: i, iStart, iEnd, slen
      slen = len_trim(str)
      iStart = 1
      do i = 1, n
         iEnd = index(str(iStart:slen), ',')
         if (iEnd == 0) then
            iEnd = slen + 1
         else
            iEnd = iEnd + iStart - 1
         end if
         vals(i) = adjustl(str(iStart:iEnd-1))
         iStart = iEnd + 1
      end do
   end subroutine parse_csv_strings!}}}

end module ocn_vcoarsen

! vim: foldmethod=marker
