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

      integer, pointer :: nCellsSolve, nVertLevels, nVcoarsenLevels
      integer, pointer :: index_temperature, index_salinity
      integer :: iCell, k, kOut
      integer, dimension(:), pointer :: maxLevelCell

      real(kind=RKIND), dimension(:, :, :), pointer :: tracers
      real(kind=RKIND), dimension(:, :), pointer :: layerThickness
      real(kind=RKIND), dimension(:, :), pointer :: temperatureCoarsened
      real(kind=RKIND), dimension(:, :), pointer :: salinityCoarsened
      real(kind=RKIND), dimension(:, :), pointer :: velocityZonalCoarsened
      real(kind=RKIND), dimension(:, :), pointer :: velocityMeridionalCoarsened
      real(kind=RKIND), dimension(:, :), pointer :: layerThicknessCoarsened

      ! Working arrays for single-column calls
      real(r8), allocatable :: coord_iface(:)   ! (nVertLevels+1)
      real(r8), allocatable :: field_1d(:)      ! (nVertLevels)
      real(r8), allocatable :: out_1d(:)        ! (nCoarsenLevels)
      real(r8) :: depthTop, depthBot, overlap, denominator

      err = 0

      if (nCoarsenLevels == 0) return

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'vcoarsenAM', amPool)

         call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
         call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_dimension(block % dimensions, 'nVertLevels', nVertLevels)
         call mpas_pool_get_dimension(block % dimensions, 'nVcoarsenLevels', nVcoarsenLevels)

         call mpas_pool_get_array(tracersPool, 'activeTracers', tracers, timeLevel)
         call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, timeLevel)
         call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)

         call mpas_pool_get_array(amPool, 'temperatureCoarsened', temperatureCoarsened)
         call mpas_pool_get_array(amPool, 'salinityCoarsened', salinityCoarsened)
         call mpas_pool_get_array(amPool, 'velocityZonalCoarsened', velocityZonalCoarsened)
         call mpas_pool_get_array(amPool, 'velocityMeridionalCoarsened', velocityMeridionalCoarsened)
         call mpas_pool_get_array(amPool, 'layerThicknessCoarsened', layerThicknessCoarsened)

         allocate(coord_iface(nVertLevels + 1))
         allocate(field_1d(nVertLevels))
         allocate(out_1d(nCoarsenLevels))

         do iCell = 1, nCellsSolve

            ! Build depth interfaces from layer thickness
            coord_iface(1) = 0.0_r8
            do k = 1, maxLevelCell(iCell)
               coord_iface(k + 1) = coord_iface(k) + layerThickness(k, iCell)
            end do
            ! Fill remaining interfaces for unused levels
            do k = maxLevelCell(iCell) + 1, nVertLevels
               coord_iface(k + 1) = coord_iface(k)
            end do

            ! Temperature
            field_1d(1:maxLevelCell(iCell)) = tracers(index_temperature, 1:maxLevelCell(iCell), iCell)
            field_1d(maxLevelCell(iCell)+1:nVertLevels) = 0.0_r8
            call shr_vcoarsen_avg(field_1d, coord_iface, nVertLevels, &
                 depthBounds(1:nCoarsenLevels+1), nCoarsenLevels, &
                 real(fillValue, r8), out_1d)
            temperatureCoarsened(1:nCoarsenLevels, iCell) = out_1d(1:nCoarsenLevels)

            ! Salinity
            field_1d(1:maxLevelCell(iCell)) = tracers(index_salinity, 1:maxLevelCell(iCell), iCell)
            field_1d(maxLevelCell(iCell)+1:nVertLevels) = 0.0_r8
            call shr_vcoarsen_avg(field_1d, coord_iface, nVertLevels, &
                 depthBounds(1:nCoarsenLevels+1), nCoarsenLevels, &
                 real(fillValue, r8), out_1d)
            salinityCoarsened(1:nCoarsenLevels, iCell) = out_1d(1:nCoarsenLevels)

            ! Zonal velocity
            field_1d(1:maxLevelCell(iCell)) = velocityZonal(1:maxLevelCell(iCell), iCell)
            field_1d(maxLevelCell(iCell)+1:nVertLevels) = 0.0_r8
            call shr_vcoarsen_avg(field_1d, coord_iface, nVertLevels, &
                 depthBounds(1:nCoarsenLevels+1), nCoarsenLevels, &
                 real(fillValue, r8), out_1d)
            velocityZonalCoarsened(1:nCoarsenLevels, iCell) = out_1d(1:nCoarsenLevels)

            ! Meridional velocity
            field_1d(1:maxLevelCell(iCell)) = velocityMeridional(1:maxLevelCell(iCell), iCell)
            field_1d(maxLevelCell(iCell)+1:nVertLevels) = 0.0_r8
            call shr_vcoarsen_avg(field_1d, coord_iface, nVertLevels, &
                 depthBounds(1:nCoarsenLevels+1), nCoarsenLevels, &
                 real(fillValue, r8), out_1d)
            velocityMeridionalCoarsened(1:nCoarsenLevels, iCell) = out_1d(1:nCoarsenLevels)

            ! Layer thickness (sum, not average - compute directly)
            do kOut = 1, nCoarsenLevels
               denominator = 0.0_RKIND
               depthTop = 0.0_RKIND
               do k = 1, maxLevelCell(iCell)
                  depthBot = depthTop + layerThickness(k, iCell)
                  overlap = max(0.0_RKIND, min(depthBot, depthBounds(kOut+1)) &
                       - max(depthTop, depthBounds(kOut)))
                  denominator = denominator + overlap
                  depthTop = depthBot
               end do
               if (denominator > 0.0_RKIND) then
                  layerThicknessCoarsened(kOut, iCell) = denominator
               else
                  layerThicknessCoarsened(kOut, iCell) = fillValue
               end if
            end do

            ! Fill unused output levels
            do kOut = nCoarsenLevels + 1, nVcoarsenLevels
               temperatureCoarsened(kOut, iCell) = fillValue
               salinityCoarsened(kOut, iCell) = fillValue
               velocityZonalCoarsened(kOut, iCell) = fillValue
               velocityMeridionalCoarsened(kOut, iCell) = fillValue
               layerThicknessCoarsened(kOut, iCell) = fillValue
            end do

         end do

         deallocate(coord_iface, field_1d, out_1d)

         block => block % next
      end do

   end subroutine ocn_compute_vcoarsen!}}}

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
