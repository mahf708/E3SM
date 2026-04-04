! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_fme_depth_coarsening
!
!> \brief MPAS ocean analysis member: FME depth coarsening
!> \author FME Training Data Support
!> \date   April 2026
!> \details
!>  Computes layer-thickness-weighted vertical averages of tracer and
!>  velocity fields onto a coarsened set of depth levels defined by
!>  user-specified depth boundaries. Uses shr_vcoarsen_avg from the
!>  shared vertical coarsening utility module.
!>
!-----------------------------------------------------------------------

module ocn_fme_depth_coarsening

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_timekeeping
   use mpas_stream_manager
   use mpas_kind_types

   use ocn_constants
   use ocn_diagnostics_variables
   use shr_vcoarsen_mod, only: shr_vcoarsen_avg
   use shr_kind_mod, only: r8 => shr_kind_r8

   implicit none
   private
   save

   public :: ocn_init_fme_depth_coarsening, &
             ocn_compute_fme_depth_coarsening, &
             ocn_restart_fme_depth_coarsening, &
             ocn_finalize_fme_depth_coarsening

   integer :: nCoarsenLevels = 0
   real(kind=RKIND), allocatable :: depthBounds(:)

   real(kind=RKIND), parameter :: fillValue = 1.0E34_RKIND

!***********************************************************************

contains

!***********************************************************************

   subroutine ocn_init_fme_depth_coarsening(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      character(len=StrKIND), pointer :: config_AM_fmeDepthCoarsening_depth_bounds
      character(len=StrKIND) :: tempStr
      integer :: nBounds, i, iStart, iEnd, strLen

      err = 0

      call mpas_pool_get_config(domain % configs, &
           'config_AM_fmeDepthCoarsening_depth_bounds', &
           config_AM_fmeDepthCoarsening_depth_bounds)

      tempStr = trim(config_AM_fmeDepthCoarsening_depth_bounds)
      if (trim(tempStr) == 'none') then
         nCoarsenLevels = 0
         return
      end if

      ! Count commas to determine number of bounds
      nBounds = 1
      strLen = len_trim(tempStr)
      do i = 1, strLen
         if (tempStr(i:i) == ',') nBounds = nBounds + 1
      end do

      allocate(depthBounds(nBounds))

      ! Parse comma-separated values
      iStart = 1
      do i = 1, nBounds
         iEnd = index(tempStr(iStart:strLen), ',')
         if (iEnd == 0) then
            iEnd = strLen + 1
         else
            iEnd = iEnd + iStart - 1
         end if
         read(tempStr(iStart:iEnd-1), *) depthBounds(i)
         iStart = iEnd + 1
      end do

      nCoarsenLevels = nBounds - 1

   end subroutine ocn_init_fme_depth_coarsening!}}}

!***********************************************************************

   subroutine ocn_compute_fme_depth_coarsening(domain, timeLevel, err)!{{{

      integer, intent(in) :: timeLevel
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: statePool
      type(mpas_pool_type), pointer :: meshPool
      type(mpas_pool_type), pointer :: tracersPool
      type(mpas_pool_type), pointer :: amPool

      integer, pointer :: nCellsSolve, nVertLevels
      integer, pointer :: index_temperature, index_salinity
      integer, pointer :: nFmeDepthLevels
      integer :: iCell, k
      integer, dimension(:), pointer :: maxLevelCell

      real(kind=RKIND), dimension(:, :, :), pointer :: tracers
      real(kind=RKIND), dimension(:, :), pointer :: layerThickness
      real(kind=RKIND), dimension(:, :), pointer :: temperatureCoarsened
      real(kind=RKIND), dimension(:, :), pointer :: salinityCoarsened
      real(kind=RKIND), dimension(:, :), pointer :: velocityZonalCoarsened
      real(kind=RKIND), dimension(:, :), pointer :: velocityMeridionalCoarsened
      real(kind=RKIND), dimension(:, :), pointer :: layerThicknessCoarsened

      ! Work arrays for shr_vcoarsen_avg (single column)
      real(r8), allocatable :: coord_iface(:)
      real(r8), allocatable :: field_1d(:)
      real(r8), allocatable :: field_out(:)
      real(r8) :: depthTop

      err = 0

      if (nCoarsenLevels == 0) return

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'fmeDepthCoarseningAM', amPool)

         call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
         call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_dimension(block % dimensions, 'nVertLevels', nVertLevels)
         call mpas_pool_get_dimension(block % dimensions, 'nFmeDepthLevels', nFmeDepthLevels)

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
         allocate(field_out(nCoarsenLevels))

         do iCell = 1, nCellsSolve

            ! Build depth interface array from layer thicknesses
            coord_iface(1) = 0.0_r8
            do k = 1, maxLevelCell(iCell)
               coord_iface(k + 1) = coord_iface(k) + layerThickness(k, iCell)
            end do
            ! Fill remaining interfaces for unused levels
            do k = maxLevelCell(iCell) + 1, nVertLevels
               coord_iface(k + 1) = coord_iface(k)
            end do

            ! Temperature coarsening
            field_1d(1:nVertLevels) = tracers(index_temperature, 1:nVertLevels, iCell)
            call shr_vcoarsen_avg(field_1d, coord_iface, maxLevelCell(iCell), &
                 depthBounds, nCoarsenLevels, real(fillValue, r8), field_out)
            temperatureCoarsened(1:nCoarsenLevels, iCell) = field_out(1:nCoarsenLevels)

            ! Salinity coarsening
            field_1d(1:nVertLevels) = tracers(index_salinity, 1:nVertLevels, iCell)
            call shr_vcoarsen_avg(field_1d, coord_iface, maxLevelCell(iCell), &
                 depthBounds, nCoarsenLevels, real(fillValue, r8), field_out)
            salinityCoarsened(1:nCoarsenLevels, iCell) = field_out(1:nCoarsenLevels)

            ! Zonal velocity coarsening
            field_1d(1:nVertLevels) = velocityZonal(1:nVertLevels, iCell)
            call shr_vcoarsen_avg(field_1d, coord_iface, maxLevelCell(iCell), &
                 depthBounds, nCoarsenLevels, real(fillValue, r8), field_out)
            velocityZonalCoarsened(1:nCoarsenLevels, iCell) = field_out(1:nCoarsenLevels)

            ! Meridional velocity coarsening
            field_1d(1:nVertLevels) = velocityMeridional(1:nVertLevels, iCell)
            call shr_vcoarsen_avg(field_1d, coord_iface, maxLevelCell(iCell), &
                 depthBounds, nCoarsenLevels, real(fillValue, r8), field_out)
            velocityMeridionalCoarsened(1:nCoarsenLevels, iCell) = field_out(1:nCoarsenLevels)

            ! Layer thickness coarsening (sum of overlap, not average)
            do k = 1, nCoarsenLevels
               layerThicknessCoarsened(k, iCell) = 0.0_RKIND
            end do
            depthTop = 0.0_RKIND
            do k = 1, maxLevelCell(iCell)
               call accumulate_thickness(depthTop, layerThickness(k, iCell), &
                    nCoarsenLevels, layerThicknessCoarsened(:, iCell))
               depthTop = depthTop + layerThickness(k, iCell)
            end do
            do k = 1, nCoarsenLevels
               if (layerThicknessCoarsened(k, iCell) == 0.0_RKIND) then
                  layerThicknessCoarsened(k, iCell) = fillValue
               end if
            end do

            ! Fill unused levels with fill value
            do k = nCoarsenLevels + 1, nFmeDepthLevels
               temperatureCoarsened(k, iCell) = fillValue
               salinityCoarsened(k, iCell) = fillValue
               velocityZonalCoarsened(k, iCell) = fillValue
               velocityMeridionalCoarsened(k, iCell) = fillValue
               layerThicknessCoarsened(k, iCell) = fillValue
            end do

         end do

         deallocate(coord_iface, field_1d, field_out)

         block => block % next
      end do

   end subroutine ocn_compute_fme_depth_coarsening!}}}

!***********************************************************************

   subroutine accumulate_thickness(depthTop, thickness, nLevs, thickOut)!{{{
      real(kind=RKIND), intent(in) :: depthTop, thickness
      integer, intent(in) :: nLevs
      real(kind=RKIND), intent(inout) :: thickOut(nLevs)

      real(kind=RKIND) :: depthBot, overlapTop, overlapBot, overlap
      integer :: kOut

      depthBot = depthTop + thickness
      do kOut = 1, nLevs
         overlapTop = max(depthTop, depthBounds(kOut))
         overlapBot = min(depthBot, depthBounds(kOut + 1))
         overlap = max(0.0_RKIND, overlapBot - overlapTop)
         thickOut(kOut) = thickOut(kOut) + overlap
      end do

   end subroutine accumulate_thickness!}}}

!***********************************************************************

   subroutine ocn_restart_fme_depth_coarsening(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      err = 0

   end subroutine ocn_restart_fme_depth_coarsening!}}}

!***********************************************************************

   subroutine ocn_finalize_fme_depth_coarsening(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      err = 0

      if (allocated(depthBounds)) deallocate(depthBounds)

   end subroutine ocn_finalize_fme_depth_coarsening!}}}

end module ocn_fme_depth_coarsening

! vim: foldmethod=marker
