! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_ace_depth_coarsening
!
!> \brief MPAS ocean analysis mode member: ACE depth coarsening
!> \author ACE Training Data Support
!> \date   March 2026
!> \details
!>  Computes layer-thickness-weighted vertical averages of tracer and
!>  velocity fields onto a coarsened set of depth levels defined by
!>  user-specified depth boundaries, for ACE training data.
!>
!-----------------------------------------------------------------------

module ocn_ace_depth_coarsening

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_timekeeping
   use mpas_stream_manager
   use mpas_kind_types

   use ocn_constants
   use ocn_diagnostics_variables

   implicit none
   private
   save

   !--------------------------------------------------------------------
   !
   ! Public parameters
   !
   !--------------------------------------------------------------------

   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: ocn_init_ace_depth_coarsening, &
             ocn_compute_ace_depth_coarsening, &
             ocn_restart_ace_depth_coarsening, &
             ocn_finalize_ace_depth_coarsening

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   integer :: nCoarsenLevels = 0
   integer :: nCoarsenFields = 0
   real(kind=RKIND), allocatable :: depthBounds(:)
   character(len=64), allocatable :: coarsenFieldNames(:)

   real(kind=RKIND), parameter :: fillValue = 1.0E34_RKIND

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_ace_depth_coarsening
!
!> \brief   Initialize MPAS-Ocean analysis member
!> \author  ACE Training Data Support
!> \date    March 2026
!> \details
!>  This routine conducts all initializations required for the
!>  MPAS-Ocean analysis member. It parses the comma-separated
!>  depth_bounds and fields configuration strings.
!
!-----------------------------------------------------------------------

   subroutine ocn_init_ace_depth_coarsening(domain, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type(domain_type), intent(inout) :: domain

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      character(len=StrKIND), pointer :: config_AM_aceDepthCoarsening_depth_bounds
      character(len=StrKIND), pointer :: config_AM_aceDepthCoarsening_fields
      character(len=StrKIND) :: tempStr
      integer :: nBounds, nFields, i, iStart, iEnd, strLen

      err = 0

      call mpas_pool_get_config(domain % configs, &
           'config_AM_aceDepthCoarsening_depth_bounds', &
           config_AM_aceDepthCoarsening_depth_bounds)
      call mpas_pool_get_config(domain % configs, &
           'config_AM_aceDepthCoarsening_fields', &
           config_AM_aceDepthCoarsening_fields)

      ! Parse depth bounds
      tempStr = trim(config_AM_aceDepthCoarsening_depth_bounds)
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

      ! Parse field names
      tempStr = trim(config_AM_aceDepthCoarsening_fields)
      if (trim(tempStr) == 'none') then
         nCoarsenFields = 0
         return
      end if

      nFields = 1
      strLen = len_trim(tempStr)
      do i = 1, strLen
         if (tempStr(i:i) == ',') nFields = nFields + 1
      end do

      nCoarsenFields = nFields
      allocate(coarsenFieldNames(nFields))

      iStart = 1
      do i = 1, nFields
         iEnd = index(tempStr(iStart:strLen), ',')
         if (iEnd == 0) then
            iEnd = strLen + 1
         else
            iEnd = iEnd + iStart - 1
         end if
         coarsenFieldNames(i) = adjustl(tempStr(iStart:iEnd-1))
         iStart = iEnd + 1
      end do

   end subroutine ocn_init_ace_depth_coarsening!}}}

!***********************************************************************
!
!  routine ocn_compute_ace_depth_coarsening
!
!> \brief   Compute MPAS-Ocean analysis member
!> \author  ACE Training Data Support
!> \date    March 2026
!> \details
!>  This routine conducts all computation required for this
!>  MPAS-Ocean analysis member. It computes layer-thickness-weighted
!>  vertical averages onto coarsened depth levels.
!
!-----------------------------------------------------------------------

   subroutine ocn_compute_ace_depth_coarsening(domain, timeLevel, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      integer, intent(in) :: timeLevel

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type(domain_type), intent(inout) :: domain

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: statePool
      type(mpas_pool_type), pointer :: meshPool
      type(mpas_pool_type), pointer :: tracersPool
      type(mpas_pool_type), pointer :: amPool

      integer, pointer :: nCellsSolve, nVertLevels
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

      real(kind=RKIND) :: depthTop, depthBot, overlapTop, overlapBot, overlap
      real(kind=RKIND) :: numerator, denominator

      integer, pointer :: nAceDepthLevels

      err = 0

      if (nCoarsenLevels == 0) return

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'aceDepthCoarseningAM', amPool)

         call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
         call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_dimension(block % dimensions, 'nVertLevels', nVertLevels)
         call mpas_pool_get_dimension(block % dimensions, 'nAceDepthLevels', nAceDepthLevels)

         call mpas_pool_get_array(tracersPool, 'activeTracers', tracers, timeLevel)
         call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, timeLevel)
         call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)

         call mpas_pool_get_array(amPool, 'temperatureCoarsened', temperatureCoarsened)
         call mpas_pool_get_array(amPool, 'salinityCoarsened', salinityCoarsened)
         call mpas_pool_get_array(amPool, 'velocityZonalCoarsened', velocityZonalCoarsened)
         call mpas_pool_get_array(amPool, 'velocityMeridionalCoarsened', velocityMeridionalCoarsened)
         call mpas_pool_get_array(amPool, 'layerThicknessCoarsened', layerThicknessCoarsened)

         do iCell = 1, nCellsSolve

            ! Temperature coarsening
            do kOut = 1, nCoarsenLevels
               numerator = 0.0_RKIND
               denominator = 0.0_RKIND
               depthTop = 0.0_RKIND
               do k = 1, maxLevelCell(iCell)
                  depthBot = depthTop + layerThickness(k, iCell)
                  overlapTop = max(depthTop, depthBounds(kOut))
                  overlapBot = min(depthBot, depthBounds(kOut+1))
                  overlap = max(0.0_RKIND, overlapBot - overlapTop)
                  if (overlap > 0.0_RKIND) then
                     numerator = numerator + tracers(index_temperature, k, iCell) * overlap
                     denominator = denominator + overlap
                  end if
                  depthTop = depthBot
               end do
               if (denominator > 0.0_RKIND) then
                  temperatureCoarsened(kOut, iCell) = numerator / denominator
               else
                  temperatureCoarsened(kOut, iCell) = fillValue
               end if
            end do

            ! Salinity coarsening
            do kOut = 1, nCoarsenLevels
               numerator = 0.0_RKIND
               denominator = 0.0_RKIND
               depthTop = 0.0_RKIND
               do k = 1, maxLevelCell(iCell)
                  depthBot = depthTop + layerThickness(k, iCell)
                  overlapTop = max(depthTop, depthBounds(kOut))
                  overlapBot = min(depthBot, depthBounds(kOut+1))
                  overlap = max(0.0_RKIND, overlapBot - overlapTop)
                  if (overlap > 0.0_RKIND) then
                     numerator = numerator + tracers(index_salinity, k, iCell) * overlap
                     denominator = denominator + overlap
                  end if
                  depthTop = depthBot
               end do
               if (denominator > 0.0_RKIND) then
                  salinityCoarsened(kOut, iCell) = numerator / denominator
               else
                  salinityCoarsened(kOut, iCell) = fillValue
               end if
            end do

            ! Zonal velocity coarsening
            do kOut = 1, nCoarsenLevels
               numerator = 0.0_RKIND
               denominator = 0.0_RKIND
               depthTop = 0.0_RKIND
               do k = 1, maxLevelCell(iCell)
                  depthBot = depthTop + layerThickness(k, iCell)
                  overlapTop = max(depthTop, depthBounds(kOut))
                  overlapBot = min(depthBot, depthBounds(kOut+1))
                  overlap = max(0.0_RKIND, overlapBot - overlapTop)
                  if (overlap > 0.0_RKIND) then
                     numerator = numerator + velocityZonal(k, iCell) * overlap
                     denominator = denominator + overlap
                  end if
                  depthTop = depthBot
               end do
               if (denominator > 0.0_RKIND) then
                  velocityZonalCoarsened(kOut, iCell) = numerator / denominator
               else
                  velocityZonalCoarsened(kOut, iCell) = fillValue
               end if
            end do

            ! Meridional velocity coarsening
            do kOut = 1, nCoarsenLevels
               numerator = 0.0_RKIND
               denominator = 0.0_RKIND
               depthTop = 0.0_RKIND
               do k = 1, maxLevelCell(iCell)
                  depthBot = depthTop + layerThickness(k, iCell)
                  overlapTop = max(depthTop, depthBounds(kOut))
                  overlapBot = min(depthBot, depthBounds(kOut+1))
                  overlap = max(0.0_RKIND, overlapBot - overlapTop)
                  if (overlap > 0.0_RKIND) then
                     numerator = numerator + velocityMeridional(k, iCell) * overlap
                     denominator = denominator + overlap
                  end if
                  depthTop = depthBot
               end do
               if (denominator > 0.0_RKIND) then
                  velocityMeridionalCoarsened(kOut, iCell) = numerator / denominator
               else
                  velocityMeridionalCoarsened(kOut, iCell) = fillValue
               end if
            end do

            ! Layer thickness coarsening (sum, not average)
            do kOut = 1, nCoarsenLevels
               denominator = 0.0_RKIND
               depthTop = 0.0_RKIND
               do k = 1, maxLevelCell(iCell)
                  depthBot = depthTop + layerThickness(k, iCell)
                  overlapTop = max(depthTop, depthBounds(kOut))
                  overlapBot = min(depthBot, depthBounds(kOut+1))
                  overlap = max(0.0_RKIND, overlapBot - overlapTop)
                  denominator = denominator + overlap
                  depthTop = depthBot
               end do
               if (denominator > 0.0_RKIND) then
                  layerThicknessCoarsened(kOut, iCell) = denominator
               else
                  layerThicknessCoarsened(kOut, iCell) = fillValue
               end if
            end do

            ! Fill unused levels with fill value
            do kOut = nCoarsenLevels + 1, nAceDepthLevels
               temperatureCoarsened(kOut, iCell) = fillValue
               salinityCoarsened(kOut, iCell) = fillValue
               velocityZonalCoarsened(kOut, iCell) = fillValue
               velocityMeridionalCoarsened(kOut, iCell) = fillValue
               layerThicknessCoarsened(kOut, iCell) = fillValue
            end do

         end do

         block => block % next
      end do

   end subroutine ocn_compute_ace_depth_coarsening!}}}

!***********************************************************************
!
!  routine ocn_restart_ace_depth_coarsening
!
!> \brief   Save restart for MPAS-Ocean analysis member
!> \author  ACE Training Data Support
!> \date    March 2026
!> \details
!>  This routine conducts computation required to save a restart state
!>  for the MPAS-Ocean analysis member.
!
!-----------------------------------------------------------------------

   subroutine ocn_restart_ace_depth_coarsening(domain, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type(domain_type), intent(inout) :: domain

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      err = 0

   end subroutine ocn_restart_ace_depth_coarsening!}}}

!***********************************************************************
!
!  routine ocn_finalize_ace_depth_coarsening
!
!> \brief   Finalize MPAS-Ocean analysis member
!> \author  ACE Training Data Support
!> \date    March 2026
!> \details
!>  This routine conducts all finalizations required for this
!>  MPAS-Ocean analysis member.
!
!-----------------------------------------------------------------------

   subroutine ocn_finalize_ace_depth_coarsening(domain, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type(domain_type), intent(inout) :: domain

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      err = 0

      if (allocated(depthBounds)) deallocate(depthBounds)
      if (allocated(coarsenFieldNames)) deallocate(coarsenFieldNames)

   end subroutine ocn_finalize_ace_depth_coarsening!}}}

end module ocn_ace_depth_coarsening

! vim: foldmethod=marker
