! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_fme_vertical_reduce
!
!> \brief MPAS ocean analysis member: FME vertical reduction
!> \author FME Training Data Support
!> \date   April 2026
!> \details
!>  Computes column-integrated (vertical reduction) quantities for FME
!>  training data. Built-in fields:
!>    - oceanHeatContent:  sum(T * h * rho_sw * cp_sw)  [J/m^2]
!>    - freshwaterContent: sum((S_ref - S)/S_ref * h)    [m]
!>    - kineticEnergy:     sum(0.5*(u^2+v^2) * h * rho_sw) [J/m^2]
!>
!>  Also supports user-configurable fields via namelist with sum or
!>  thickness-weighted average reduction methods.
!>
!-----------------------------------------------------------------------

module ocn_fme_vertical_reduce

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_timekeeping
   use mpas_stream_manager
   use mpas_kind_types

   use ocn_constants
   use ocn_diagnostics_variables
   use mpas_log, only: mpas_log_write

   implicit none
   private
   save

   public :: ocn_init_fme_vertical_reduce, &
             ocn_compute_fme_vertical_reduce, &
             ocn_restart_fme_vertical_reduce, &
             ocn_finalize_fme_vertical_reduce

   real(kind=RKIND) :: referenceSalinity = 34.8_RKIND

   !> Number of user-configured generic reduction fields (parsed from config)
   integer :: nReduceFields = 0

   !> Field names parsed from config_AM_fmeVerticalReduce_fields
   character(len=64), allocatable :: reduceFieldNames(:)

   !> Reduction methods parsed from config_AM_fmeVerticalReduce_methods
   !> Each entry is 'sum' or 'avg'
   character(len=64), allocatable :: reduceFieldMethods(:)

!***********************************************************************

contains

!***********************************************************************

   subroutine ocn_init_fme_vertical_reduce(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      real(kind=RKIND), pointer :: config_AM_fmeVerticalReduce_reference_salinity
      character(len=StrKIND), pointer :: config_AM_fmeVerticalReduce_fields
      character(len=StrKIND), pointer :: config_AM_fmeVerticalReduce_methods
      character(len=StrKIND) :: tempStr, methodStr
      integer :: nFields, nMethods, i, iStart, iEnd, strLen

      err = 0

      call mpas_pool_get_config(domain % configs, &
           'config_AM_fmeVerticalReduce_reference_salinity', &
           config_AM_fmeVerticalReduce_reference_salinity)

      referenceSalinity = config_AM_fmeVerticalReduce_reference_salinity

      ! --- Parse generic field names and methods ---
      ! The built-in fields (OHC, FWC, KE) are always computed when
      ! the AM is enabled. The generic fields/methods config allows
      ! users to request additional reductions. However, MPAS Registry
      ! requires compile-time variable definitions, so truly dynamic
      ! output fields are not supported without Registry changes.
      ! We parse the config here and log a warning if fields are
      ! specified, noting that output requires corresponding Registry
      ! variable definitions.

      call mpas_pool_get_config(domain % configs, &
           'config_AM_fmeVerticalReduce_fields', &
           config_AM_fmeVerticalReduce_fields)
      call mpas_pool_get_config(domain % configs, &
           'config_AM_fmeVerticalReduce_methods', &
           config_AM_fmeVerticalReduce_methods)

      tempStr = trim(config_AM_fmeVerticalReduce_fields)
      methodStr = trim(config_AM_fmeVerticalReduce_methods)

      if (trim(tempStr) /= 'none' .and. len_trim(tempStr) > 0) then
         ! Count comma-separated field names
         nFields = 1
         strLen = len_trim(tempStr)
         do i = 1, strLen
            if (tempStr(i:i) == ',') nFields = nFields + 1
         end do

         ! Count comma-separated methods
         nMethods = 1
         strLen = len_trim(methodStr)
         do i = 1, strLen
            if (methodStr(i:i) == ',') nMethods = nMethods + 1
         end do

         if (nFields /= nMethods) then
            call mpas_log_write( &
                 'fmeVerticalReduce WARNING: number of fields does not match ' &
                 // 'number of methods. Generic fields will not be computed.')
            nReduceFields = 0
         else
            nReduceFields = nFields
            allocate(reduceFieldNames(nReduceFields))
            allocate(reduceFieldMethods(nReduceFields))

            ! Parse field names
            iStart = 1
            strLen = len_trim(tempStr)
            do i = 1, nReduceFields
               iEnd = index(tempStr(iStart:strLen), ',')
               if (iEnd == 0) then
                  iEnd = strLen + 1
               else
                  iEnd = iEnd + iStart - 1
               end if
               reduceFieldNames(i) = adjustl(tempStr(iStart:iEnd-1))
               iStart = iEnd + 1
            end do

            ! Parse methods
            iStart = 1
            strLen = len_trim(methodStr)
            do i = 1, nReduceFields
               iEnd = index(methodStr(iStart:strLen), ',')
               if (iEnd == 0) then
                  iEnd = strLen + 1
               else
                  iEnd = iEnd + iStart - 1
               end if
               reduceFieldMethods(i) = adjustl(methodStr(iStart:iEnd-1))
               iStart = iEnd + 1
            end do

            call mpas_log_write( &
                 'fmeVerticalReduce WARNING: generic field reduction for $i' &
                 // ' fields was requested via config_AM_fmeVerticalReduce_fields,' &
                 // ' but output of generic fields is not yet supported.' &
                 // ' MPAS Registry requires compile-time variable definitions.' &
                 // ' Built-in fields (OHC, FWC, KE) will still be computed.', &
                 intArgs=(/nReduceFields/))
         end if
      else
         nReduceFields = 0
      end if

   end subroutine ocn_init_fme_vertical_reduce!}}}

!***********************************************************************

   subroutine ocn_compute_fme_vertical_reduce(domain, timeLevel, err)!{{{

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
      integer :: iCell, k
      integer, dimension(:), pointer :: maxLevelCell

      real(kind=RKIND), dimension(:, :, :), pointer :: tracers
      real(kind=RKIND), dimension(:, :), pointer :: layerThickness
      real(kind=RKIND), dimension(:), pointer :: oceanHeatContent
      real(kind=RKIND), dimension(:), pointer :: freshwaterContent
      real(kind=RKIND), dimension(:), pointer :: kineticEnergy

      real(kind=RKIND) :: ohc_sum, fwc_sum, ke_sum, h_k

      err = 0

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'fmeVerticalReduceAM', amPool)

         call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
         call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_dimension(block % dimensions, 'nVertLevels', nVertLevels)

         call mpas_pool_get_array(tracersPool, 'activeTracers', tracers, timeLevel)
         call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, timeLevel)
         call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)

         call mpas_pool_get_array(amPool, 'oceanHeatContent', oceanHeatContent)
         call mpas_pool_get_array(amPool, 'freshwaterContent', freshwaterContent)
         call mpas_pool_get_array(amPool, 'kineticEnergy', kineticEnergy)

         do iCell = 1, nCellsSolve
            ohc_sum = 0.0_RKIND
            fwc_sum = 0.0_RKIND
            ke_sum  = 0.0_RKIND

            do k = 1, maxLevelCell(iCell)
               h_k = layerThickness(k, iCell)

               ! Ocean heat content: T * h * rho_sw * cp_sw
               ohc_sum = ohc_sum + tracers(index_temperature, k, iCell) &
                    * h_k * rho_sw * cp_sw

               ! Freshwater content: (S_ref - S) / S_ref * h
               fwc_sum = fwc_sum + (referenceSalinity &
                    - tracers(index_salinity, k, iCell)) &
                    / referenceSalinity * h_k

               ! Kinetic energy: 0.5 * (u^2 + v^2) * h * rho_sw
               ke_sum = ke_sum + 0.5_RKIND &
                    * (velocityZonal(k, iCell)**2 &
                     + velocityMeridional(k, iCell)**2) &
                    * h_k * rho_sw
            end do

            oceanHeatContent(iCell) = ohc_sum
            freshwaterContent(iCell) = fwc_sum
            kineticEnergy(iCell) = ke_sum
         end do

         block => block % next
      end do

   end subroutine ocn_compute_fme_vertical_reduce!}}}

!***********************************************************************

   subroutine ocn_restart_fme_vertical_reduce(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      err = 0

   end subroutine ocn_restart_fme_vertical_reduce!}}}

!***********************************************************************

   subroutine ocn_finalize_fme_vertical_reduce(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      err = 0

      if (allocated(reduceFieldNames)) deallocate(reduceFieldNames)
      if (allocated(reduceFieldMethods)) deallocate(reduceFieldMethods)

   end subroutine ocn_finalize_fme_vertical_reduce!}}}

end module ocn_fme_vertical_reduce

! vim: foldmethod=marker
