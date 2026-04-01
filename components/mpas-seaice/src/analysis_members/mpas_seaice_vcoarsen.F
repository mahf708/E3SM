! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_vcoarsen
!
!> \brief MPAS sea ice analysis mode member: category aggregation
!> \author E3SM Vertical Coarsening Support
!> \date   April 2026
!> \details
!>  Aggregates sea ice fields across thickness categories using the
!>  shared shr_vcoarsen_mod routines. Computes total ice area, volume,
!>  snow volume (sums), and area-weighted mean surface temperature.
!>
!-----------------------------------------------------------------------

module seaice_vcoarsen

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_timekeeping
   use mpas_stream_manager
   use mpas_kind_types

   use shr_kind_mod, only: r8 => shr_kind_r8

   implicit none
   private
   save

   public :: seaice_bootstrap_vcoarsen, &
             seaice_init_vcoarsen, &
             seaice_precompute_vcoarsen, &
             seaice_compute_vcoarsen, &
             seaice_restart_vcoarsen, &
             seaice_finalize_vcoarsen

   real(kind=RKIND), parameter :: fillValue = 0.0_RKIND

!***********************************************************************

contains

!***********************************************************************

   subroutine seaice_bootstrap_vcoarsen(domain, instance, err)!{{{
      character(len=*), intent(in) :: instance
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err
      err = 0
   end subroutine seaice_bootstrap_vcoarsen!}}}

!***********************************************************************

   subroutine seaice_init_vcoarsen(domain, instance, err)!{{{
      character(len=*), intent(in) :: instance
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err
      err = 0
   end subroutine seaice_init_vcoarsen!}}}

!***********************************************************************

   subroutine seaice_precompute_vcoarsen(domain, instance, timeLevel, err)!{{{
      character(len=*), intent(in) :: instance
      integer, intent(in) :: timeLevel
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err
      err = 0
   end subroutine seaice_precompute_vcoarsen!}}}

!***********************************************************************
!
!  routine seaice_compute_vcoarsen
!
!> \brief   Compute category aggregation using shared utility
!> \details
!>  Aggregates sea ice tracer fields across thickness categories.
!>  Uses shr_vcoarsen_cat_sum for total area/volume/snow and
!>  shr_vcoarsen_cat_wgtavg for area-weighted mean temperature.
!
!-----------------------------------------------------------------------

   subroutine seaice_compute_vcoarsen(domain, instance, timeLevel, err)!{{{

      use shr_vcoarsen_mod, only: shr_vcoarsen_cat_sum, shr_vcoarsen_cat_wgtavg

      character(len=*), intent(in) :: instance
      integer, intent(in) :: timeLevel
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: tracersPool, amPool

      integer, pointer :: nCellsSolve, nCategories
      integer :: iCell, iCat

      real(kind=RKIND), dimension(:,:,:), pointer :: &
           iceAreaCategory, iceVolumeCategory, snowVolumeCategory, surfaceTemperature

      real(kind=RKIND), dimension(:), pointer :: &
           iceAreaTotal, iceVolumeTotal, snowVolumeTotal, &
           iceThicknessMean, surfaceTemperatureMean

      ! Working arrays for shared utility calls
      real(r8), allocatable :: field_2d(:,:)    ! (nCells, nCat)
      real(r8), allocatable :: weights_2d(:,:)  ! (nCells, nCat)
      real(r8), allocatable :: result_1d(:)     ! (nCells)

      err = 0

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'vcoarsenAM', amPool)

         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_dimension(block % dimensions, 'nCategories', nCategories)

         call mpas_pool_get_array(tracersPool, 'iceAreaCategory', iceAreaCategory, timeLevel)
         call mpas_pool_get_array(tracersPool, 'iceVolumeCategory', iceVolumeCategory, timeLevel)
         call mpas_pool_get_array(tracersPool, 'snowVolumeCategory', snowVolumeCategory, timeLevel)
         call mpas_pool_get_array(tracersPool, 'surfaceTemperature', surfaceTemperature, timeLevel)

         call mpas_pool_get_array(amPool, 'iceAreaTotal', iceAreaTotal)
         call mpas_pool_get_array(amPool, 'iceVolumeTotal', iceVolumeTotal)
         call mpas_pool_get_array(amPool, 'snowVolumeTotal', snowVolumeTotal)
         call mpas_pool_get_array(amPool, 'iceThicknessMean', iceThicknessMean)
         call mpas_pool_get_array(amPool, 'surfaceTemperatureMean', surfaceTemperatureMean)

         allocate(field_2d(nCellsSolve, nCategories))
         allocate(weights_2d(nCellsSolve, nCategories))
         allocate(result_1d(nCellsSolve))

         ! --- Total ice area (sum over categories) ---
         do iCat = 1, nCategories
            do iCell = 1, nCellsSolve
               field_2d(iCell, iCat) = iceAreaCategory(1, iCat, iCell)
            end do
         end do
         call shr_vcoarsen_cat_sum(field_2d, nCellsSolve, nCategories, result_1d)
         iceAreaTotal(1:nCellsSolve) = result_1d(1:nCellsSolve)

         ! --- Total ice volume (sum over categories) ---
         do iCat = 1, nCategories
            do iCell = 1, nCellsSolve
               field_2d(iCell, iCat) = iceVolumeCategory(1, iCat, iCell)
            end do
         end do
         call shr_vcoarsen_cat_sum(field_2d, nCellsSolve, nCategories, result_1d)
         iceVolumeTotal(1:nCellsSolve) = result_1d(1:nCellsSolve)

         ! --- Total snow volume (sum over categories) ---
         do iCat = 1, nCategories
            do iCell = 1, nCellsSolve
               field_2d(iCell, iCat) = snowVolumeCategory(1, iCat, iCell)
            end do
         end do
         call shr_vcoarsen_cat_sum(field_2d, nCellsSolve, nCategories, result_1d)
         snowVolumeTotal(1:nCellsSolve) = result_1d(1:nCellsSolve)

         ! --- Mean ice thickness = total volume / total area ---
         do iCell = 1, nCellsSolve
            if (iceAreaTotal(iCell) > 0.0_RKIND) then
               iceThicknessMean(iCell) = iceVolumeTotal(iCell) / iceAreaTotal(iCell)
            else
               iceThicknessMean(iCell) = fillValue
            end if
         end do

         ! --- Area-weighted mean surface temperature ---
         ! weights = ice area per category; field = surface temperature
         do iCat = 1, nCategories
            do iCell = 1, nCellsSolve
               weights_2d(iCell, iCat) = iceAreaCategory(1, iCat, iCell)
               field_2d(iCell, iCat) = surfaceTemperature(1, iCat, iCell)
            end do
         end do
         call shr_vcoarsen_cat_wgtavg(field_2d, weights_2d, nCellsSolve, nCategories, &
              real(fillValue, r8), result_1d)
         surfaceTemperatureMean(1:nCellsSolve) = result_1d(1:nCellsSolve)

         deallocate(field_2d, weights_2d, result_1d)

         block => block % next
      end do

   end subroutine seaice_compute_vcoarsen!}}}

!***********************************************************************

   subroutine seaice_restart_vcoarsen(domain, instance, err)!{{{
      character(len=*), intent(in) :: instance
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err
      err = 0
   end subroutine seaice_restart_vcoarsen!}}}

!***********************************************************************

   subroutine seaice_finalize_vcoarsen(domain, instance, err)!{{{
      character(len=*), intent(in) :: instance
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err
      err = 0
   end subroutine seaice_finalize_vcoarsen!}}}

end module seaice_vcoarsen

! vim: foldmethod=marker
