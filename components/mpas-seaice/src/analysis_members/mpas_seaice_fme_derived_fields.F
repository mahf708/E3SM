!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_fme_derived_fields
!
!> \brief MPAS sea ice analysis mode member: fme_derived_fields
!> \author FME Development Team
!> \date   April 2026
!> \details
!>  MPAS sea ice analysis mode member: fme_derived_fields
!>  Computes derived fields for Full Model Emulation (FME) by
!>  aggregating category-level tracers into cell-level quantities.
!>
!-----------------------------------------------------------------------

module seaice_fme_derived_fields

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_timekeeping
   use mpas_stream_manager
   use mpas_kind_types

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

   public :: seaice_bootstrap_fme_derived_fields, &
             seaice_init_fme_derived_fields, &
             seaice_precompute_fme_derived_fields, &
             seaice_compute_fme_derived_fields, &
             seaice_restart_fme_derived_fields, &
             seaice_finalize_fme_derived_fields

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine seaice_bootstrap_fme_derived_fields
!
!> \brief   Bootstrap MPAS-Seaice analysis member
!> \author  FME Development Team
!> \date    April 2026
!> \details
!>  This routine conducts all bootstraps required for the
!>  MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_bootstrap_fme_derived_fields(domain, instance, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      character(len=*), intent(in) :: instance

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: domain

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

   end subroutine seaice_bootstrap_fme_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_init_fme_derived_fields
!
!> \brief   Initialize MPAS-Seaice analysis member
!> \author  FME Development Team
!> \date    April 2026
!> \details
!>  This routine conducts all initializations required for the
!>  MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_init_fme_derived_fields(domain, instance, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      character(len=*), intent(in) :: instance

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: domain

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

   end subroutine seaice_init_fme_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_precompute_fme_derived_fields
!
!> \brief   Precompute MPAS-Seaice analysis member
!> \author  FME Development Team
!> \date    April 2026
!> \details
!>  This routine conducts all pre-computation required for this
!>  MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_precompute_fme_derived_fields(domain, instance, timeLevel, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      character(len=*), intent(in) :: instance

      integer, intent(in) :: timeLevel

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: domain

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

   end subroutine seaice_precompute_fme_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_compute_fme_derived_fields
!
!> \brief   Compute MPAS-Seaice analysis member
!> \author  FME Development Team
!> \date    April 2026
!> \details
!>  This routine conducts all computation required for this
!>  MPAS-Seaice analysis member. It aggregates category-level sea ice
!>  tracers into cell-level derived fields for FME output.
!
!-----------------------------------------------------------------------

   subroutine seaice_compute_fme_derived_fields(domain, instance, timeLevel, err)!{{{

      use seaice_mesh, only: seaice_latlon_vector_rotation_backward

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      character(len=*), intent(in) :: instance

      integer, intent(in) :: timeLevel

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: domain

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

      type (block_type), pointer :: &
           block

      type (mpas_pool_type), pointer :: &
           tracersPool, &
           meshPool, &
           velocitySolverPool, &
           amPool

      real(kind=RKIND), dimension(:,:,:), pointer :: &
           iceAreaCategory, &
           iceVolumeCategory, &
           snowVolumeCategory, &
           surfaceTemperature

      real(kind=RKIND), dimension(:), pointer :: &
           iceAreaTotal, &
           iceVolumeTotal, &
           snowVolumeTotal, &
           iceThicknessMean, &
           surfaceTemperatureMean

      ! air stress rotation variables
      real(kind=RKIND), dimension(:), pointer :: &
           airStressCellU, &
           airStressCellV, &
           airStressZonal, &
           airStressMeridional, &
           latCell, &
           lonCell, &
           xCell, &
           yCell, &
           zCell

      real(kind=RKIND), pointer :: &
           sphereRadius

      logical, pointer :: &
           rotateCartesianGrid

      integer, pointer :: &
           nCellsSolve, &
           nCategories

      integer :: &
           iCell, &
           iCategory

      real(kind=RKIND) :: &
           areaSum, &
           tempSum

      err = 0

      ! get configs
      call MPAS_pool_get_config(domain % configs, 'config_rotate_cartesian_grid', rotateCartesianGrid)

      block => domain % blocklist
      do while (associated(block))

         call mpas_pool_get_subpool(block % structs, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(block % structs, 'velocity_solver', velocitySolverPool)
         call mpas_pool_get_subpool(block % structs, 'fmeSeaiceDerivedFieldsAM', amPool)

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

         ! air stress rotation arrays
         call MPAS_pool_get_config(meshPool, 'sphere_radius', sphereRadius)
         call MPAS_pool_get_array(meshPool, 'latCell', latCell)
         call MPAS_pool_get_array(meshPool, 'lonCell', lonCell)
         call MPAS_pool_get_array(meshPool, 'xCell', xCell)
         call MPAS_pool_get_array(meshPool, 'yCell', yCell)
         call MPAS_pool_get_array(meshPool, 'zCell', zCell)

         call MPAS_pool_get_array(velocitySolverPool, 'airStressCellU', airStressCellU)
         call MPAS_pool_get_array(velocitySolverPool, 'airStressCellV', airStressCellV)

         call MPAS_pool_get_array(amPool, 'airStressZonal', airStressZonal)
         call MPAS_pool_get_array(amPool, 'airStressMeridional', airStressMeridional)

         do iCell = 1, nCellsSolve

            ! Sum over categories
            areaSum = 0.0_RKIND
            iceVolumeTotal(iCell) = 0.0_RKIND
            snowVolumeTotal(iCell) = 0.0_RKIND
            tempSum = 0.0_RKIND

            do iCategory = 1, nCategories
               areaSum = areaSum + iceAreaCategory(1, iCategory, iCell)
               iceVolumeTotal(iCell) = iceVolumeTotal(iCell) + iceVolumeCategory(1, iCategory, iCell)
               snowVolumeTotal(iCell) = snowVolumeTotal(iCell) + snowVolumeCategory(1, iCategory, iCell)
               tempSum = tempSum + iceAreaCategory(1, iCategory, iCell) * surfaceTemperature(1, iCategory, iCell)
            end do

            iceAreaTotal(iCell) = areaSum

            ! Area-weighted mean ice thickness
            if (areaSum > 0.0_RKIND) then
               iceThicknessMean(iCell) = iceVolumeTotal(iCell) / areaSum
               surfaceTemperatureMean(iCell) = tempSum / areaSum
            else
               iceThicknessMean(iCell) = 0.0_RKIND
               surfaceTemperatureMean(iCell) = 0.0_RKIND
            end if

            ! Rotate air stress from model coordinates to geographical coordinates
            call seaice_latlon_vector_rotation_backward(&
                 airStressZonal(iCell), &
                 airStressMeridional(iCell), &
                 airStressCellU(iCell), &
                 airStressCellV(iCell), &
                 latCell(iCell), &
                 lonCell(iCell), &
                 xCell(iCell), &
                 yCell(iCell), &
                 zCell(iCell), &
                 sphereRadius, &
                 rotateCartesianGrid)

         end do ! iCell

         block => block % next
      end do

   end subroutine seaice_compute_fme_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_restart_fme_derived_fields
!
!> \brief   Save restart for MPAS-Seaice analysis member
!> \author  FME Development Team
!> \date    April 2026
!> \details
!>  This routine conducts computation required to save a restart state
!>  for the MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_restart_fme_derived_fields(domain, instance, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      character(len=*), intent(in) :: instance

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: domain

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

   end subroutine seaice_restart_fme_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_finalize_fme_derived_fields
!
!> \brief   Finalize MPAS-Seaice analysis member
!> \author  FME Development Team
!> \date    April 2026
!> \details
!>  This routine conducts all finalizations required for this
!>  MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_finalize_fme_derived_fields(domain, instance, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      character(len=*), intent(in) :: instance

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: domain

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

   end subroutine seaice_finalize_fme_derived_fields!}}}

end module seaice_fme_derived_fields

! vim: foldmethod=marker
