!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_ace_derived_fields
!
!> \brief MPAS-Seaice analysis mode member: ace_derived_fields
!> \author ACE Training Data
!> \date   2026/03/28
!> \details
!>  MPAS-Seaice analysis mode member: ace_derived_fields
!>  Computes category-aggregated sea ice fields for ACE training data.
!>
!-----------------------------------------------------------------------

module seaice_ace_derived_fields

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_timekeeping
   use mpas_stream_manager

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

   public :: seaice_bootstrap_ace_derived_fields, &
             seaice_init_ace_derived_fields, &
             seaice_precompute_ace_derived_fields, &
             seaice_compute_ace_derived_fields, &
             seaice_restart_ace_derived_fields, &
             seaice_finalize_ace_derived_fields

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine seaice_bootstrap_ace_derived_fields
!
!> \brief   Bootstrap MPAS-Seaice analysis member
!> \author  ACE Training Data
!> \date    2026/03/28
!> \details
!>  This routine conducts all bootstraps required for the
!>  MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_bootstrap_ace_derived_fields(domain, instance, err)!{{{

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

   end subroutine seaice_bootstrap_ace_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_init_ace_derived_fields
!
!> \brief   Initialize MPAS-Seaice analysis member
!> \author  ACE Training Data
!> \date    2026/03/28
!> \details
!>  This routine conducts all initializations required for the
!>  MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_init_ace_derived_fields(domain, instance, err)!{{{

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

   end subroutine seaice_init_ace_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_precompute_ace_derived_fields
!
!> \brief   Precompute MPAS-Seaice analysis member
!> \author  ACE Training Data
!> \date    2026/03/28
!> \details
!>  This routine conducts all precomputation required for this
!>  MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_precompute_ace_derived_fields(domain, instance, timeLevel, err)!{{{

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

   end subroutine seaice_precompute_ace_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_compute_ace_derived_fields
!
!> \brief   Compute MPAS-Seaice analysis member
!> \author  ACE Training Data
!> \date    2026/03/28
!> \details
!>  This routine computes category-aggregated sea ice fields for
!>  ACE training data support.
!
!-----------------------------------------------------------------------

   subroutine seaice_compute_ace_derived_fields(domain, instance, timeLevel, err)!{{{

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

      type (dm_info) :: dminfo
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: tracersPool
      type (mpas_pool_type), pointer :: aceSeaiceDerivedFieldsAMPool

      real (kind=RKIND), dimension(:,:,:), pointer :: &
           iceAreaCategory, &
           iceVolumeCategory, &
           snowVolumeCategory, &
           surfaceTemperature

      real (kind=RKIND), dimension(:), pointer :: &
           iceAreaTotal, &
           iceVolumeTotal, &
           snowVolumeTotal, &
           iceThicknessMean, &
           surfaceTemperatureMean

      integer, pointer :: &
           nCellsSolve, &
           nCategories

      integer :: &
           iCell, &
           iCategory

      real (kind=RKIND) :: &
           areaSum, &
           tempWeightedSum

      err = 0

      dminfo = domain % dminfo

      block => domain % blocklist
      do while (associated(block))

         ! get pointers to pools
         call mpas_pool_get_subpool(block % structs, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'aceSeaiceDerivedFieldsAM', aceSeaiceDerivedFieldsAMPool)

         ! get dimensions
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_dimension(block % dimensions, 'nCategories', nCategories)

         ! get input tracer arrays (dimensions: ONE x nCategories x nCells)
         call mpas_pool_get_array(tracersPool, 'iceAreaCategory', iceAreaCategory, timeLevel)
         call mpas_pool_get_array(tracersPool, 'iceVolumeCategory', iceVolumeCategory, timeLevel)
         call mpas_pool_get_array(tracersPool, 'snowVolumeCategory', snowVolumeCategory, timeLevel)
         call mpas_pool_get_array(tracersPool, 'surfaceTemperature', surfaceTemperature, timeLevel)

         ! get output arrays
         call mpas_pool_get_array(aceSeaiceDerivedFieldsAMPool, 'iceAreaTotal', iceAreaTotal)
         call mpas_pool_get_array(aceSeaiceDerivedFieldsAMPool, 'iceVolumeTotal', iceVolumeTotal)
         call mpas_pool_get_array(aceSeaiceDerivedFieldsAMPool, 'snowVolumeTotal', snowVolumeTotal)
         call mpas_pool_get_array(aceSeaiceDerivedFieldsAMPool, 'iceThicknessMean', iceThicknessMean)
         call mpas_pool_get_array(aceSeaiceDerivedFieldsAMPool, 'surfaceTemperatureMean', surfaceTemperatureMean)

         ! compute derived fields by summing over categories
         do iCell = 1, nCellsSolve

            areaSum = 0.0_RKIND
            iceVolumeTotal(iCell) = 0.0_RKIND
            snowVolumeTotal(iCell) = 0.0_RKIND
            tempWeightedSum = 0.0_RKIND

            do iCategory = 1, nCategories
               areaSum = areaSum + iceAreaCategory(1, iCategory, iCell)
               iceVolumeTotal(iCell) = iceVolumeTotal(iCell) + iceVolumeCategory(1, iCategory, iCell)
               snowVolumeTotal(iCell) = snowVolumeTotal(iCell) + snowVolumeCategory(1, iCategory, iCell)
               tempWeightedSum = tempWeightedSum + &
                    iceAreaCategory(1, iCategory, iCell) * surfaceTemperature(1, iCategory, iCell)
            end do

            iceAreaTotal(iCell) = areaSum

            if (areaSum > 0.0_RKIND) then
               iceThicknessMean(iCell) = iceVolumeTotal(iCell) / areaSum
               surfaceTemperatureMean(iCell) = tempWeightedSum / areaSum
            else
               iceThicknessMean(iCell) = 0.0_RKIND
               surfaceTemperatureMean(iCell) = 0.0_RKIND
            end if

         end do

         block => block % next
      end do

   end subroutine seaice_compute_ace_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_restart_ace_derived_fields
!
!> \brief   Save restart for MPAS-Seaice analysis member
!> \author  ACE Training Data
!> \date    2026/03/28
!> \details
!>  This routine conducts computation required to save a restart state
!>  for the MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_restart_ace_derived_fields(domain, instance, err)!{{{

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

   end subroutine seaice_restart_ace_derived_fields!}}}

!***********************************************************************
!
!  routine seaice_finalize_ace_derived_fields
!
!> \brief   Finalize MPAS-Seaice analysis member
!> \author  ACE Training Data
!> \date    2026/03/28
!> \details
!>  This routine conducts all finalizations required for this
!>  MPAS-Seaice analysis member.
!
!-----------------------------------------------------------------------

   subroutine seaice_finalize_ace_derived_fields(domain, instance, err)!{{{

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

   end subroutine seaice_finalize_ace_derived_fields!}}}

end module seaice_ace_derived_fields

! vim: foldmethod=marker
