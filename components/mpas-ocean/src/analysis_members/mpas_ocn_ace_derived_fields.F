! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_ace_derived_fields
!
!> \brief MPAS ocean analysis mode member: ACE derived surface fields
!> \author ACE Training Data Support
!> \date   March 2026
!> \details
!>  Extracts sea surface temperature (SST) and sea surface salinity (SSS)
!>  from the top level of the active tracer arrays for ACE training data.
!>
!-----------------------------------------------------------------------

module ocn_ace_derived_fields

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_timekeeping
   use mpas_stream_manager

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

   public :: ocn_init_ace_derived_fields, &
             ocn_compute_ace_derived_fields, &
             ocn_restart_ace_derived_fields, &
             ocn_finalize_ace_derived_fields

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_ace_derived_fields
!
!> \brief   Initialize MPAS-Ocean analysis member
!> \author  ACE Training Data Support
!> \date    March 2026
!> \details
!>  This routine conducts all initializations required for the
!>  MPAS-Ocean analysis member.
!
!-----------------------------------------------------------------------

   subroutine ocn_init_ace_derived_fields(domain, err)!{{{

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

   end subroutine ocn_init_ace_derived_fields!}}}

!***********************************************************************
!
!  routine ocn_compute_ace_derived_fields
!
!> \brief   Compute MPAS-Ocean analysis member
!> \author  ACE Training Data Support
!> \date    March 2026
!> \details
!>  This routine conducts all computation required for this
!>  MPAS-Ocean analysis member. It extracts SST and SSS from the
!>  top level of the active tracer arrays.
!
!-----------------------------------------------------------------------

   subroutine ocn_compute_ace_derived_fields(domain, timeLevel, err)!{{{

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

      integer, pointer :: nCellsSolve, index_temperature, index_salinity
      integer :: iCell

      real(kind=RKIND), dimension(:, :, :), pointer :: tracers
      real(kind=RKIND), dimension(:), pointer :: sst, sss

      err = 0

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'aceDerivedFieldsAM', amPool)

         call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
         call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)

         call mpas_pool_get_array(tracersPool, 'activeTracers', tracers, timeLevel)
         call mpas_pool_get_array(amPool, 'sst', sst)
         call mpas_pool_get_array(amPool, 'sss', sss)

         do iCell = 1, nCellsSolve
            sst(iCell) = tracers(index_temperature, 1, iCell)
            sss(iCell) = tracers(index_salinity, 1, iCell)
         end do

         block => block % next
      end do

   end subroutine ocn_compute_ace_derived_fields!}}}

!***********************************************************************
!
!  routine ocn_restart_ace_derived_fields
!
!> \brief   Save restart for MPAS-Ocean analysis member
!> \author  ACE Training Data Support
!> \date    March 2026
!> \details
!>  This routine conducts computation required to save a restart state
!>  for the MPAS-Ocean analysis member.
!
!-----------------------------------------------------------------------

   subroutine ocn_restart_ace_derived_fields(domain, err)!{{{

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

   end subroutine ocn_restart_ace_derived_fields!}}}

!***********************************************************************
!
!  routine ocn_finalize_ace_derived_fields
!
!> \brief   Finalize MPAS-Ocean analysis member
!> \author  ACE Training Data Support
!> \date    March 2026
!> \details
!>  This routine conducts all finalizations required for this
!>  MPAS-Ocean analysis member.
!
!-----------------------------------------------------------------------

   subroutine ocn_finalize_ace_derived_fields(domain, err)!{{{

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

   end subroutine ocn_finalize_ace_derived_fields!}}}

end module ocn_ace_derived_fields

! vim: foldmethod=marker
