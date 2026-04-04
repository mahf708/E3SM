! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_fme_derived_fields
!
!> \brief MPAS ocean analysis member: FME derived surface fields
!> \author FME Training Data Support
!> \date   April 2026
!> \details
!>  Computes derived surface fields for Full Model Emulation training data:
!>  - SST: sea surface temperature from top-level temperature tracer
!>  - SSS: sea surface salinity from top-level salinity tracer
!>  - surfaceHeatFluxTotal: combined SW + LW_down + latent + sensible
!>
!-----------------------------------------------------------------------

module ocn_fme_derived_fields

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

   public :: ocn_init_fme_derived_fields, &
             ocn_compute_fme_derived_fields, &
             ocn_restart_fme_derived_fields, &
             ocn_finalize_fme_derived_fields

!***********************************************************************

contains

!***********************************************************************

   subroutine ocn_init_fme_derived_fields(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      err = 0

   end subroutine ocn_init_fme_derived_fields!}}}

!***********************************************************************

   subroutine ocn_compute_fme_derived_fields(domain, timeLevel, err)!{{{

      integer, intent(in) :: timeLevel
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: statePool
      type(mpas_pool_type), pointer :: meshPool
      type(mpas_pool_type), pointer :: tracersPool
      type(mpas_pool_type), pointer :: forcingPool
      type(mpas_pool_type), pointer :: amPool

      integer, pointer :: nCellsSolve, index_temperature, index_salinity
      integer :: iCell

      real(kind=RKIND), dimension(:, :, :), pointer :: tracers
      real(kind=RKIND), dimension(:), pointer :: sst, sss, surfaceHeatFluxTotal
      real(kind=RKIND), dimension(:), pointer :: shortWaveHeatFlux
      real(kind=RKIND), dimension(:), pointer :: longWaveHeatFluxDown
      real(kind=RKIND), dimension(:), pointer :: latentHeatFlux
      real(kind=RKIND), dimension(:), pointer :: sensibleHeatFlux

      err = 0

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'forcing', forcingPool)
         call mpas_pool_get_subpool(block % structs, 'fmeDerivedFieldsAM', amPool)

         call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
         call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)

         call mpas_pool_get_array(tracersPool, 'activeTracers', tracers, timeLevel)
         call mpas_pool_get_array(amPool, 'sst', sst)
         call mpas_pool_get_array(amPool, 'sss', sss)
         call mpas_pool_get_array(amPool, 'surfaceHeatFluxTotal', surfaceHeatFluxTotal)

         call mpas_pool_get_array(forcingPool, 'shortWaveHeatFlux', shortWaveHeatFlux)
         call mpas_pool_get_array(forcingPool, 'longWaveHeatFluxDown', longWaveHeatFluxDown)
         call mpas_pool_get_array(forcingPool, 'latentHeatFlux', latentHeatFlux)
         call mpas_pool_get_array(forcingPool, 'sensibleHeatFlux', sensibleHeatFlux)

         do iCell = 1, nCellsSolve
            sst(iCell) = tracers(index_temperature, 1, iCell)
            sss(iCell) = tracers(index_salinity, 1, iCell)

            if (associated(shortWaveHeatFlux) .and. &
                associated(longWaveHeatFluxDown) .and. &
                associated(latentHeatFlux) .and. &
                associated(sensibleHeatFlux)) then
               surfaceHeatFluxTotal(iCell) = shortWaveHeatFlux(iCell) &
                    + longWaveHeatFluxDown(iCell) &
                    + latentHeatFlux(iCell) &
                    + sensibleHeatFlux(iCell)
            else
               surfaceHeatFluxTotal(iCell) = 0.0_RKIND
            end if
         end do

         block => block % next
      end do

   end subroutine ocn_compute_fme_derived_fields!}}}

!***********************************************************************

   subroutine ocn_restart_fme_derived_fields(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      err = 0

   end subroutine ocn_restart_fme_derived_fields!}}}

!***********************************************************************

   subroutine ocn_finalize_fme_derived_fields(domain, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      err = 0

   end subroutine ocn_finalize_fme_derived_fields!}}}

end module ocn_fme_derived_fields

! vim: foldmethod=marker
