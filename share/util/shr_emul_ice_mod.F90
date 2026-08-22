module shr_emul_ice_mod

  !-----------------------------------------------------------------------------
  ! A side channel from the emulator ocean to the emulator atmosphere carrying
  ! the one field the MCT coupler has nowhere to put: the sea ice fraction the
  ! ocean emulator predicts internally.
  !
  ! SamudrACE's ocean-to-atmosphere exchange is exactly
  ! [ocean_sea_ice_fraction, sst], and its coupler splits the non-land fraction
  ! with
  !
  !     ICEFRAC = ocean_sea_ice_fraction * (1 - LANDFRAC)
  !     OCNFRAC = max(1 - LANDFRAC - ICEFRAC, 0)
  !
  ! (fme/coupled/stepper.py, CoupledOceanFractionConfig and OceanData; the
  ! identity holds to float32 in the published initial conditions).  E3SM's
  ! coupler expresses the same identity as lfrac + ifrac + ofrac = 1, but it
  ! fills ifrac from a sea ice *component*, and with a stub ice there is none --
  ! so the atmosphere is told the polar ocean is open water.
  !
  ! This module exists to measure what that costs, not to be the fix.  The fix
  ! is a sea ice component that reports the emulator's fraction as Si_ifrac, at
  ! which point the coupler computes ifrac itself and this channel is dead code.
  ! It is gated behind eatm_icefrac_from_ocn, which defaults to .false.
  !
  ! It only works because every component of an E3SM executable shares an
  ! address space, and only when the atmosphere and ocean share a grid and a
  ! decomposition -- which is true for the emulated pair and nothing else.  The
  ! stored size is checked on every read for exactly that reason.
  !-----------------------------------------------------------------------------

  use shr_kind_mod, only: R8=>SHR_KIND_R8

  implicit none
  private
  save

  real(R8), allocatable :: ice_frac(:)   ! fraction of the sea surface, not of the cell
  integer               :: nstored = 0
  logical               :: valid   = .false.

  public :: shr_emul_ice_put
  public :: shr_emul_ice_get
  public :: shr_emul_ice_avail

contains

  !===============================================================================
  subroutine shr_emul_ice_put(frac)

    ! Publish the emulator's sea ice fraction.  Called by the ocean once per
    ! coupling step, after it has blended its bracketing states.

    real(R8), intent(in) :: frac(:)

    if (allocated(ice_frac)) then
      if (size(ice_frac) /= size(frac)) deallocate(ice_frac)
    end if
    if (.not. allocated(ice_frac)) allocate(ice_frac(size(frac)))

    ice_frac(:) = frac(:)
    nstored     = size(frac)
    valid       = .true.

  end subroutine shr_emul_ice_put

  !===============================================================================
  logical function shr_emul_ice_avail(n)

    ! Is there a published fraction, and is it the size the caller expects?
    ! A size mismatch means the two components are not on the same grid or the
    ! same decomposition, and the answer is no rather than a silent mis-index.

    integer, intent(in) :: n

    shr_emul_ice_avail = valid .and. (nstored == n)

  end function shr_emul_ice_avail

  !===============================================================================
  subroutine shr_emul_ice_get(frac)

    real(R8), intent(out) :: frac(:)

    if (.not. shr_emul_ice_avail(size(frac))) then
      frac(:) = 0.0_R8
    else
      frac(:) = ice_frac(:)
    end if

  end subroutine shr_emul_ice_get

end module shr_emul_ice_mod
