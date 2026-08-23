module shr_emul_flux_mod

  !-----------------------------------------------------------------------------
  ! A side channel from the emulator atmosphere to the emulator ocean carrying
  ! the ten surface forcing channels the atmosphere emulator actually generated.
  !
  ! Why this exists.  In reference SamudrACE the ocean is forced by the
  ! atmosphere emulator's own generated output channels -- the coupler in
  ! fme/coupled/stepper.py averages ACE's raw TAUX, TAUY, precipitation, FLUS,
  ! FSUS, FLDS, FSDS, LHFLX and SHFLX over the ocean's step and hands them
  ! straight to Samudra.  Nothing is recomputed and nothing is weighted.
  !
  ! Routed through E3SM's coupler instead, none of those channels survives.
  ! The MCT driver discards the emulator's turbulent fluxes and rebuilds them
  ! with bulk formulae from the atmosphere's lowest-level state and the ocean's
  ! own surface temperature, then hands the ocean only the open-water share:
  ! prep_ocn_merge builds Foxx_lat, Foxx_sen, Foxx_lwup and Foxx_evap as
  ! afrac*<atm/ocn flux> with no ice term at all.  Recovering a whole-cell flux
  ! from that means dividing by afrac, which is unbounded as the ice closes;
  ! and the shortwave has to be split into up and down components with an
  ! assumed ocean albedo because only the net crosses the coupler.
  !
  ! Both halves of the emulated pair come from one SamudrACE checkpoint, so the
  ! channels the atmosphere writes are, name for name and sign for sign, the
  ! channels the ocean reads.  Publishing them directly removes the bulk-flux
  ! recomputation, the ice-fraction unweighting and the assumed albedo in one
  ! step, and makes the E3SM pair reproduce the reference forcing path.
  !
  ! Like shr_emul_ice_mod, this only works because every component of an E3SM
  ! executable shares an address space, and only when the atmosphere and ocean
  ! share a grid and a decomposition -- true for the emulated pair and nothing
  ! else.  The stored grid shape -- not just the point count -- is checked on
  ! every read for exactly that reason: a 180x360 producer and a 360x180
  ! consumer agree on npts and would otherwise exchange ten transposed fields
  ! that are finite, in range and silently wrong.  A reader that does not like
  ! what it finds is expected to fall back to the coupler's fields rather than
  ! proceed.
  !-----------------------------------------------------------------------------

  use shr_kind_mod, only: R8=>SHR_KIND_R8

  implicit none
  private
  save

  ! Channel order.  Public so that both ends index the same slot by name and
  ! neither has to know the other's storage layout.
  integer, parameter, public :: shr_emul_flux_nchan = 10
  integer, parameter, public :: shr_emul_flux_taux  =  1  ! N/m2,     stress on the atmosphere
  integer, parameter, public :: shr_emul_flux_tauy  =  2  ! N/m2,     stress on the atmosphere
  integer, parameter, public :: shr_emul_flux_prec  =  3  ! kg/m2/s,  total precipitation
  integer, parameter, public :: shr_emul_flux_snow  =  4  ! kg/m2/s,  frozen precipitation
  integer, parameter, public :: shr_emul_flux_flus  =  5  ! W/m2,     positive upward
  integer, parameter, public :: shr_emul_flux_fsus  =  6  ! W/m2,     positive upward
  integer, parameter, public :: shr_emul_flux_flds  =  7  ! W/m2,     positive downward
  integer, parameter, public :: shr_emul_flux_fsds  =  8  ! W/m2,     positive downward
  integer, parameter, public :: shr_emul_flux_lhflx =  9  ! W/m2,     positive upward
  integer, parameter, public :: shr_emul_flux_shflx = 10  ! W/m2,     positive upward

  real(R8), allocatable :: chan(:,:)      ! (shr_emul_flux_nchan, npts)
  integer               :: nstored = 0
  integer               :: nx_stored = 0  ! producer's grid shape, so that a
  integer               :: ny_stored = 0  ! transposed consumer cannot match
  logical               :: valid   = .false.

  public :: shr_emul_flux_put
  public :: shr_emul_flux_get
  public :: shr_emul_flux_avail

contains

  !===============================================================================
  subroutine shr_emul_flux_put(f, nx, ny)

    ! Publish the atmosphere emulator's raw surface forcing channels, in the
    ! emulator's own units and sign convention -- exactly the numbers the
    ! network produced, before any coupler field they are also used to fill.
    ! Called once per coupling step so that a reader accumulating per step sees
    ! each emulator step weighted by the number of coupling steps it spans.

    real(R8), intent(in) :: f(:,:)   ! (shr_emul_flux_nchan, npts)
    integer,  intent(in) :: nx, ny   ! the producer's grid shape, nx*ny = npts

    if (size(f,1) /= shr_emul_flux_nchan .or. nx*ny /= size(f,2)) then
      valid = .false.
      return
    end if

    if (allocated(chan)) then
      if (size(chan,2) /= size(f,2)) deallocate(chan)
    end if
    if (.not. allocated(chan)) allocate(chan(shr_emul_flux_nchan, size(f,2)))

    chan(:,:) = f(:,:)
    nstored   = size(f,2)
    nx_stored = nx
    ny_stored = ny
    valid     = .true.

  end subroutine shr_emul_flux_put

  !===============================================================================
  logical function shr_emul_flux_avail(nx, ny)

    ! Is there a published set of channels on the grid the caller expects?  A
    ! false here is not an error: it is the ordinary answer whenever the
    ! atmosphere is EAM or EAMxx rather than the emulator, and the caller is
    ! expected to use the coupler's fields instead.  The shape is compared, not
    ! only the point count, so a transposed decomposition is rejected.

    integer, intent(in) :: nx, ny

    shr_emul_flux_avail = valid .and. allocated(chan) .and. &
         (nstored == nx*ny) .and. (nx_stored == nx) .and. (ny_stored == ny)

  end function shr_emul_flux_avail

  !===============================================================================
  subroutine shr_emul_flux_get(f, nx, ny)

    ! Read the published channels.  The caller must have checked
    ! shr_emul_flux_avail with its own grid shape first; a mismatch here is a
    ! programming error and leaves f untouched.

    real(R8), intent(inout) :: f(:,:)   ! (shr_emul_flux_nchan, npts)
    integer,  intent(in)    :: nx, ny

    if (.not. valid) return
    if (.not. allocated(chan)) return
    if (size(f,1) /= shr_emul_flux_nchan) return
    if (size(f,2) /= nstored) return
    if (nx /= nx_stored .or. ny /= ny_stored) return

    f(:,:) = chan(:,:)

  end subroutine shr_emul_flux_get

end module shr_emul_flux_mod
