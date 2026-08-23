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
  real(R8), allocatable :: sea_sst(:)    ! the ocean's own sea surface temperature (K)
  integer               :: nstored = 0
  logical               :: valid   = .false.
  logical               :: sst_valid = .false.

  ! Grid, published once by the ocean at init so the ice half of the entity
  ! needs no I/O of its own and cannot disagree about the mesh.
  real(R8), allocatable :: g_lon(:), g_lat(:), g_area(:), g_frac(:)
  integer               :: gsize_stored = 0   ! this task's share
  integer               :: gsize_global = 0   ! the whole mesh
  logical               :: grid_valid   = .false.

  public :: shr_emul_ice_put
  public :: shr_emul_ice_get
  public :: shr_emul_ice_avail
  public :: shr_emul_ice_put_grid
  public :: shr_emul_ice_get_grid
  public :: shr_emul_ice_grid_size
  public :: shr_emul_ice_grid_gsize
  public :: shr_emul_ice_sst_avail
  public :: shr_emul_ice_get_sst

contains

  !===============================================================================
  subroutine shr_emul_ice_put(frac, sst)

    ! Publish the emulator's sea ice fraction, and optionally its sea surface
    ! temperature.  Called by the ocean once per coupling step, after it has
    ! blended its bracketing states.
    !
    ! The sea surface temperature is here because the coupler's merged Sx_t is
    ! not what a prescribed-ocean atmosphere emulator was trained to consume.
    ! SamudrACE's atmosphere config is
    !     ocean = {surface_temperature_name: TS, ocean_fraction_name: OCNFRAC,
    !              interpolate: true}
    ! so its reference coupler builds TS = OCNFRAC*sst + (1-OCNFRAC)*TS_atm and
    ! the atmosphere keeps its own predicted surface temperature over land *and
    ! over sea ice*.  Reproducing that needs the ocean's unmerged sst, which no
    ! x2a field carries.  See eatm_ts_from_ocn.

    real(R8), intent(in)           :: frac(:)
    real(R8), intent(in), optional :: sst(:)

    if (allocated(ice_frac)) then
      if (size(ice_frac) /= size(frac)) deallocate(ice_frac)
    end if
    if (.not. allocated(ice_frac)) allocate(ice_frac(size(frac)))

    ice_frac(:) = frac(:)
    nstored     = size(frac)
    valid       = .true.

    if (present(sst)) then
      if (size(sst) == size(frac)) then
        if (allocated(sea_sst)) then
          if (size(sea_sst) /= size(sst)) deallocate(sea_sst)
        end if
        if (.not. allocated(sea_sst)) allocate(sea_sst(size(sst)))
        sea_sst(:) = sst(:)
        sst_valid  = .true.
      end if
    end if

  end subroutine shr_emul_ice_put

  !===============================================================================
  logical function shr_emul_ice_sst_avail(n)

    ! Is there a published sea surface temperature of the size the caller
    ! expects?  Separate from shr_emul_ice_avail because the fraction is
    ! published by every ocean build and the temperature only by one that was
    ! asked for it.

    integer, intent(in) :: n

    shr_emul_ice_sst_avail = sst_valid .and. (nstored == n) .and. allocated(sea_sst)

  end function shr_emul_ice_sst_avail

  !===============================================================================
  subroutine shr_emul_ice_get_sst(sst)

    real(R8), intent(out) :: sst(:)

    if (.not. shr_emul_ice_sst_avail(size(sst))) then
      sst(:) = 0.0_R8
    else
      sst(:) = sea_sst(:)
    end if

  end subroutine shr_emul_ice_get_sst

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

  !===============================================================================
  subroutine shr_emul_ice_put_grid(lon, lat, area, frac, gsize)

    ! Publish the ocean's grid.  Called once, at ocean init -- which the MCT
    ! driver runs before ice init, so the ice half always finds it.

    real(R8), intent(in) :: lon(:), lat(:), area(:), frac(:)
    integer,  intent(in) :: gsize      ! global, not this task's share

    gsize_global = gsize
    gsize_stored = size(lon)
    if (allocated(g_lon)) deallocate(g_lon, g_lat, g_area, g_frac)
    allocate(g_lon(gsize_stored), g_lat(gsize_stored), &
             g_area(gsize_stored), g_frac(gsize_stored))
    g_lon  = lon
    g_lat  = lat
    g_area = area
    g_frac = frac
    grid_valid = .true.

  end subroutine shr_emul_ice_put_grid

  !===============================================================================
  integer function shr_emul_ice_grid_size()

    if (grid_valid) then
      shr_emul_ice_grid_size = gsize_stored
    else
      shr_emul_ice_grid_size = 0
    end if

  end function shr_emul_ice_grid_size

  !===============================================================================
  integer function shr_emul_ice_grid_gsize()

    if (grid_valid) then
      shr_emul_ice_grid_gsize = gsize_global
    else
      shr_emul_ice_grid_gsize = 0
    end if

  end function shr_emul_ice_grid_gsize

  !===============================================================================
  subroutine shr_emul_ice_get_grid(lon, lat, area, frac)

    real(R8), intent(out) :: lon(:), lat(:), area(:), frac(:)

    lon  = g_lon
    lat  = g_lat
    area = g_area
    frac = g_frac

  end subroutine shr_emul_ice_get_grid

end module shr_emul_ice_mod
