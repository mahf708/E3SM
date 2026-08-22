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
