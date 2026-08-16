module eatmMod

  ! !USES:

  use shr_kind_mod, only: &
    IN=>SHR_KIND_IN, &
    R4=>SHR_KIND_R4, &
    R8=>SHR_KIND_R8, &
    CS=>SHR_KIND_CS, &
    CL=>SHR_KIND_CL

  ! !PUBLIC TYPES:

  implicit none
  private ! except

  !--------------------------------------------------------------------------
  ! Public module data
  !--------------------------------------------------------------------------
  integer, public           :: gsize, lsize, lsize_x, lsize_y
  character(CL), public     :: restart_file
  character(CL), public     :: case_name      ! case name
  character(len=16), public :: inst_name
  character(len=16), public :: inst_suffix = ""    ! char string associated with instance (ie. "_0001" or "")

  !--------------------------------------------------------------------------
  ! eatm_inparm namelist settings (read in atm_comp_mct::eatm_read_namelist)
  !--------------------------------------------------------------------------
  logical, public       :: do_eatm                ! run the emulator at all
  character(CL), public :: filename_eatm          ! SCRIP mesh describing the eatm grid
  character(CL), public :: eatm_emulator          ! which emulator's channel table to use
  character(CL), public :: eatm_model_file        ! traced TorchScript model (.pt)
  character(CL), public :: eatm_ic_file           ! initial-condition file for a startup run
  character(CL), public :: eatm_model_device      ! 'cpu' or 'gpu'
  character(CL), public :: eatm_frzprec_units     ! units of the frozen precip channel
  logical, public       :: eatm_pass_forcing      ! append next-step forcing channels
  logical, public       :: eatm_legacy_surface    ! reproduce the pre-review surface diagnostics
  integer, public       :: eatm_iradsw            ! radiation interval (coupler steps)
  character(CL), public :: eatm_surface_layer     ! 'near_surface' or 'lowest_level'

  ! Reference height for the exported atmospheric state when the emulator
  ! predicts near-surface diagnostics.  10 m is what datm hands this same ocean
  ! and sea ice under JRA forcing (datm_comp_mod.F90:1029), so an EATM run and
  ! the GMPAS-JRA1p5-2023 baseline present the surface-flux scheme with states
  ! at the same height.
  real(R8), parameter, public :: eatm_ref_height = 10.0_R8

  ! Orbital parameters (set from coupler infodata at init)
  real(kind=R8), public :: orb_eccen     ! orbital eccentricity
  real(kind=R8), public :: orb_obliqr    ! obliquity in radians
  real(kind=R8), public :: orb_lambm0    ! mean lon of perihelion at vernal equinox (rad)
  real(kind=R8), public :: orb_mvelpp    ! moving vernal equinox lon of perihelion + pi (rad)

  ! imported arrays first
  real(kind=R8), dimension(:,:), allocatable, public :: shf          ! sensible heat flux
  real(kind=R8), dimension(:,:), allocatable, public :: cflx         ! constituent flux (emissions)
  real(kind=R8), dimension(:,:), allocatable, public :: lhf          ! latent heat flux
  real(kind=R8), dimension(:,:), allocatable, public :: wsx          ! surface u-stress (N)
  real(kind=R8), dimension(:,:), allocatable, public :: wsy          ! surface v-stress (N)
  real(kind=R8), dimension(:,:), allocatable, public :: lwup         ! longwave up radiative flux
  real(kind=R8), dimension(:,:), allocatable, public :: asdir        ! albedo: shortwave, direct
  real(kind=R8), dimension(:,:), allocatable, public :: aldir        ! albedo: longwave, direct
  real(kind=R8), dimension(:,:), allocatable, public :: asdif        ! albedo: shortwave, diffuse
  real(kind=R8), dimension(:,:), allocatable, public :: aldif        ! albedo: longwave, diffuse
  real(kind=R8), dimension(:,:), allocatable, public :: ts           ! merged surface temp
  real(kind=R8), dimension(:,:), allocatable, public :: sst          ! sea surface temp
  real(kind=R8), dimension(:,:), allocatable, public :: snowhland    ! snow depth (liquid water equivalent) over land
  real(kind=R8), dimension(:,:), allocatable, public :: snowhice     ! snow depth over ice
  real(kind=R8), dimension(:,:), allocatable, public :: tref         ! ref height surface air temp
  real(kind=R8), dimension(:,:), allocatable, public :: qref         ! ref height specific humidity
  real(kind=R8), dimension(:,:), allocatable, public :: u10          ! 10m wind speed
  real(kind=R8), dimension(:,:), allocatable, public :: u10withgusts ! 10m wind speed with gustiness
  real(kind=R8), dimension(:,:), allocatable, public :: icefrac      ! sea-ice areal fraction
  real(kind=R8), dimension(:,:), allocatable, public :: ocnfrac      ! ocean areal fraction
  real(kind=R8), dimension(:,:), allocatable, public :: lndfrac      ! land area fraction

  ! exported arrays
  real(kind=R8), dimension(:,:), allocatable, public :: topo         ! surface height above sea level
  real(kind=R8), dimension(:,:), allocatable, public :: zbot         ! bot level height above surface
  real(kind=R8), dimension(:,:), allocatable, public :: ubot         ! bot level u wind
  real(kind=R8), dimension(:,:), allocatable, public :: vbot         ! bot level v wind
  real(kind=R8), dimension(:,:), allocatable, public :: tbot         ! bot level temperature
  real(kind=R8), dimension(:,:), allocatable, public :: ptem         ! bot level potential temperature
  real(kind=R8), dimension(:,:), allocatable, public :: shum         ! bot level specific humidity
  real(kind=R8), dimension(:,:), allocatable, public :: dens         ! bot level density
  real(kind=R8), dimension(:,:), allocatable, public :: pbot         ! bot level pressure
  real(kind=R8), dimension(:,:), allocatable, public :: pslv         ! sea level atm pressure
  real(kind=R8), dimension(:,:), allocatable, public :: lwdn         ! Down longwave flux at surface
  real(kind=R8), dimension(:,:), allocatable, public :: rainc        ! liquid "convective" precip
  real(kind=R8), dimension(:,:), allocatable, public :: rainl        ! liquid "large scale" precip
  real(kind=R8), dimension(:,:), allocatable, public :: snowc        ! frozen "convective" precip
  real(kind=R8), dimension(:,:), allocatable, public :: snowl        ! frozen "large scale" precip
  real(kind=R8), dimension(:,:), allocatable, public :: swndr        ! direct near-infrared incident solar radiation
  real(kind=R8), dimension(:,:), allocatable, public :: swvdr        ! direct visible incident solar radiation
  real(kind=R8), dimension(:,:), allocatable, public :: swndf        ! diffuse near-infrared incident solar radiation
  real(kind=R8), dimension(:,:), allocatable, public :: swvdf        ! diffuse visible incident solar radiation
  real(kind=R8), dimension(:,:), allocatable, public :: swnet        ! net shortwave radiation

  !
  real(kind=R4), dimension(:, :, :, :), allocatable, target, public :: net_inputs
  real(kind=R4), dimension(:, :, :, :), allocatable, target, public :: net_inputs_nn
  real(kind=R4), dimension(:, :, :, :), allocatable, target, public :: net_outputs

  character(CS), public :: myModelName = 'atm'   ! user defined model name

  character(len=*), parameter, public :: rpfile = 'rpointer.atm'

  ! Two bracketing emulator states, (channel, x, y).  The emulator advances on
  ! its own (6-hourly) timestep; the coupler sees a linear interpolation
  ! between t_im1 (state at the last emulator step) and t_ip1 (state at the
  ! next one).  Both levels are written to the restart file so that a restart
  ! reproduces the interpolation exactly.
  !
  ! NOTE: deliberately a plain derived type rather than a parameterized one --
  ! PDT support is uneven across the compilers E3SM has to build with.
  type :: t_eatm_interpolator
    real(kind=R4), dimension(:, :, :), allocatable :: t_im1
    real(kind=R4), dimension(:, :, :), allocatable :: t_ip1
  end type t_eatm_interpolator

  type(t_eatm_interpolator), public :: eatm_intrp

  ! NOTE: normalization and denormalization used to live here, driven by the
  ! ace2_EAMv3_{,de}normalize.nc statistics files.  They are no longer needed:
  ! the traced TorchScript model produced by the ACE tracing script bakes
  ! normalization, denormalization and the atmosphere correctors into the
  ! graph.  See git history (pre eatm channel-table refactor) to recover them.

end module eatmMod
