module eocnMod

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
  character(len=16), public :: inst_suffix = ""

  !--------------------------------------------------------------------------
  ! eocn_inparm namelist settings (read in ocn_comp_mct::eocn_read_namelist)
  !--------------------------------------------------------------------------
  logical, public       :: do_eocn             ! run the emulator at all
  character(CL), public :: filename_eocn       ! SCRIP mesh describing the eocn grid
  character(CL), public :: eocn_emulator       ! which emulator's channel table to use
  character(CL), public :: eocn_model_file     ! traced TorchScript model (.pt)
  character(CL), public :: eocn_ic_file        ! initial-condition file for a startup run
  character(CL), public :: eocn_model_device   ! 'cpu' or 'gpu'
  integer, public       :: eocn_rng_seed       ! libtorch RNG seed; <0 leaves it unseeded
  logical, public       :: eocn_interp_state   ! interpolate between emulator brackets
  logical, public       :: eocn_flux_ifrac_unweight ! undo the coupler's open-water weighting
  logical, public       :: eocn_unweight_stress     ! apply that un-weighting to TAUX/TAUY too
  character(CS), public :: eocn_precip_units        ! units of the emulator's precipitation channels
  character(CS), public :: eocn_forcing_source      ! 'coupler' or 'atm_raw'

  !--------------------------------------------------------------------------
  ! Imported coupler fluxes, accumulated over the emulator interval.
  !
  ! Samudra's ten atmospheric forcing channels are means over its 5 day step
  ! (cell_methods "time: mean" in the training stream), while the coupler
  ! recomputes and re-imports its fluxes every coupling step.  Reading them
  ! once at the emulator boundary would hand a 5 day mean channel a single
  ! 30 min sample.  These accumulate every coupling step and are averaged and
  ! reset when the emulator advances.
  !--------------------------------------------------------------------------
  real(R8), dimension(:,:), allocatable, public :: acc_taux   ! N/m2, on the atmosphere
  real(R8), dimension(:,:), allocatable, public :: acc_tauy   ! N/m2, on the atmosphere
  real(R8), dimension(:,:), allocatable, public :: acc_prec   ! m/s, liquid + frozen
  real(R8), dimension(:,:), allocatable, public :: acc_snow   ! m/s, frozen only
  real(R8), dimension(:,:), allocatable, public :: acc_flus   ! W/m2, upward   longwave
  real(R8), dimension(:,:), allocatable, public :: acc_fsus   ! W/m2, upward   shortwave
  real(R8), dimension(:,:), allocatable, public :: acc_flds   ! W/m2, downward longwave
  real(R8), dimension(:,:), allocatable, public :: acc_fsds   ! W/m2, downward shortwave
  real(R8), dimension(:,:), allocatable, public :: acc_lhflx  ! W/m2, upward latent
  real(R8), dimension(:,:), allocatable, public :: acc_shflx  ! W/m2, upward sensible
  integer, public :: acc_n = 0                                ! coupling steps accumulated

  ! The same ten channels again, taken instead from the atmosphere emulator's
  ! own generated output when one is sharing this executable and publishing
  ! them (shr_emul_flux_mod).  Accumulated alongside the coupler's fields on
  ! every step, whichever is selected, so that a run reports what the other
  ! path would have given it: the gap between them is the whole of what the
  ! coupler's bulk-flux recomputation and open-water weighting do to the
  ! forcing, and it is not otherwise measurable from inside a run.
  ! Stored in the emulator's own sign convention -- positive upward for the
  ! turbulent and upward radiative channels -- so they need no conversion.
  real(R8), dimension(:,:), allocatable, public :: raw_taux
  real(R8), dimension(:,:), allocatable, public :: raw_tauy
  real(R8), dimension(:,:), allocatable, public :: raw_prec
  real(R8), dimension(:,:), allocatable, public :: raw_snow
  real(R8), dimension(:,:), allocatable, public :: raw_flus
  real(R8), dimension(:,:), allocatable, public :: raw_fsus
  real(R8), dimension(:,:), allocatable, public :: raw_flds
  real(R8), dimension(:,:), allocatable, public :: raw_fsds
  real(R8), dimension(:,:), allocatable, public :: raw_lhflx
  real(R8), dimension(:,:), allocatable, public :: raw_shflx
  integer, public :: raw_n = 0                                ! steps with a published set

  ! Seconds elapsed since the emulator last advanced.  Counted rather than
  ! derived from the calendar because the emulator step is longer than a day,
  ! so mod() on the time of day cannot express the cadence at all.  Carried
  ! across a restart with the flux accumulators.
  integer, public :: eocn_elapsed = 0

  !--------------------------------------------------------------------------
  ! Exported ocean state
  !--------------------------------------------------------------------------
  real(R8), dimension(:,:), allocatable, public :: so_t       ! sea surface temperature (K)
  real(R8), dimension(:,:), allocatable, public :: so_s       ! sea surface salinity (g/kg)
  real(R8), dimension(:,:), allocatable, public :: so_u       ! surface zonal current (m/s)
  real(R8), dimension(:,:), allocatable, public :: so_v       ! surface meridional current (m/s)
  real(R8), dimension(:,:), allocatable, public :: so_ssh     ! sea surface height (m)
  real(R8), dimension(:,:), allocatable, public :: so_dhdx    ! zonal SSH slope (m/m)
  real(R8), dimension(:,:), allocatable, public :: so_dhdy    ! meridional SSH slope (m/m)
  real(R8), dimension(:,:), allocatable, public :: so_ifrac   ! emulator sea ice fraction

  !--------------------------------------------------------------------------
  ! Grid.  Cell centre coordinates are needed for the SSH slope, and the ocean
  ! mask decides where the emulator's state means anything at all: Samudra
  ! carries NaN over land in its own initial conditions.
  !--------------------------------------------------------------------------
  real(R8), dimension(:,:), allocatable, public :: cell_lat   ! degrees north
  real(R8), dimension(:,:), allocatable, public :: cell_lon   ! degrees east
  real(R8), dimension(:,:), allocatable, public :: ocn_mask   ! 1 where the emulator is valid

  ! Where the emulator's sea ice channels mean anything.  Samudra's ocean mask
  ! covers 44,892 cells, but ocean_sea_ice_fraction and iceVolumeTotal are
  ! masked far more tightly in training -- 25,923 cells -- and outside that the
  ! network was never shown ice and its output there is not a prediction.
  ! Exporting it anyway puts sea ice in the tropics.  See samudra_export.
  real(R8), dimension(:,:), allocatable, public :: ice_mask   ! 1 where sea ice is predicted

  real(kind=R4), dimension(:, :, :, :), allocatable, target, public :: net_inputs
  real(kind=R4), dimension(:, :, :, :), allocatable, target, public :: net_inputs_nn
  real(kind=R4), dimension(:, :, :, :), allocatable, target, public :: net_outputs

  character(CS), public :: myModelName = 'ocn'

  character(len=*), parameter, public :: rpfile = 'rpointer.ocn'

  ! Two bracketing emulator states, (channel, x, y).  Same construction as
  ! EATM's: the emulator advances on its own (5 day) timestep and the coupler
  ! sees a linear interpolation between the bracketing states, so a 5 day step
  ! in SST does not arrive at the atmosphere as a discontinuity.  Both levels
  ! go to the restart file so a restart reproduces the interpolation exactly.
  type :: t_eocn_interpolator
    real(kind=R4), dimension(:, :, :), allocatable :: t_im1
    real(kind=R4), dimension(:, :, :), allocatable :: t_ip1
  end type t_eocn_interpolator

  type(t_eocn_interpolator), public :: eocn_intrp

end module eocnMod
