module eatm_channels_mod

  !-----------------------------------------------------------------------------
  ! Channel tables for the emulators that EATM knows how to drive.
  !
  ! A traced ACE model is a black box that consumes a [1, n_in, ny, nx] tensor
  ! and produces a [1, n_out, ny, nx] tensor.  The only thing EATM needs to know
  ! about a given checkpoint is which physical field sits in which channel, so
  ! that it can
  !
  !   (a) fill the input tensor from the coupler + its own previous output,
  !   (b) pull the fields the coupler wants back out of the output tensor,
  !   (c) name the channels when it writes them to a restart file.
  !
  ! Everything here is derived from the channel lists stored in the checkpoint
  ! (`in_names` / `out_names` of the fme stepper config), which are also written
  ! to the `*_metadata.yaml` emitted alongside the traced `.pt` file.  Keep the
  ! tables below in sync with that yaml.
  !
  ! Adding a new emulator means adding one `set_table_*` routine and one branch
  ! in eatm_channels_init -- no changes anywhere else in EATM.
  !-----------------------------------------------------------------------------

  use shr_kind_mod, only: R8=>SHR_KIND_R8, CL=>SHR_KIND_CL
  use shr_sys_mod,  only: shr_sys_abort

  implicit none
  private
  save

  integer, parameter, public :: eatm_nlen  = 48   ! channel name length
  integer, parameter, public :: eatm_llen  = 128  ! channel long_name length
  integer, parameter, public :: eatm_ulen  = 16   ! channel units length

  !----------------------------------------------------------------------------
  ! Table sizes.  Set by eatm_channels_init; treated as read-only afterwards.
  !----------------------------------------------------------------------------
  integer, public :: n_input_channels   = 0  ! state channels fed to the net
  integer, public :: n_output_channels  = 0  ! channels the net returns
  integer, public :: n_forcing_channels = 0  ! extra "next step" channels the
                                             ! traced graph appends after the
                                             ! state block (may be 0)

  character(len=eatm_nlen), allocatable, public :: in_names(:)
  character(len=eatm_nlen), allocatable, public :: out_names(:)
  character(len=eatm_nlen), allocatable, public :: forcing_names(:)

  !----------------------------------------------------------------------------
  ! For every input channel, the output channel it is carried forward from.
  ! Zero means the channel is not a prognostic feedback and must be supplied by
  ! EATM (coupler import, persisted boundary field, or computed insolation).
  !----------------------------------------------------------------------------
  integer, allocatable, public :: in_from_out(:)

  !----------------------------------------------------------------------------
  ! Named channel indices (1-based; 0 means "this emulator does not have it").
  ! Only the channels EATM actually touches are resolved here.
  !----------------------------------------------------------------------------
  integer, public :: ix_in_landfrac = 0
  integer, public :: ix_in_ocnfrac  = 0
  integer, public :: ix_in_icefrac  = 0
  integer, public :: ix_in_phis     = 0
  integer, public :: ix_in_solin    = 0
  integer, public :: ix_in_ps       = 0
  integer, public :: ix_in_ts       = 0

  integer, public :: ix_out_ps      = 0
  integer, public :: ix_out_ts      = 0
  integer, public :: ix_out_tbot    = 0   ! lowest model layer temperature
  integer, public :: ix_out_ubot    = 0   ! lowest model layer zonal wind
  integer, public :: ix_out_vbot    = 0   ! lowest model layer meridional wind
  integer, public :: ix_out_qbot    = 0   ! lowest model layer total water
  integer, public :: ix_out_flds    = 0   ! downward longwave at surface
  integer, public :: ix_out_fsds    = 0   ! downward shortwave at surface
  integer, public :: ix_out_fsus    = 0   ! upward   shortwave at surface
  integer, public :: ix_out_precip  = 0   ! total surface precipitation rate
  integer, public :: ix_out_snow    = 0   ! frozen fraction of the above
  integer, public :: ix_out_tref    = 0   ! 2 m temperature
  integer, public :: ix_out_qref    = 0   ! 2 m specific humidity
  integer, public :: ix_out_u10     = 0   ! 10 m zonal wind
  integer, public :: ix_out_v10     = 0   ! 10 m meridional wind

  !----------------------------------------------------------------------------
  ! Hybrid-coordinate interface between the lowest emulator layer and the layer
  ! above it: p_interface = ak_bot + bk_bot * PS.  Used to place the lowest
  ! layer in height for the surface-flux reference level.  Both ACE2-EAMv3 and
  ! SamudrACE-E3SMv3 use the same 8-layer vertical coarsening of EAMv3.
  !----------------------------------------------------------------------------
  real(R8), public :: eatm_ak_bot = 2328.474853515625_R8  ! Pa
  real(R8), public :: eatm_bk_bot = 0.8722758889198303_R8 ! 1

  !----------------------------------------------------------------------------
  ! Emulator timestep in seconds (the cadence at which inference is called).
  !----------------------------------------------------------------------------
  integer, public :: eatm_model_dt = 6 * 60 * 60

  character(len=CL), public :: eatm_emulator_name = 'unset'

  public :: eatm_channels_init
  public :: eatm_channels_final
  public :: eatm_channel_index
  public :: eatm_channel_metadata

contains

  !=============================================================================
  subroutine eatm_channels_init(emulator, logunit)

    ! Build the channel table for the named emulator and resolve every index
    ! EATM depends on.  Aborts if the emulator is unknown or if a channel EATM
    ! cannot do without is missing.

    character(len=*), intent(in) :: emulator
    integer,          intent(in) :: logunit

    integer :: k
    character(len=len(emulator)) :: lname
    character(len=*), parameter  :: subname = '(eatm_channels_init) '

    lname = to_lower(emulator)

    select case (trim(lname))
    case ('ace2-eamv3')
       call set_table_ace2_eamv3()
    case ('samudrace-e3smv3')
       call set_table_samudrace_e3smv3()
    case default
       call shr_sys_abort(trim(subname)//' ERROR: unknown eatm_emulator "'// &
            trim(emulator)//'". Known: ACE2-EAMv3, SamudrACE-E3SMv3')
    end select

    eatm_emulator_name = trim(emulator)

    !--- boundary / forcing inputs EATM has to supply itself ---
    ix_in_landfrac = eatm_channel_index(in_names, 'LANDFRAC')
    ix_in_ocnfrac  = eatm_channel_index(in_names, 'OCNFRAC')
    ix_in_icefrac  = eatm_channel_index(in_names, 'ICEFRAC')
    ix_in_phis     = eatm_channel_index(in_names, 'PHIS')
    ix_in_solin    = eatm_channel_index(in_names, 'SOLIN')
    ix_in_ps       = eatm_channel_index(in_names, 'PS')
    ix_in_ts       = eatm_channel_index(in_names, 'TS')

    !--- outputs the coupler needs ---
    ix_out_ps   = eatm_channel_index(out_names, 'PS')
    ix_out_ts   = eatm_channel_index(out_names, 'TS')
    ix_out_tbot = eatm_channel_index(out_names, 'T_7')
    ix_out_ubot = eatm_channel_index(out_names, 'U_7')
    ix_out_vbot = eatm_channel_index(out_names, 'V_7')
    ix_out_flds = eatm_channel_index(out_names, 'FLDS')
    ix_out_fsds = eatm_channel_index(out_names, 'FSDS')

    ! lowest-layer total water: named differently by the two checkpoints
    ix_out_qbot = eatm_channel_index(out_names, 'STW_7')
    if (ix_out_qbot == 0) &
         ix_out_qbot = eatm_channel_index(out_names, 'specific_total_water_7')

    ! upward surface shortwave: only needed for the diagnostic net flux
    ix_out_fsus = eatm_channel_index(out_names, 'FSUS')
    if (ix_out_fsus == 0) &
         ix_out_fsus = eatm_channel_index(out_names, 'surface_upward_shortwave_flux')

    ix_out_precip = eatm_channel_index(out_names, 'surface_precipitation_rate')
    ix_out_snow   = eatm_channel_index(out_names, 'frozen_precipitation_rate')

    ix_out_tref = eatm_channel_index(out_names, 'Tat2m')
    ix_out_qref = eatm_channel_index(out_names, 'Qat2m')
    ix_out_u10  = eatm_channel_index(out_names, 'Uat10m')
    ix_out_v10  = eatm_channel_index(out_names, 'Vat10m')

    !--- channels EATM genuinely cannot run without ---
    call require(ix_in_landfrac, 'LANDFRAC', 'input')
    call require(ix_in_ocnfrac,  'OCNFRAC',  'input')
    call require(ix_in_icefrac,  'ICEFRAC',  'input')
    call require(ix_in_phis,     'PHIS',     'input')
    call require(ix_in_solin,    'SOLIN',    'input')
    call require(ix_out_ps,      'PS',       'output')
    call require(ix_out_ts,      'TS',       'output')
    call require(ix_out_tbot,    'T_7',      'output')
    call require(ix_out_ubot,    'U_7',      'output')
    call require(ix_out_vbot,    'V_7',      'output')
    call require(ix_out_qbot,    'STW_7/specific_total_water_7', 'output')
    call require(ix_out_flds,    'FLDS',     'output')
    call require(ix_out_fsds,    'FSDS',     'output')
    call require(ix_out_precip,  'surface_precipitation_rate', 'output')

    !--- prognostic feedback map: any input whose name is also an output is
    !--- carried forward from the previous inference.
    allocate(in_from_out(n_input_channels))
    in_from_out(:) = 0
    do k = 1, n_input_channels
       in_from_out(k) = eatm_channel_index(out_names, trim(in_names(k)))
    end do
    ! Boundary conditions are owned by EATM even when the emulator also
    ! predicts them (TS is blended with the coupler's surface temperature;
    ! the surface fractions come from the coupler).
    if (ix_in_ts       > 0) in_from_out(ix_in_ts)       = 0
    if (ix_in_landfrac > 0) in_from_out(ix_in_landfrac) = 0
    if (ix_in_ocnfrac  > 0) in_from_out(ix_in_ocnfrac)  = 0
    if (ix_in_icefrac  > 0) in_from_out(ix_in_icefrac)  = 0

    write(logunit,'(a)')    '(eatm_channels_init) --------------------------------'
    write(logunit,'(2a)')   '(eatm_channels_init) emulator          = ', trim(eatm_emulator_name)
    write(logunit,'(a,i5)') '(eatm_channels_init) n_input_channels  = ', n_input_channels
    write(logunit,'(a,i5)') '(eatm_channels_init) n_output_channels = ', n_output_channels
    write(logunit,'(a,i5)') '(eatm_channels_init) n_forcing_channels= ', n_forcing_channels
    write(logunit,'(a,i5)') '(eatm_channels_init) prognostic feedbacks = ', count(in_from_out > 0)
    write(logunit,'(a)')    '(eatm_channels_init) --------------------------------'

  contains

    subroutine require(idx, name, kind)
      integer,          intent(in) :: idx
      character(len=*), intent(in) :: name, kind
      if (idx <= 0) call shr_sys_abort(trim(subname)//' ERROR: emulator "'// &
           trim(eatm_emulator_name)//'" has no '//trim(kind)//' channel '//trim(name))
    end subroutine require

  end subroutine eatm_channels_init

  !=============================================================================
  subroutine eatm_channels_final()
    if (allocated(in_names))      deallocate(in_names)
    if (allocated(out_names))     deallocate(out_names)
    if (allocated(forcing_names)) deallocate(forcing_names)
    if (allocated(in_from_out))   deallocate(in_from_out)
    n_input_channels   = 0
    n_output_channels  = 0
    n_forcing_channels = 0
  end subroutine eatm_channels_final

  !=============================================================================
  integer function eatm_channel_index(names, target)

    ! Position of `target` in `names`, or 0 when absent.  Case sensitive: the
    ! channel names come straight out of the checkpoint.

    character(len=*), intent(in) :: names(:)
    character(len=*), intent(in) :: target

    integer :: k

    eatm_channel_index = 0
    do k = 1, size(names)
       if (trim(names(k)) == trim(target)) then
          eatm_channel_index = k
          return
       end if
    end do

  end function eatm_channel_index

  !=============================================================================
  subroutine eatm_channel_metadata(name, long_name, units)

    ! CF-ish metadata for a channel, used when EATM writes restart files.
    ! Level-indexed channels are matched by prefix so the table stays short.

    character(len=*), intent(in)  :: name
    character(len=*), intent(out) :: long_name
    character(len=*), intent(out) :: units

    character(len=1) :: lev

    long_name = trim(name)
    units     = 'unknown'
    lev       = '?'

    if (len_trim(name) > 2) lev = name(len_trim(name):len_trim(name))

    if (starts_with(name, 'T_')) then
       long_name = 'Temperature, emulator layer '//lev ; units = 'K'
    else if (starts_with(name, 'U_')) then
       long_name = 'Zonal wind, emulator layer '//lev ; units = 'm/s'
    else if (starts_with(name, 'V_')) then
       long_name = 'Meridional wind, emulator layer '//lev ; units = 'm/s'
    else if (starts_with(name, 'STW_') .or. &
             starts_with(name, 'specific_total_water_')) then
       long_name = 'Specific total water, emulator layer '//lev ; units = 'kg/kg'
    else
       select case (trim(name))
       case ('PS')       ; long_name = 'Surface pressure'                     ; units = 'Pa'
       case ('TS')       ; long_name = 'Surface temperature (radiative)'      ; units = 'K'
       case ('PHIS')     ; long_name = 'Surface geopotential'                 ; units = 'm2/s2'
       case ('SOLIN')    ; long_name = 'Solar insolation at TOA'              ; units = 'W/m2'
       case ('LANDFRAC') ; long_name = 'Fraction of sfc area covered by land' ; units = '1'
       case ('OCNFRAC')  ; long_name = 'Fraction of sfc area covered by ocean'; units = '1'
       case ('ICEFRAC')  ; long_name = 'Fraction of sfc area covered by ice'  ; units = '1'
       case ('LHFLX')    ; long_name = 'Surface latent heat flux'             ; units = 'W/m2'
       case ('SHFLX')    ; long_name = 'Surface sensible heat flux'           ; units = 'W/m2'
       case ('FLDS')     ; long_name = 'Downward longwave flux at surface'    ; units = 'W/m2'
       case ('FLUT')     ; long_name = 'Upward longwave flux at TOA'          ; units = 'W/m2'
       case ('FSDS')     ; long_name = 'Downward shortwave flux at surface'   ; units = 'W/m2'
       case ('TAUX')     ; long_name = 'Zonal surface stress'                 ; units = 'N/m2'
       case ('TAUY')     ; long_name = 'Meridional surface stress'            ; units = 'N/m2'
       case ('Tat2m')    ; long_name = 'Temperature at 2 m'                   ; units = 'K'
       case ('Qat2m')    ; long_name = 'Specific humidity at 2 m'             ; units = 'kg/kg'
       case ('Uat10m')   ; long_name = 'Zonal wind at 10 m'                   ; units = 'm/s'
       case ('Vat10m')   ; long_name = 'Meridional wind at 10 m'              ; units = 'm/s'
       case ('DTENDTTW') ; long_name = 'Tendency of total water path from advection' ; units = 'kg/m2/s'
       case ('FLUS', 'surface_upward_longwave_flux')
          long_name = 'Upward longwave flux at surface'                       ; units = 'W/m2'
       case ('FSUS', 'surface_upward_shortwave_flux')
          long_name = 'Upward shortwave flux at surface'                      ; units = 'W/m2'
       case ('FSUTOA', 'top_of_atmos_upward_shortwave_flux')
          long_name = 'Upward shortwave flux at TOA'                          ; units = 'W/m2'
       case ('surface_precipitation_rate')
          long_name = 'Surface precipitation rate (all phases)'               ; units = 'kg/m2/s'
       case ('frozen_precipitation_rate')
          ! the checkpoint metadata says m/s, inherited from EAM's PRECS, but
          ! the data are kg/m2/s -- see eatm_frzprec_units in the namelist
          ! definition for the evidence
          long_name = 'Surface frozen precipitation rate (water equivalent)'  ; units = 'kg/m2/s'
       case ('tendency_of_total_water_path_due_to_advection')
          long_name = 'Tendency of total water path from advection'           ; units = 'kg/m2/s'
       end select
    end if

  end subroutine eatm_channel_metadata

  !=============================================================================
  ! Channel tables
  !=============================================================================

  subroutine set_table_ace2_eamv3()

    ! ACE2-EAMv3 (atmosphere-only, deterministic SFNO, prescribed SST).
    ! Matches ace2_EAMv3_ckpt*.tar traced with
    !   trace.py <ckpt> --add-normalization --add-corrector [--device cuda]
    ! 39 state inputs, 1 next-step forcing channel (SOLIN), 44 outputs.

    integer :: k, n

    n_input_channels   = 39
    n_output_channels  = 44
    n_forcing_channels = 1

    allocate(in_names(n_input_channels))
    allocate(out_names(n_output_channels))
    allocate(forcing_names(n_forcing_channels))

    in_names(1) = 'LANDFRAC'
    in_names(2) = 'OCNFRAC'
    in_names(3) = 'ICEFRAC'
    in_names(4) = 'PHIS'
    in_names(5) = 'SOLIN'
    in_names(6) = 'PS'
    in_names(7) = 'TS'
    n = 7
    do k = 0, 7 ; n = n + 1 ; in_names(n) = 'T_'//digit(k)                    ; end do
    do k = 0, 7 ; n = n + 1 ; in_names(n) = 'specific_total_water_'//digit(k) ; end do
    do k = 0, 7 ; n = n + 1 ; in_names(n) = 'U_'//digit(k)                    ; end do
    do k = 0, 7 ; n = n + 1 ; in_names(n) = 'V_'//digit(k)                    ; end do

    out_names(1) = 'PS'
    out_names(2) = 'TS'
    n = 2
    do k = 0, 7 ; n = n + 1 ; out_names(n) = 'T_'//digit(k)                    ; end do
    do k = 0, 7 ; n = n + 1 ; out_names(n) = 'specific_total_water_'//digit(k) ; end do
    do k = 0, 7 ; n = n + 1 ; out_names(n) = 'U_'//digit(k)                    ; end do
    do k = 0, 7 ; n = n + 1 ; out_names(n) = 'V_'//digit(k)                    ; end do
    out_names(35) = 'LHFLX'
    out_names(36) = 'SHFLX'
    out_names(37) = 'surface_precipitation_rate'
    out_names(38) = 'surface_upward_longwave_flux'
    out_names(39) = 'FLUT'
    out_names(40) = 'FLDS'
    out_names(41) = 'FSDS'
    out_names(42) = 'surface_upward_shortwave_flux'
    out_names(43) = 'top_of_atmos_upward_shortwave_flux'
    out_names(44) = 'tendency_of_total_water_path_due_to_advection'

    forcing_names(1) = 'SOLIN_next_step'

  end subroutine set_table_ace2_eamv3

  !-----------------------------------------------------------------------------
  subroutine set_table_samudrace_e3smv3()

    ! Atmosphere half of SamudrACE-E3SMv3 (NoiseConditionedSFNO, stochastic).
    ! Matches SamudrACE-E3SMv3-atmosphere.tar (or the atmosphere extracted from
    ! the coupled checkpoint) traced with
    !   trace.py <ckpt> --add-normalization --add-corrector [--device cuda]
    ! 43 state inputs, 1 next-step forcing channel (SOLIN), 51 outputs.
    !
    ! Relative to ACE2-EAMv3: specific_total_water_* -> STW_*, the surface and
    ! TOA flux channels get their EAM names (FLUS/FSUS/FSUTOA/DTENDTTW), and
    ! the model gains near-surface diagnostics (Tat2m/Qat2m/Uat10m/Vat10m,
    ! which are also inputs), surface stresses (TAUX/TAUY) and an explicit
    ! frozen precipitation rate.

    integer :: k, n

    n_input_channels   = 43
    n_output_channels  = 51
    n_forcing_channels = 1

    allocate(in_names(n_input_channels))
    allocate(out_names(n_output_channels))
    allocate(forcing_names(n_forcing_channels))

    in_names(1) = 'LANDFRAC'
    in_names(2) = 'OCNFRAC'
    in_names(3) = 'ICEFRAC'
    in_names(4) = 'PHIS'
    in_names(5) = 'SOLIN'
    in_names(6) = 'PS'
    in_names(7) = 'TS'
    n = 7
    do k = 0, 7 ; n = n + 1 ; in_names(n) = 'T_'//digit(k)   ; end do
    do k = 0, 7 ; n = n + 1 ; in_names(n) = 'STW_'//digit(k) ; end do
    do k = 0, 7 ; n = n + 1 ; in_names(n) = 'U_'//digit(k)   ; end do
    do k = 0, 7 ; n = n + 1 ; in_names(n) = 'V_'//digit(k)   ; end do
    in_names(40) = 'Qat2m'
    in_names(41) = 'Uat10m'
    in_names(42) = 'Vat10m'
    in_names(43) = 'Tat2m'

    out_names(1) = 'PS'
    out_names(2) = 'TS'
    n = 2
    do k = 0, 7 ; n = n + 1 ; out_names(n) = 'T_'//digit(k)   ; end do
    do k = 0, 7 ; n = n + 1 ; out_names(n) = 'STW_'//digit(k) ; end do
    do k = 0, 7 ; n = n + 1 ; out_names(n) = 'U_'//digit(k)   ; end do
    do k = 0, 7 ; n = n + 1 ; out_names(n) = 'V_'//digit(k)   ; end do
    out_names(35) = 'LHFLX'
    out_names(36) = 'SHFLX'
    out_names(37) = 'surface_precipitation_rate'
    out_names(38) = 'frozen_precipitation_rate'
    out_names(39) = 'FLUS'
    out_names(40) = 'FLUT'
    out_names(41) = 'FLDS'
    out_names(42) = 'FSDS'
    out_names(43) = 'FSUS'
    out_names(44) = 'FSUTOA'
    out_names(45) = 'DTENDTTW'
    out_names(46) = 'TAUX'
    out_names(47) = 'TAUY'
    out_names(48) = 'Qat2m'
    out_names(49) = 'Uat10m'
    out_names(50) = 'Vat10m'
    out_names(51) = 'Tat2m'

    forcing_names(1) = 'SOLIN_next_step'

  end subroutine set_table_samudrace_e3smv3

  !=============================================================================
  ! Small helpers
  !=============================================================================

  character(len=1) function digit(k)
    integer, intent(in) :: k
    digit = achar(iachar('0') + k)
  end function digit

  logical function starts_with(string, prefix)
    character(len=*), intent(in) :: string, prefix
    starts_with = .false.
    if (len_trim(string) < len_trim(prefix)) return
    starts_with = (string(1:len_trim(prefix)) == trim(prefix))
  end function starts_with

  function to_lower(string) result(lowered)
    character(len=*), intent(in)  :: string
    character(len=len(string))    :: lowered
    integer :: k, ic
    lowered = string
    do k = 1, len_trim(string)
       ic = iachar(string(k:k))
       if (ic >= iachar('A') .and. ic <= iachar('Z')) then
          lowered(k:k) = achar(ic + 32)
       end if
    end do
  end function to_lower

end module eatm_channels_mod
