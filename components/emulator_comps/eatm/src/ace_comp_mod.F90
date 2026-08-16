module ace_comp_mod

  !-----------------------------------------------------------------------------
  ! Drives a traced ACE-family emulator through FTorch and translates between
  ! its channel layout and the fields the MCT coupler exchanges.
  !
  ! The emulator advances on its own timestep (6 h for both ACE2-EAMv3 and the
  ! SamudrACE-E3SMv3 atmosphere).  The coupler runs much faster than that, so
  ! inference happens only on emulator-step boundaries and the coupler is handed
  ! a linear interpolation between the two bracketing emulator states.
  !
  ! Which physical field lives in which channel is not hardwired here -- see
  ! eatm_channels_mod.  Everything below is written against named indices so a
  ! different checkpoint only needs a new table there plus a namelist change.
  !-----------------------------------------------------------------------------

  use esmf
  use eatmMod
  use eatmIO
  use eatm_channels_mod
  use mct_mod
  use seq_timemgr_mod, only: seq_timemgr_EClockGetData
  use shr_const_mod
  use shr_kind_mod, only: R4=>SHR_KIND_R4, R8=>SHR_KIND_R8, CS=>SHR_KIND_CS, CL=>SHR_KIND_CL, IN=>SHR_KIND_IN
  use shr_sys_mod,  only: shr_sys_flush, shr_sys_abort
  use shr_orb_mod,  only: shr_orb_decl, shr_orb_cosz
  use shr_cal_mod,  only: shr_cal_date2julian

  use ftorch, only: &
    torch_kCPU, &
    torch_kCUDA, &
    torch_model, &
    torch_tensor, &
    torch_delete, &
    torch_kFloat32, &
    torch_model_load, &
    torch_model_forward, &
    torch_tensor_from_blob

  use, intrinsic :: iso_c_binding, only: c_loc, c_int64_t, c_int

  implicit none
  private ! except

  !--------------------------------------------------------------------------
  ! Public interfaces
  !--------------------------------------------------------------------------
  public :: ace_comp_init
  public :: ace_comp_run
  public :: ace_comp_finalize

  !--------------------------------------------------------------------------
  ! Private module data
  !--------------------------------------------------------------------------
  real(R8), parameter :: rdair  = SHR_CONST_RDAIR  ! dry air gas constant   ~ J/K/kg
  real(R8), parameter :: tKFrz  = SHR_CONST_TKFRZ
  real(R8), parameter :: grav   = SHR_CONST_G      ! gravitational acceleration m/s2
  real(R8), parameter :: rhofw  = SHR_CONST_RHOFW  ! density of fresh water kg/m3

  real(R8), parameter :: solar_const = 1368.22_R8  ! total solar irradiance (W/m2), matches RRTMG

  ! Partitioning of the downwelling surface shortwave into the four bands the
  ! coupler expects.  Same fixed split datm uses; see eatm/REVIEW.md.
  real(R8), parameter :: frac_swvdr = 0.28_R8
  real(R8), parameter :: frac_swndr = 0.31_R8
  real(R8), parameter :: frac_swvdf = 0.24_R8
  real(R8), parameter :: frac_swndf = 0.17_R8

  ! Set up Torch data structures
  type(torch_model) :: ace_model
  type(torch_tensor), dimension(1) :: input_tensor
  type(torch_tensor), dimension(1) :: output_tensor

  integer(c_int) :: tensor_layout(4)
  integer(c_int64_t) :: input_tensor_shape(4)
  integer(c_int64_t) :: output_tensor_shape(4)

  integer(c_int) :: model_device      ! torch_kCPU or torch_kCUDA
  integer        :: n_tensor_in       ! channels handed to the traced graph
  logical        :: frzprec_is_depth  ! frozen precip channel is m/s, not kg/m2/s

  save

  !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CONTAINS

  subroutine ace_comp_init(EClock, ggrid, read_restart)

    implicit none

    type(ESMF_Clock), intent(in)          :: EClock
    type(mct_gGrid),  intent(in), pointer :: ggrid
    logical,          intent(in)          :: read_restart

    integer     :: i, j, k     ! loop indicies
    real(R8)    :: t_frac      ! frac through eatm timestep
    integer     :: t_modulo    ! int remainder of curr. time over eatm dt
    integer(in) :: CurrentTOD  ! model sec into model date
    character(len=*), parameter :: subname = '(ace_comp_init) '

    !--- how many channels the traced graph is handed ---
    n_tensor_in = n_input_channels
    if (eatm_pass_forcing) n_tensor_in = n_input_channels + n_forcing_channels

    frzprec_is_depth = (trim(eatm_frzprec_units) == 'm/s')

    input_tensor_shape = [ &
      int(1, kind=c_int64_t), &
      int(n_tensor_in, kind=c_int64_t), &
      int(lsize_y, kind=c_int64_t), &
      int(lsize_x, kind=c_int64_t) &
    ]

    output_tensor_shape = [ &
      int(1, kind=c_int64_t), &
      int(n_output_channels, kind=c_int64_t), &
      int(lsize_y, kind=c_int64_t), &
      int(lsize_x, kind=c_int64_t) &
    ]

    tensor_layout = [1_c_int, 2_c_int, 4_c_int, 3_c_int]

    select case (trim(eatm_model_device))
    case ('gpu', 'GPU', 'cuda', 'CUDA')
       model_device = torch_kCUDA
    case ('cpu', 'CPU')
       model_device = torch_kCPU
    case default
       call shr_sys_abort(trim(subname)//' ERROR: eatm_model_device must be '// &
            '"cpu" or "gpu", got "'//trim(eatm_model_device)//'"')
    end select

    if (len_trim(eatm_model_file) == 0) &
         call shr_sys_abort(trim(subname)//' ERROR: eatm_model_file is not set')

    write(logunit_atm,*) trim(subname)//'loading traced model ', trim(eatm_model_file)
    write(logunit_atm,*) trim(subname)//'device                = ', trim(eatm_model_device)
    write(logunit_atm,*) trim(subname)//'input tensor channels = ', n_tensor_in
    call shr_sys_flush(logunit_atm)

    ! load the traced model
    call torch_model_load(ace_model, trim(eatm_model_file), model_device)

    if (read_restart) then

      call seq_timemgr_EClockGetData( EClock, curr_tod=CurrentTOD )

      ! int remainder (in sec) of coupler timestep relative to ACE timestep
      t_modulo = mod(CurrentTOD, eatm_model_dt)
      ! turn integer remainder into fraction through ACE timestep
      t_frac = real(t_modulo, kind=R8) / real(eatm_model_dt, kind=R8)

      do k = 1, n_output_channels
        do j = 1, lsize_y
          do i = 1, lsize_x
            net_outputs(1, k, i, j) = eatm_intrp%t_im1(k, i, j) + &
                t_frac * (eatm_intrp%t_ip1(k, i, j) - eatm_intrp%t_im1(k, i, j))
          end do
        end do
      end do

    else
      call ace_compute_solin(EClock, ggrid)

      call ace_inference()

      ! fill both time levels of intrp struct with restart data
      do k = 1, n_output_channels
        do j = 1, lsize_y
          do i = 1, lsize_x
            eatm_intrp%t_im1(k, i, j) = net_outputs(1, k, i, j)
            eatm_intrp%t_ip1(k, i, j) = net_outputs(1, k, i, j)
          end do
        end do
      end do
    endif

    ! using restart data from ACE set the fields passed to the coupler
    call ace_eatm_export(ggrid, verbose=.true.)

  end subroutine ace_comp_init

  subroutine ace_comp_run(EClock, ggrid)
    ! !DESCRIPTION: run method for ace model
    implicit none

    ! !INPUT/OUTPUT PARAMETERS:
    type(ESMF_Clock), intent(in) :: EClock
    type(mct_gGrid), intent(in), pointer :: ggrid

    !--- local ---
    integer     :: i, j, k           ! loop indicies
    real(R8)    :: t_frac            ! frac of cpl_t / eatm_t
    integer     :: t_modulo          ! frac of cpl_t / eatm_t
    integer(in) :: cpl_idt           ! integer timestep
    integer(in) :: stepno            ! step number
    integer(in) :: CurrentYMD        ! model date
    integer(in) :: CurrentTOD        ! model sec into model date

    call seq_timemgr_EClockGetData( EClock, curr_ymd=CurrentYMD, curr_tod=CurrentTOD)
    call seq_timemgr_EClockGetData( EClock, stepno=stepno, dtime=cpl_idt)

    ! integer remainder (in sec) of coupler timestep relative to ACE timestep
    t_modulo = mod(CurrentTOD, eatm_model_dt)

    if (t_modulo .eq. 0) then

      ! One line per emulator step, not per coupler step: at ATM_NCPL=48 the
      ! latter is 48 flushed writes a day, ~100 MB of atm.log over five years.
      write(logunit_atm, '(a,i9,a,i9.8,a,i6,a,i7,a)') &
           'eatm step ', stepno, ' date ', CurrentYMD, ' tod ', CurrentTOD, &
           ' (cpl dt ', cpl_idt, ' s) -- advancing emulator'
      call shr_sys_flush(logunit_atm)

      ! Feed the emulator its own state *at this time*, which is the prediction
      ! made one emulator step ago (t_ip1).  net_outputs currently still holds
      ! the field the coupler was handed at the end of the previous coupler
      ! step, which is a partial interpolation towards t_ip1 and is therefore
      ! not a state the emulator was ever trained to consume.
      call ace_eatm_import(eatm_intrp%t_ip1)
      call ace_compute_solin(EClock, ggrid)

      call ace_inference()

      ! advance the time levels: old t_ip1 (state at current time T)
      ! becomes t_im1; new inference (state at T+dt) becomes t_ip1.
      ! Interpolation between them gives smooth transition over [T, T+dt].
      do k = 1, n_output_channels
        do j = 1, lsize_y
          do i = 1, lsize_x
            eatm_intrp%t_im1(k, i, j) = eatm_intrp%t_ip1(k, i, j)
            eatm_intrp%t_ip1(k, i, j) = net_outputs(1, k, i, j)
          end do
        end do
      end do

    end if

    t_frac = real(t_modulo, kind=r8) / real(eatm_model_dt, kind=r8)

    ! time interpolate the results
    do k = 1, n_output_channels
      do j = 1, lsize_y
        do i = 1, lsize_x
          net_outputs(1, k, i, j) = eatm_intrp%t_im1(k, i, j) + &
              t_frac * (eatm_intrp%t_ip1(k, i, j) - eatm_intrp%t_im1(k, i, j))
        end do
      end do
    end do

    call ace_eatm_export(ggrid, verbose=(t_modulo == 0))

  end subroutine ace_comp_run

  subroutine ace_comp_finalize()
    call torch_delete(ace_model)
  end subroutine ace_comp_finalize

  !===============================================================================
  subroutine ace_inference()

    !----------------------------------------------------------------
    ! Copy net_inputs (plus, optionally, the next-step forcing block)
    ! into the tensor staging buffer, run the traced model, and leave
    ! the denormalized, corrected result in net_outputs.
    !
    ! Normalization, the atmosphere correctors and (if it was traced in)
    ! the ocean SST prescription all live inside the TorchScript graph --
    ! see scripts/trace_ace_model.py in the ACE repository.
    !----------------------------------------------------------------
    implicit none

    integer :: k

    net_inputs_nn(1, 1:n_input_channels, :, :) = net_inputs(1, :, :, :)

    if (eatm_pass_forcing) then
      ! The only next-step forcing these checkpoints declare is SOLIN, and
      ! ace_compute_solin already puts the next-step value into the SOLIN
      ! state channel (that is what fme's next_step_forcing_names means), so
      ! the appended copy is the same field.
      do k = 1, n_forcing_channels
        net_inputs_nn(1, n_input_channels + k, :, :) = net_inputs(1, ix_in_solin, :, :)
      end do
    end if

    call torch_tensor_from_blob(&
      input_tensor(1), &
      c_loc(net_inputs_nn), &
      ndims=4_c_int, &
      tensor_shape=input_tensor_shape, &
      layout=tensor_layout, &
      dtype=torch_kFloat32, &
      device_type=model_device &
    )
    call torch_tensor_from_blob(&
      output_tensor(1), &
      c_loc(net_outputs), &
      ndims=4_c_int, &
      tensor_shape=output_tensor_shape, &
      layout=tensor_layout, &
      dtype=torch_kFloat32, &
      device_type=torch_kCPU &
    )

    call torch_model_forward(ace_model, input_tensor, output_tensor)

    ! Clean up C++ pointers
    call torch_delete(input_tensor)
    call torch_delete(output_tensor)

  end subroutine ace_inference

  !===============================================================================
  subroutine ace_eatm_import(state)
    !----------------------------------------------------------------
    ! !DESCRIPTION:
    ! Build the emulator's input state for the step about to be taken.
    !
    !   * every input channel that is also an output channel is carried
    !     forward from `state` (the emulator's own prediction for now),
    !   * the surface fractions come from the coupler, EXCEPT that the land
    !     fraction is the part of the cell no surface model covers -- see below,
    !   * TS is the coupler's merged surface temperature completed with the
    !     emulator's own TS over that same uncovered fraction,
    !   * PHIS persists from the initial condition / restart,
    !   * SOLIN is computed by ace_compute_solin.
    !
    ! The coupler's own `Sf_lfrac` cannot be used for either. It is the fraction
    ! claimed by the *land model*, and in the GMPAS-EATM compset the land model
    ! is a stub, so `Sf_lfrac` is identically zero everywhere. Over land that
    ! leaves all three of lfrac, ofrac and ifrac at zero, and the coupler's
    ! merged surface temperature `Sx_t` -- built as
    ! lfrac*Sl_t + ifrac*Si_t + ofrac*So_t -- at exactly 0 K over 26% of the
    ! globe. Passing that through told the emulator the surface was at absolute
    ! zero over every land point and that the planet had no land at all, which
    ! drove a 12%-of-area cold pool below 200 K.
    !
    ! So the land fraction EATM reports is the deficit, 1 - ofrac - ifrac, and
    ! the emulator owns the surface temperature there. This is correct whether
    ! or not a land model is running: with ELM active, lfrac is nonzero, the
    ! fractions sum to one, the deficit is zero, and `Sx_t` is passed through
    ! unchanged.
    !----------------------------------------------------------------
    implicit none

    real(R4), intent(in) :: state(:,:,:)   ! (channel, x, y)

    ! !LOCAL VARIABLES:
    integer  :: i, j, k
    real(R8) :: covered   ! cell fraction the surface models accounted for
    real(R8) :: deficit   ! the rest, which the emulator owns

    do k = 1, n_input_channels
      if (in_from_out(k) > 0) then
        do j = 1, lsize_y
          do i = 1, lsize_x
            net_inputs(1, k, i, j) = state(in_from_out(k), i, j)
          enddo
        enddo
      end if
    end do

    do j = 1, lsize_y
      do i = 1, lsize_x

        covered = min(max(ocnfrac(i, j) + icefrac(i, j) + lndfrac(i, j), 0.0_R8), 1.0_R8)
        deficit = 1.0_R8 - covered

        net_inputs(1, ix_in_landfrac, i, j) = real(lndfrac(i, j) + deficit, R4)
        net_inputs(1, ix_in_ocnfrac,  i, j) = real(ocnfrac(i, j), R4)
        net_inputs(1, ix_in_icefrac,  i, j) = real(icefrac(i, j), R4)

        if (ix_in_ts > 0) then
          net_inputs(1, ix_in_ts, i, j) = real( &
               ts(i, j) + deficit * real(state(ix_out_ts, i, j), R8), R4)
        end if

      enddo
    enddo

    ! Two lines, ranges only.  Sx_t straight from the coupler: a minimum of 0 K
    ! is expected and fine, it is the cells no surface model covers, which the
    ! deficit term fills in.  What must never be 0 is the TS handed to the
    ! emulator, which is why both are reported side by side.
    write(logunit_atm, '(a,4(1x,a,2f9.3))') '  cpl in ', &
         'Sx_t',  minval(ts),      maxval(ts),      &
         'lfrac', minval(lndfrac), maxval(lndfrac), &
         'ofrac', minval(ocnfrac), maxval(ocnfrac), &
         'ifrac', minval(icefrac), maxval(icefrac)
    if (ix_in_ts > 0) then
      write(logunit_atm, '(a,2(1x,a,2f9.3),1x,a,2es11.3)') '  net in ', &
           'LANDFRAC', minval(net_inputs(1, ix_in_landfrac, :, :)), &
                       maxval(net_inputs(1, ix_in_landfrac, :, :)), &
           'TS',       minval(net_inputs(1, ix_in_ts, :, :)),       &
                       maxval(net_inputs(1, ix_in_ts, :, :)),       &
           'shf',      minval(shf),     maxval(shf)
    end if
    call shr_sys_flush(logunit_atm)

  end subroutine ace_eatm_import

  !===============================================================================
  subroutine ace_eatm_export(ggrid, verbose)

    !----------------------------------------------------------------
    ! Turn the emulator's output channels into the state and flux fields the
    ! coupler expects from an atmosphere.
    !----------------------------------------------------------------
    implicit none

    type(mct_gGrid), pointer :: ggrid   ! unused; kept for interface symmetry
    logical, intent(in), optional :: verbose  ! log the field ranges

    integer  :: i, j
    real(R8) :: esat, p_int, tv, precip, snow, fsds_dn
    real(R8) :: raw
    logical  :: do_log
    logical  :: use_near_surface   ! export at eatm_ref_height, not the layer midpoint

    ! Counts of cells where a physically-required floor had to be applied to a
    ! predicted field.  An emulator predicting negative water or a negative
    ! downwelling flux is the failure mode most worth seeing, and silently
    ! clamping it makes it invisible.
    integer  :: n_clip_shum, n_clip_precip, n_clip_snow, n_clip_fsds, n_clip_swnet

    do_log = .false.
    if (present(verbose)) do_log = verbose

    ! Only available where the emulator predicts all four near-surface
    ! diagnostics -- SamudrACE-E3SMv3 does, ACE2-EAMv3 has no such channels.
    use_near_surface = (trim(eatm_surface_layer) == 'near_surface') .and. &
         .not. eatm_legacy_surface .and. &
         ix_out_tref > 0 .and. ix_out_qref > 0 .and. &
         ix_out_u10  > 0 .and. ix_out_v10  > 0

    n_clip_shum   = 0
    n_clip_precip = 0
    n_clip_snow   = 0
    n_clip_fsds   = 0
    n_clip_swnet  = 0

    do j = 1, lsize_y
      do i = 1, lsize_x

        pslv(i, j) = net_outputs(1, ix_out_ps, i, j)   ! PS; == SLP where PHIS == 0
        tbot(i, j) = net_outputs(1, ix_out_tbot, i, j)
        ubot(i, j) = net_outputs(1, ix_out_ubot, i, j)
        vbot(i, j) = net_outputs(1, ix_out_vbot, i, j)
        topo(i, j) = net_inputs(1, ix_in_phis, i, j) / grav

        ! pressure at the interface between the lowest emulator layer and the
        ! one above it
        p_int = eatm_ak_bot + eatm_bk_bot * pslv(i, j)

        if (use_near_surface) then
          !--- Hand the coupler a state at a genuine surface-layer height.
          !---
          !--- The emulator's lowest layer is ~900 hPa thick, so its midpoint is
          !--- ~450 m up.  shr_flux_atmOcn applies Monin-Obukhov constant-flux-
          !--- layer similarity between Sa_z and the surface, which is not valid
          !--- over that depth: the state arrives too cold and too dry and the
          !--- scheme returns too much latent and sensible heat.
          !---
          !--- Where the emulator predicts near-surface diagnostics we use them
          !--- instead, at the same 10 m reference height datm uses to force this
          !--- very ocean from JRA (datm_comp_mod.F90:1029, the IAF_JRA_1p5
          !--- datamode), including its pbot = pslv convention.  Wind is exactly
          !--- at 10 m; T and q are at 2 m, an inconsistency of a few tenths of a
          !--- kelvin -- against the ~1.5 K the 450 m dry-adiabatic reduction was
          !--- injecting.
          zbot(i, j) = eatm_ref_height
          ubot(i, j) = net_outputs(1, ix_out_u10, i, j)
          vbot(i, j) = net_outputs(1, ix_out_v10, i, j)
          tbot(i, j) = net_outputs(1, ix_out_tref, i, j)
          raw        = real(net_outputs(1, ix_out_qref, i, j), R8)
          if (raw < 0.0_R8) n_clip_shum = n_clip_shum + 1
          shum(i, j) = max(raw, 0.0_R8)
          pbot(i, j) = pslv(i, j)

        else if (eatm_legacy_surface) then
          !--- pre-review behaviour, retained so earlier runs can be reproduced:
          !--- the reference level is the layer top, its height is a standard
          !--- atmosphere pressure altitude above *sea level*, and the humidity
          !--- is saturated rather than predicted.
          pbot(i, j) = p_int
          ! https://en.wikipedia.org/wiki/Pressure_altitude w/ Pa --> hPa
          zbot(i, j) = 44307.694_R8 * ( 1.0_R8 - (pbot(i, j) / SHR_CONST_PSTD)**0.190284_R8 )
          esat = datm_shr_esat(tbot(i, j), tbot(i, j))
          shum(i, j) = (0.622_R8 * esat)/(pbot(i, j) - 0.378_R8 * esat)
        else
          !--- the layer-mean fields T_7/U_7/V_7/STW_7 belong at the layer
          !--- midpoint, and Sa_z is a height *above the surface*, so use the
          !--- hypsometric thickness from PS to the midpoint rather than a
          !--- pressure altitude above sea level.
          pbot(i, j) = 0.5_R8 * (pslv(i, j) + p_int)
          raw        = real(net_outputs(1, ix_out_qbot, i, j), R8)
          if (raw < 0.0_R8) n_clip_shum = n_clip_shum + 1
          shum(i, j) = max(raw, 0.0_R8)
          tv         = tbot(i, j) * (1.0_R8 + 0.608_R8 * shum(i, j))
          zbot(i, j) = (rdair * tv / grav) * log(pslv(i, j) / pbot(i, j))
        end if

        ptem(i, j) = tbot(i,j) * (pslv(i,j)/pbot(i,j))**(rdair/SHR_CONST_CPDAIR)
        dens(i, j) = pbot(i, j)  / (rdair * tbot(i, j) * (1 + 0.608_R8 * shum(i, j)))

        lwdn(i, j) = net_outputs(1, ix_out_flds, i, j)

        !--- precipitation: the emulator has no convective/large-scale split,
        !--- so everything is reported as large scale.
        raw = real(net_outputs(1, ix_out_precip, i, j), R8)
        if (raw < 0.0_R8) n_clip_precip = n_clip_precip + 1
        precip = max(raw, 0.0_R8)
        snowc(i, j) = 0.0_R8
        rainc(i, j) = 0.0_R8

        if (ix_out_snow > 0 .and. .not. eatm_legacy_surface) then
          raw = real(net_outputs(1, ix_out_snow, i, j), R8)
          if (raw < 0.0_R8) n_clip_snow = n_clip_snow + 1
          snow = max(raw, 0.0_R8)
          if (frzprec_is_depth) snow = snow * rhofw   ! m/s water equivalent -> kg/m2/s
          snow = min(snow, precip)
          snowl(i, j) = snow
          rainl(i, j) = precip - snow
        else
          ! no predicted frozen fraction: split on the lowest-layer temperature
          if (tbot(i, j) < tKFrz) then
            rainl(i, j) = 0.0_R8
            snowl(i, j) = precip
          else
            rainl(i, j) = precip
            snowl(i, j) = 0.0_R8
          endif
        end if

        !--- shortwave.  Faxa_swnet is diagnostic in the coupler (the net flux
        !--- the surface models see is rebuilt from the four downwelling bands
        !--- below and their own albedos), so report the emulator's actual net
        !--- flux when it predicts the upward component.
        raw = real(net_outputs(1, ix_out_fsds, i, j), R8)
        if (raw < 0.0_R8) n_clip_fsds = n_clip_fsds + 1
        fsds_dn = max(raw, 0.0_R8)
        swvdr(i, j) = fsds_dn * frac_swvdr
        swndr(i, j) = fsds_dn * frac_swndr
        swvdf(i, j) = fsds_dn * frac_swvdf
        swndf(i, j) = fsds_dn * frac_swndf

        if (ix_out_fsus > 0 .and. .not. eatm_legacy_surface) then
          raw = fsds_dn - max(real(net_outputs(1, ix_out_fsus, i, j), R8), 0.0_R8)
          if (raw < 0.0_R8) n_clip_swnet = n_clip_swnet + 1
          swnet(i, j) = max(raw, 0.0_R8)
        else
          swnet(i, j) = fsds_dn
        end if

      enddo
    enddo

    ! Only worth logging on emulator steps: in between, every field below is a
    ! linear interpolation between two states already reported, so at
    ! ATM_NCPL=48 eleven of every twelve blocks are redundant.
    if (.not. do_log) return

    write(logunit_atm, '(a,3(1x,a,2f9.2),1x,a,2es11.3)') '  net out', &
         'tbot', minval(tbot), maxval(tbot), &
         'zbot', minval(zbot), maxval(zbot), &
         'pbot', minval(pbot), maxval(pbot), &
         'shum', minval(shum), maxval(shum)
    write(logunit_atm, '(a,4(1x,a,2f9.2),2(1x,a,2es11.3))') '  net out', &
         'ubot',  minval(ubot),  maxval(ubot),  &
         'vbot',  minval(vbot),  maxval(vbot),  &
         'swnet', minval(swnet), maxval(swnet), &
         'lwdn',  minval(lwdn),  maxval(lwdn),  &
         'rainl', minval(rainl), maxval(rainl), &
         'snowl', minval(snowl), maxval(snowl)
    if (n_clip_shum + n_clip_precip + n_clip_snow + n_clip_fsds + n_clip_swnet > 0) then
      write(logunit_atm, '(a,5(1x,a,i0),a,i0,a)') '  clamped', &
           'shum=',   n_clip_shum,   'precip=', n_clip_precip, &
           'snow=',   n_clip_snow,   'fsds=',   n_clip_fsds,   &
           'swnet=',  n_clip_swnet,  ' (of ', lsize_x*lsize_y, ' cells)'
    end if
    call shr_sys_flush(logunit_atm)

  end subroutine ace_eatm_export

  !===============================================================================
  real(R8) function datm_shr_eSat(tK,tKbot)

    !--- arguments ---
    real(R8),intent(in) :: tK    ! temp used in polynomial calculation
    real(R8),intent(in) :: tKbot ! bottom atm temp

    !--- local ---
    real(R8)           :: t     ! tK converted to Celcius

    !--- coefficients for esat over water ---
    real(R8),parameter :: a0=6.107799961_R8
    real(R8),parameter :: a1=4.436518521e-01_R8
    real(R8),parameter :: a2=1.428945805e-02_R8
    real(R8),parameter :: a3=2.650648471e-04_R8
    real(R8),parameter :: a4=3.031240396e-06_R8
    real(R8),parameter :: a5=2.034080948e-08_R8
    real(R8),parameter :: a6=6.136820929e-11_R8

    !--- coefficients for esat over ice ---
    real(R8),parameter :: b0=6.109177956_R8
    real(R8),parameter :: b1=5.034698970e-01_R8
    real(R8),parameter :: b2=1.886013408e-02_R8
    real(R8),parameter :: b3=4.176223716e-04_R8
    real(R8),parameter :: b4=5.824720280e-06_R8
    real(R8),parameter :: b5=4.838803174e-08_R8
    real(R8),parameter :: b6=1.838826904e-10_R8

    !----------------------------------------------------------------------------
    ! use polynomials to calculate saturation vapor pressure and derivative with
    ! respect to temperature: over water when t > 0 c and over ice when t <= 0 c
    ! required to convert relative humidity to specific humidity
    !----------------------------------------------------------------------------

    t = min( 50.0_R8, max(-50.0_R8,(tK-tKfrz)) )
    if ( tKbot < tKfrz) then
       datm_shr_eSat = 100.0_R8*(b0+t*(b1+t*(b2+t*(b3+t*(b4+t*(b5+t*b6))))))
    else
       datm_shr_eSat = 100.0_R8*(a0+t*(a1+t*(a2+t*(a3+t*(a4+t*(a5+t*a6))))))
    end if

  end function datm_shr_eSat

  !===============================================================================
  subroutine ace_compute_solin(EClock, ggrid)
    !----------------------------------------------------------------
    ! Compute SOLIN (solar insolation at TOA) from orbital mechanics.
    ! SOLIN = S0 * eccf * max(0, cosz)
    !
    ! The ACE emulator predicts state at T+dt from inputs at T, and declares
    ! SOLIN in next_step_forcing_names -- meaning the SOLIN *input* channel
    ! carries the value at T+dt, not at T.  So SOLIN is computed for the
    ! prediction target time, which also makes the output FSDS consistent with
    ! the solar geometry at the time the state is handed to the coupler.
    !----------------------------------------------------------------
    implicit none
    type(ESMF_Clock), intent(in) :: EClock
    type(mct_gGrid),  intent(in), pointer :: ggrid

    integer(IN)       :: CurrentYMD, CurrentTOD
    character(len=CS) :: calendar
    real(R8)          :: julday
    real(R8)          :: delta, eccf
    real(R8)          :: lat_r, lon_r
    real(R8)          :: cosz_val, solin_val
    real(R8), parameter :: degtorad = SHR_CONST_PI / 180.0_R8

    integer     :: klat, klon, n, i, j
    real(R8), pointer :: yc(:), xc(:)

    call seq_timemgr_EClockGetData(EClock, curr_ymd=CurrentYMD, curr_tod=CurrentTOD)
    call seq_timemgr_EClockGetData(EClock, calendar=calendar)

    call shr_cal_date2julian(CurrentYMD, CurrentTOD, julday, calendar)

    ! Advance julday by one ACE timestep: the emulator predicts state
    ! at T+dt, so SOLIN must represent solar geometry at T+dt.
    julday = julday + real(eatm_model_dt, R8) / SHR_CONST_CDAY

    call shr_orb_decl(julday, orb_eccen, orb_mvelpp, orb_lambm0, orb_obliqr, delta, eccf)

    allocate(yc(lsize), xc(lsize))
    klat = mct_aVect_indexRA(ggrid%data, 'lat')
    klon = mct_aVect_indexRA(ggrid%data, 'lon')
    yc(:) = ggrid%data%rAttr(klat, :)
    xc(:) = ggrid%data%rAttr(klon, :)

    n = 0
    do j = 1, lsize_y
      do i = 1, lsize_x
        n = n + 1
        lat_r = yc(n) * degtorad
        lon_r = xc(n) * degtorad
        cosz_val = shr_orb_cosz(julday, lat_r, lon_r, delta)
        solin_val = solar_const * eccf * max(0.0_R8, cosz_val)
        net_inputs(1, ix_in_solin, i, j) = real(solin_val, R4)
      end do
    end do

    deallocate(yc, xc)

  end subroutine ace_compute_solin

end module ace_comp_mod
