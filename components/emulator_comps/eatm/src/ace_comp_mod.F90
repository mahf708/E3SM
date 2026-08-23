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
  use shr_emul_ice_mod, only: shr_emul_ice_get, shr_emul_ice_avail
  use shr_emul_ice_mod, only: shr_emul_ice_get_sst, shr_emul_ice_sst_avail

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
  ! Seed libtorch's generators (eatm_torch_seed.cpp).  FTorch has no interface
  ! to them, and without a seed the stochastic SamudrACE atmosphere cannot be
  ! A/B tested at all -- its run-to-run spread is of order 5 W/m2 over 20 days.
  !--------------------------------------------------------------------------
  interface
    subroutine eatm_torch_manual_seed(seed) bind(C, name='eatm_torch_manual_seed')
      import :: c_int64_t
      integer(c_int64_t), value :: seed
    end subroutine eatm_torch_manual_seed
  end interface

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

  !--------------------------------------------------------------------------
  ! Model time at which the emulator was last advanced.  The driver calls
  ! atm_init_mct twice for a prognostic atmosphere -- once in phase 1 and again
  ! in phase 2 after the ocean and ice have initialized -- and the ESMF clock
  ! is not advanced between them (driver-mct/main/cime_comp_mod.F90:1532 and
  ! :2446; ClockAdvance appears once in the whole driver, at :2826, at the top
  ! of the run loop).  The phase-2 call runs the full run method, so a guard of
  ! `mod(tod, dt) == 0` alone fires twice at tod = 0 and leaves the emulator
  ! state permanently one emulator step ahead of the coupler clock.
  !
  ! Recording the time of the last advance makes the advance idempotent in
  ! model time, which is the property that actually matters and does not depend
  ! on counting driver phases.
  !--------------------------------------------------------------------------
  integer(IN) :: last_adv_ymd = -1
  integer(IN) :: last_adv_tod = -1

  !--- cell areas (rad2), cached from the domain for the budget report ---
  real(R8), allocatable :: cell_area(:,:)
  real(R8), allocatable :: cell_lat(:,:)    ! for the latitude-band breakdown
  real(R8)              :: area_total = 0.0_R8

  !--------------------------------------------------------------------------
  ! Running sums of the coupler's applied surface exchange over the emulator
  ! interval currently open.
  !
  ! The emulator's flux channels are means over the whole 6 h step, but the
  ! coupler recomputes its fluxes every coupling step and atm_import_mct
  ! overwrites them each time.  Reading them once at the emulator boundary
  ! therefore compares a 6 h mean against a single 30 min sample -- exactly the
  ! kind of unlike-for-unlike comparison that made the first version of this
  ! report understate the coupler (see REVIEW.md #44).  Sampling four fixed
  ! phases a day happens to integrate a diurnal or semi-diurnal signal without
  ! bias, so the error is smaller than it looks, but it is not a budget.
  !
  ! These accumulate every coupling step and are reported and reset at the
  ! emulator boundary, which makes both columns interval means over the same
  ! interval.
  !--------------------------------------------------------------------------
  real(R8), allocatable :: acc_lhf(:,:)    ! latent   heat flux
  real(R8), allocatable :: acc_shf(:,:)    ! sensible heat flux
  real(R8), allocatable :: acc_lwup(:,:)   ! upward longwave at the surface
  real(R8), allocatable :: acc_wsx(:,:)    ! zonal stress
  real(R8), allocatable :: acc_wsy(:,:)    ! meridional stress
  real(R8), allocatable :: acc_swabs(:,:)  ! shortwave the surface really absorbs
  real(R8), allocatable :: acc_cov(:,:)    ! covered fraction, same cadence
  integer               :: acc_n = 0       ! coupling steps accumulated

  logical :: outputs_validated = .false.   ! full range check done once

  !--------------------------------------------------------------------------
  ! Shortwave disaggregation.
  !
  ! The emulator's FSDS is a mean over its 6 h step, and once SOLIN was
  ! corrected to the window mean it was fed (#35) that mean became a smeared
  ! band 90 degrees of longitude wide rather than anything resembling the
  ! instantaneous sun.  The surface models' albedo is instantaneous, and is set
  ! to 1 where the sun is down, so a flat 6-hourly FSDS delivers 13% of the
  ! ocean's shortwave onto cells the coupler considers night, where it is
  ! multiplied by (1 - 1) and discarded.  Measured against 0.3% before.
  !
  ! Net energy loss is small -- what is thrown away at dusk is over-delivered
  ! at dawn, and the ocean's heat drift bounds the whole effect at about
  ! 1 W/m2 -- but the diurnal cycle of ocean and ice heating is wrong, and
  ! nothing about it is defensible.
  !
  ! Scaling the bands by the instantaneous insolation over its window mean
  ! preserves the window mean exactly, because the mean of the instantaneous
  ! field over the window *is* the window mean by construction.  It is what
  ! datm does with an interval-mean shortwave (tintalgo = 'coszen').
  !--------------------------------------------------------------------------
  real(R8), allocatable :: solin_win(:,:)    ! window mean used for this step
  real(R8), allocatable :: solin_now(:,:)    ! instantaneous, this coupler step
  logical               :: solin_scale_ready = .false.

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
    integer(in) :: CurrentYMD  ! model date
    integer(in) :: CurrentTOD  ! model sec into model date
    integer(in) :: stepno      ! coupler step number, for the RNG seed offset
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

    call ace_cache_areas(ggrid)

    call seq_timemgr_EClockGetData( EClock, curr_ymd=CurrentYMD, curr_tod=CurrentTOD )
    call seq_timemgr_EClockGetData( EClock, stepno=stepno )

    !--- make a stochastic emulator reproducible ---
    if (eatm_rng_seed >= 0) then
      ! Offset by the step number rather than seeding with the same value every
      ! time.  A multi-segment run re-initializes at the start of each segment,
      ! and seeding identically there would replay the same noise realization in
      ! every segment -- for 1-year segments, a spurious annual periodicity in
      ! the emulator's stochasticity.  Adding stepno keeps the run reproducible
      ! (it is a function of the seed and the start date alone) while making the
      ! sequence continue rather than repeat.
      call eatm_torch_manual_seed(int(eatm_rng_seed, c_int64_t) + int(stepno, c_int64_t))
      write(logunit_atm,'(a,i0,a,i0)') &
           '(ace_comp_init) libtorch RNG seeded with eatm_rng_seed + stepno = ', &
           eatm_rng_seed, ' + ', stepno
    else
      write(logunit_atm,'(a)') &
           '(ace_comp_init) libtorch RNG left unseeded (eatm_rng_seed < 0);'// &
           ' a stochastic emulator will not reproduce between runs'
    end if

    if (read_restart) then

      ! int remainder (in sec) of coupler timestep relative to ACE timestep
      t_modulo = mod(CurrentTOD, eatm_model_dt)
      ! turn integer remainder into fraction through ACE timestep
      t_frac = real(t_modulo, kind=R8) / real(eatm_model_dt, kind=R8)

      call ace_bracket_blend(t_frac)
      call ace_capture_solin_window()   ! restored from the restart file

    else
      ! The initial condition is the emulator state at T0.  One inference
      ! carries it to T0 + dt, which is the *upper* bracket of the first
      ! interpolation interval -- SOLIN is computed for T0 + dt to match.
      call ace_compute_solin(EClock, ggrid)
      call ace_capture_solin_window()

      call ace_inference()

      ! Lower bracket: the state at T0 itself, taken from the initial
      ! condition wherever the channel exists there.  Seeding both brackets
      ! from the prediction instead (as this used to) throws away the one
      ! state whose valid time is known exactly and hands the coupler a T0+dt
      ! atmosphere at T0.  The flux and precipitation channels have no
      ! initial-condition counterpart, so they are held at the prediction for
      ! the first interval; there is nothing better available.
      do k = 1, n_output_channels
        if (out_from_in(k) > 0 .and. eatm_clock_align) then
          do j = 1, lsize_y
            do i = 1, lsize_x
              eatm_intrp%t_im1(k, i, j) = net_inputs(1, out_from_in(k), i, j)
            end do
          end do
        else
          do j = 1, lsize_y
            do i = 1, lsize_x
              eatm_intrp%t_im1(k, i, j) = net_outputs(1, k, i, j)
            end do
          end do
        end if
        do j = 1, lsize_y
          do i = 1, lsize_x
            eatm_intrp%t_ip1(k, i, j) = net_outputs(1, k, i, j)
          end do
        end do
      end do

      ! t_frac = 0 at T0: hand the coupler the initial condition for the
      ! snapshot channels, and the first predicted interval mean for the rest.
      call ace_bracket_blend(0.0_R8)
    endif

    ! The emulator has now been advanced to cover the interval that starts at
    ! the current model time; the driver's second initialization call arrives
    ! at this same time and must not advance it again.
    last_adv_ymd = CurrentYMD
    last_adv_tod = CurrentTOD

    ! using restart data from ACE set the fields passed to the coupler
    if (eatm_sw_diurnal) call ace_solin_now(EClock, ggrid)
    call ace_eatm_export(ggrid, verbose=.true.)

  end subroutine ace_comp_init

  !===============================================================================
  subroutine ace_cache_areas(ggrid)

    ! Cache the domain's cell areas so the surface-exchange budget can be
    ! reported as a global mean rather than a set of extrema.

    implicit none
    type(mct_gGrid), intent(in), pointer :: ggrid

    integer :: karea, klat, n, i, j

    if (allocated(cell_area)) return   ! init runs once, but do not rely on it

    allocate(cell_area(lsize_x, lsize_y))
    allocate(cell_lat(lsize_x, lsize_y))
    allocate(acc_lhf(lsize_x, lsize_y), acc_shf(lsize_x, lsize_y))
    allocate(acc_lwup(lsize_x, lsize_y), acc_swabs(lsize_x, lsize_y))
    allocate(acc_wsx(lsize_x, lsize_y), acc_wsy(lsize_x, lsize_y))
    allocate(acc_cov(lsize_x, lsize_y))
    allocate(solin_win(lsize_x, lsize_y), solin_now(lsize_x, lsize_y))
    solin_win(:,:) = 0.0_R8
    solin_now(:,:) = 0.0_R8
    call ace_reset_accumulators()

    karea = mct_aVect_indexRA(ggrid%data, 'area')
    klat  = mct_aVect_indexRA(ggrid%data, 'lat')

    n = 0
    area_total = 0.0_R8
    do j = 1, lsize_y
      do i = 1, lsize_x
        n = n + 1
        cell_area(i, j) = ggrid%data%rAttr(karea, n)
        cell_lat(i, j)  = ggrid%data%rAttr(klat, n)
        area_total = area_total + cell_area(i, j)
      end do
    end do

    if (area_total <= 0.0_R8) call shr_sys_abort( &
         '(ace_cache_areas) ERROR: domain cell areas sum to zero')

  end subroutine ace_cache_areas

  !===============================================================================
  subroutine ace_reset_accumulators()
    ! Start a fresh emulator interval.
    implicit none
    acc_lhf(:,:)   = 0.0_R8
    acc_shf(:,:)   = 0.0_R8
    acc_lwup(:,:)  = 0.0_R8
    acc_wsx(:,:)   = 0.0_R8
    acc_wsy(:,:)   = 0.0_R8
    acc_swabs(:,:) = 0.0_R8
    acc_cov(:,:)   = 0.0_R8
    acc_n          = 0
  end subroutine ace_reset_accumulators

  !===============================================================================
  subroutine ace_accumulate_coupler()

    !----------------------------------------------------------------
    ! Add this coupling step's applied surface exchange to the running sums.
    !
    ! Called once per coupler step, right after atm_import_mct has refreshed
    ! the coupler fields and before anything overwrites them.  The imported
    ! Faxx_* are the fluxes the coupler computed for the interval that just
    ! ended, using the atmospheric state EATM exported on the previous step --
    ! so pairing them with the albedos imported now, and with the shortwave
    ! bands still held from that previous export, reproduces what the surface
    ! models actually did.
    !
    ! The shortwave is the reason this is not simply an average of imported
    ! fields.  EATM exports four downwelling bands and the ocean and sea ice
    ! then absorb them using their *own* band-dependent albedos, so the net
    ! shortwave reaching the surface is not the emulator's FSDS - FSUS at all.
    ! Rebuilding it here from the bands and the merged albedos is the only way
    ! to see that part of the interface.  Since a merged Sx_ field is a
    ! fraction-weighted sum rather than a mean (it is zero where no surface
    ! model covers the cell), the absorbed flux is
    !
    !     sum over bands of  band * (covered_fraction - merged_albedo)
    !----------------------------------------------------------------
    implicit none

    integer  :: i, j
    real(R8) :: cov

    do j = 1, lsize_y
      do i = 1, lsize_x
        cov = min(max(ocnfrac(i, j) + icefrac(i, j) + lndfrac(i, j), 0.0_R8), 1.0_R8)

        acc_lhf(i, j)  = acc_lhf(i, j)  + lhf(i, j)
        acc_shf(i, j)  = acc_shf(i, j)  + shf(i, j)
        acc_lwup(i, j) = acc_lwup(i, j) + lwup(i, j)
        acc_wsx(i, j)  = acc_wsx(i, j)  + wsx(i, j)
        acc_wsy(i, j)  = acc_wsy(i, j)  + wsy(i, j)
        acc_cov(i, j)  = acc_cov(i, j)  + cov

        acc_swabs(i, j) = acc_swabs(i, j)                  &
             + swvdr(i, j) * (cov - asdir(i, j))           &
             + swndr(i, j) * (cov - aldir(i, j))           &
             + swvdf(i, j) * (cov - asdif(i, j))           &
             + swndf(i, j) * (cov - aldif(i, j))
      end do
    end do

    acc_n = acc_n + 1

  end subroutine ace_accumulate_coupler

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

    ! Add this coupling step's applied exchange to the interval that is closing.
    ! Done before the boundary test so the sample at the boundary itself belongs
    ! to the interval it ends, not the one it starts.
    call ace_accumulate_coupler()

    ! An emulator step is due at every multiple of eatm_model_dt, but only
    ! once per model time: the driver's phase-2 initialization call runs this
    ! routine again at the time the startup inference already covered.
    if (t_modulo == 0 .and. &
        .not. (eatm_clock_align .and. &
               CurrentYMD == last_adv_ymd .and. CurrentTOD == last_adv_tod)) then

      ! One line per emulator step, not per coupler step: at ATM_NCPL=48 the
      ! latter is 48 flushed writes a day, ~100 MB of atm.log over five years.
      write(logunit_atm, '(a,i9,a,i9.8,a,i6,a,i7,a)') &
           'eatm step ', stepno, ' date ', CurrentYMD, ' tod ', CurrentTOD, &
           ' (cpl dt ', cpl_idt, ' s) -- advancing emulator'
      call shr_sys_flush(logunit_atm)

      ! What the coupler did with the state handed over for the interval that
      ! just closed, next to what the emulator thought it was doing.  Both
      ! columns are now means over that same interval.
      call ace_flux_budget_report()
      call ace_reset_accumulators()

      ! Feed the emulator its own state *at this time*, which is the prediction
      ! made one emulator step ago (t_ip1).  net_outputs currently still holds
      ! the field the coupler was handed at the end of the previous coupler
      ! step, which is a partial interpolation towards t_ip1 and is therefore
      ! not a state the emulator was ever trained to consume.
      if (eatm_autoregress_state) then
        call ace_eatm_import(eatm_intrp%t_ip1)
      else
        ! Ablation only (#8): the pre-branch code fed the emulator the field
        ! last handed to the coupler, which is a partial interpolation towards
        ! t_ip1 rather than any state the emulator was trained to consume.
        call ace_eatm_import(net_outputs(1, :, :, :))
      end if
      call ace_compute_solin(EClock, ggrid)
      call ace_capture_solin_window()

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

      last_adv_ymd = CurrentYMD
      last_adv_tod = CurrentTOD

    end if

    t_frac = real(t_modulo, kind=r8) / real(eatm_model_dt, kind=r8)

    call ace_bracket_blend(t_frac)

    ! diurnal shape for this coupler step, applied to the shortwave on export
    if (eatm_sw_diurnal) call ace_solin_now(EClock, ggrid)

    call ace_eatm_export(ggrid, verbose=(t_modulo == 0))

  end subroutine ace_comp_run

  !===============================================================================
  subroutine ace_bracket_blend(t_frac)

    !----------------------------------------------------------------
    ! Combine the two bracketing emulator states into the field the coupler is
    ! handed at a fraction t_frac through the current emulator interval.
    !
    ! The two kinds of channel are combined differently, because they mean
    ! different things (see eatm_channel_is_interval_mean).
    !
    !   snapshot channels -- PS, TS, the layer state, the near-surface
    !     diagnostics -- are instantaneous values at the two bracket times, so
    !     they are linearly interpolated.
    !
    !   interval-mean channels -- every radiative flux, the turbulent fluxes,
    !     the stresses, precipitation -- are already the mean over the interval
    !     being stepped across.  t_ip1 *is* the answer everywhere inside it.
    !     Interpolating them from the previous interval's mean, as this used
    !     to, hands the coupler a value that only reaches the correct one at
    !     the very end of the window: averaged over the interval the applied
    !     flux is (mean_previous + mean_current)/2, a half-step lag.  At a 6 h
    !     emulator step that is a 3 h lag on the surface radiation and
    !     precipitation, i.e. 45 degrees of diurnal phase, smeared by an
    !     equivalent smoothing.
    !----------------------------------------------------------------
    implicit none

    real(R8), intent(in) :: t_frac

    integer  :: i, j, k
    real(R4) :: f

    f = real(t_frac, R4)

    do k = 1, n_output_channels
      if (out_is_mean(k) .and. eatm_flux_interval_mean) then
        do j = 1, lsize_y
          do i = 1, lsize_x
            net_outputs(1, k, i, j) = eatm_intrp%t_ip1(k, i, j)
          end do
        end do
      else
        do j = 1, lsize_y
          do i = 1, lsize_x
            net_outputs(1, k, i, j) = eatm_intrp%t_im1(k, i, j) + &
                f * (eatm_intrp%t_ip1(k, i, j) - eatm_intrp%t_im1(k, i, j))
          end do
        end do
      end if
    end do

  end subroutine ace_bracket_blend

  subroutine ace_comp_finalize()
    call torch_delete(ace_model)
    if (allocated(cell_area)) deallocate(cell_area)
    if (allocated(cell_lat))  deallocate(cell_lat)
    if (allocated(acc_lhf))   deallocate(acc_lhf, acc_shf, acc_lwup, acc_wsx, acc_wsy, acc_swabs, acc_cov)
    if (allocated(solin_win)) deallocate(solin_win, solin_now)
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

    call ace_validate_outputs()

  end subroutine ace_inference

  !===============================================================================
  subroutine ace_validate_outputs()

    !----------------------------------------------------------------
    ! Check what came back out of the traced graph before anything downstream
    ! consumes it.
    !
    ! Two distinct failures are caught here.
    !
    ! Non-finite output is always fatal.  The emulator is autoregressive and
    ! the spherical harmonic transform inside an SFNO is global, so a single
    ! NaN in one channel becomes every channel on the next step and stays that
    ! way for the rest of the run.  Nothing downstream detects it: the export
    ! clamps compare against zero, and `NaN < 0` is false, so a NaN passes
    ! through max() untouched and reaches the ocean as a NaN forcing.
    !
    ! On the first inference the named channels are also range-checked.  A
    ! traced model is an opaque graph; nothing at load time ties its channel
    ! order to the table in eatm_channels_mod.  If eatm_emulator names the
    ! wrong table, or a checkpoint is re-traced with a different layout, every
    ! index silently reads the wrong field -- a surface pressure of 0.3 would
    ! be exported as Sa_pslv and the run would continue.  Checking once is
    ! enough to establish the contract, and it costs one pass over the block.
    !----------------------------------------------------------------
    implicit none

    integer  :: i, j, k, nbad, nout
    real(R8) :: lo, hi, vmin, vmax, v
    logical  :: checked
    character(len=CL) :: msg

    !--- non-finite: every step, fatal ---
    nbad = 0
    do k = 1, n_output_channels
      do j = 1, lsize_y
        do i = 1, lsize_x
          if (net_outputs(1, k, i, j) /= net_outputs(1, k, i, j) .or. &
              abs(net_outputs(1, k, i, j)) > huge(0.0_R4) * 0.5_R4) then
            if (nbad == 0) write(logunit_atm,'(a)') &
                 '(ace_validate_outputs) ERROR: first non-finite output in channel '// &
                 trim(out_names(k))
            nbad = nbad + 1
          end if
        end do
      end do
    end do
    if (nbad > 0) then
      write(logunit_atm,'(a,i0,a,i0,a)') &
           '(ace_validate_outputs) ERROR: ', nbad, ' non-finite values in ', &
           n_output_channels * lsize_x * lsize_y, ' emulator outputs'
      call shr_sys_flush(logunit_atm)
      call shr_sys_abort('(ace_validate_outputs) ERROR: emulator returned '// &
           'non-finite output; the state is unrecoverable, see the atm log')
    end if

    if (outputs_validated) return

    !--- physical ranges: first inference only, fatal ---
    nout = 0
    do k = 1, n_output_channels
      call eatm_channel_range(out_names(k), lo, hi, checked)
      if (.not. checked) cycle
      vmin =  huge(1.0_R8)
      vmax = -huge(1.0_R8)
      do j = 1, lsize_y
        do i = 1, lsize_x
          v = real(net_outputs(1, k, i, j), R8)
          vmin = min(vmin, v)
          vmax = max(vmax, v)
        end do
      end do
      if (vmin < lo .or. vmax > hi) then
        write(logunit_atm,'(a,2es13.5,a,2es13.5,a)') &
             '(ace_validate_outputs) ERROR: channel '//trim(out_names(k))// &
             ' spans ', vmin, vmax, ', outside the admissible ', lo, hi, &
             ' -- the traced graph does not match the compiled channel table'
        nout = nout + 1
        if (nout == 1) write(msg,'(a)') &
             '(ace_validate_outputs) ERROR: emulator output channel '// &
             trim(out_names(k))//' is out of physical range.  Check that '// &
             'eatm_emulator matches the checkpoint eatm_model_file was traced '// &
             'from (compare against its *_metadata.yaml).'
      end if
    end do
    call shr_sys_flush(logunit_atm)
    if (nout > 0) call shr_sys_abort(trim(msg))

    write(logunit_atm,'(a,i0,a)') &
         '(ace_validate_outputs) channel contract verified: ', &
         n_output_channels, ' output channels within physical range'
    call shr_sys_flush(logunit_atm)

    outputs_validated = .true.

  end subroutine ace_validate_outputs

  !===============================================================================
  subroutine ace_flux_budget_report()

    !----------------------------------------------------------------
    ! Report the surface exchange the emulator predicted next to the exchange
    ! the coupler actually applied, as area-weighted global means.
    !
    ! These are two different quantities and there is no mechanism in the MCT
    ! driver for the first to override the second.  The emulator predicts
    ! LHFLX and SHFLX (and, for SamudrACE, TAUX/TAUY) and evolves its own
    ! atmosphere consistently with them.  The coupler ignores those channels
    ! and rebuilds the turbulent fluxes from the exported state and the SST
    ! with shr_flux_atmOcn's bulk formula; that is what the ocean and sea ice
    ! integrate.  Whatever the two disagree by is energy and momentum that
    ! enters the ocean without leaving the atmosphere, or the reverse.
    !
    ! It is the single largest known error term in a coupled EATM run, it is
    ! not removable from inside the atmosphere component, and until it is it
    ! needs to be *measured* -- in the run, at run time, rather than
    ! reconstructed afterwards from history files that do not carry the
    ! emulator's own flux channels at all.
    !
    ! Two conventions have to be reconciled before the columns mean anything.
    !
    ! Sign: everything is reported as "positive = surface loses energy to the
    ! atmosphere", matching the emulator's LHFLX/SHFLX.  The imported coupler
    ! fields already carry EAM's convention (atm_comp_mct.F90:456 negates
    ! Faxx_sen and Faxx_taux, matching eam/src/cpl/atm_comp_mct.F90:1801), and
    ! EAM's own TAUX/FLUS history fields are those same imported values, so the
    ! emulator's TAUX and FLUS channels compare to wsx and lwup directly.
    !
    ! Area basis: this is the subtle one.  The coupler's Faxx_* are *merged*
    ! fluxes, `sum over surface types of frac_s * F_s`.  With a stub land model
    ! lfrac is zero, so over the ~34% of the globe no surface model covers, the
    ! merged flux is not "small", it is structurally absent.  Taking a plain
    ! area mean of it and comparing against the emulator's full-cell flux
    ! understates the coupler by that fraction and makes a large disagreement
    ! look like a small one.  Both columns are therefore reported per unit
    ! *covered* area: the coupler's merged flux divided by the mean covered
    ! fraction, and the emulator's flux weighted by that same fraction.
    !----------------------------------------------------------------
    implicit none

    integer  :: i, j
    real(R8) :: emu_lh, emu_sh, cpl_lh, cpl_sh
    real(R8) :: emu_tx, emu_ty, cpl_tx, cpl_ty
    real(R8) :: emu_net, cpl_net
    real(R8) :: fsds_m, flds_m, fsus_m, flus_m, lwup_m
    real(R8) :: emu_swnet, cpl_swnet
    real(R8) :: w_cov(lsize_x, lsize_y)   ! area * covered fraction
    real(R8) :: cov_total                 ! its sum

    if (ix_out_lhflx <= 0 .or. ix_out_shflx <= 0) return
    if (acc_n <= 0) return   ! nothing accumulated yet

    ! Covered fraction on the same footing as the accumulated fluxes: the mean
    ! over the interval, not its value at the closing instant.
    cov_total = 0.0_R8
    do j = 1, lsize_y
      do i = 1, lsize_x
        w_cov(i, j) = cell_area(i, j) * acc_cov(i, j) / real(acc_n, R8)
        cov_total = cov_total + w_cov(i, j)
      end do
    end do

    if (cov_total <= 0.0_R8) then
      write(logunit_atm,'(a)') &
           '  sfc exchange: no surface model covers any cell, budget not reported'
      return
    end if

    emu_lh = emu_mean(ix_out_lhflx)
    emu_sh = emu_mean(ix_out_shflx)

    cpl_lh = -cpl_mean(acc_lhf)   ! Faxx_lat is not negated on import
    cpl_sh =  cpl_mean(acc_shf)   ! Faxx_sen is

    write(logunit_atm,'(a,f6.4,a,i0,a)') &
         '  sfc exchange over the covered fraction (', cov_total / area_total, &
         ' of area), W/m2, +ve = surface -> atmosphere, ', acc_n, ' cpl steps'
    write(logunit_atm,'(a)') &
         '                                                  emulator     coupler        diff'
    write(logunit_atm,'(a,3f12.4)') '    latent                                    ', &
         emu_lh, cpl_lh, cpl_lh - emu_lh
    write(logunit_atm,'(a,3f12.4)') '    sensible                                  ', &
         emu_sh, cpl_sh, cpl_sh - emu_sh
    write(logunit_atm,'(a,3f12.4)') '    turbulent total                           ', &
         emu_lh + emu_sh, cpl_lh + cpl_sh, (cpl_lh + cpl_sh) - (emu_lh + emu_sh)

    !--- radiative terms, for the full surface energy budget ---
    fsds_m = emu_mean(ix_out_fsds)
    flds_m = emu_mean(ix_out_flds)
    fsus_m = 0.0_R8 ; if (ix_out_fsus > 0) fsus_m = emu_mean(ix_out_fsus)
    flus_m = 0.0_R8 ; if (ix_out_flus > 0) flus_m = emu_mean(ix_out_flus)
    lwup_m = cpl_mean(acc_lwup)

    ! Shortwave: the emulator's own net against what the surface really absorbs.
    ! These are different numbers.  EATM hands the coupler four downwelling
    ! bands split by fixed fractions, and the ocean and sea ice absorb them with
    ! their own band-dependent albedos; the emulator's FSDS - FSUS never reaches
    ! anything.  Reporting only the emulator's version, as this used to, hides
    ! the shortwave part of the interface mismatch entirely -- which matters
    ! most over snow and sea ice, where the spectral albedo contrast is largest.
    emu_swnet = fsds_m - fsus_m
    cpl_swnet = cpl_mean(acc_swabs)
    write(logunit_atm,'(a,3f12.4)') '    net shortwave absorbed                    ', &
         emu_swnet, cpl_swnet, cpl_swnet - emu_swnet

    if (ix_out_flus > 0) then
      ! Net downward at the surface.  Only the downwelling longwave is common to
      ! both columns; the shortwave now differs too, so this is the full
      ! interface disagreement rather than just its turbulent part.
      emu_net = emu_swnet + (flds_m - flus_m) - (emu_lh + emu_sh)
      cpl_net = cpl_swnet + (flds_m - lwup_m) - (cpl_lh + cpl_sh)
      write(logunit_atm,'(a,3f12.4)') '    net surface (downward)                    ', &
           emu_net, cpl_net, cpl_net - emu_net
      write(logunit_atm,'(a,3f12.4)') '    surface LW up                             ', &
           flus_m, lwup_m, lwup_m - flus_m
    end if

    if (ix_out_taux > 0 .and. ix_out_tauy > 0) then
      emu_tx = emu_mean(ix_out_taux)
      emu_ty = emu_mean(ix_out_tauy)
      cpl_tx = cpl_mean(acc_wsx)
      cpl_ty = cpl_mean(acc_wsy)
      write(logunit_atm,'(a,3es12.4)') '    stress x (N/m2)                           ', &
           emu_tx, cpl_tx, cpl_tx - emu_tx
      write(logunit_atm,'(a,3es12.4)') '    stress y (N/m2)                           ', &
           emu_ty, cpl_ty, cpl_ty - emu_ty
    end if

    !--- where the turbulent mismatch lives.
    !
    ! A global mean of zero can be two large regional errors of opposite sign.
    ! That matters directly for eatm_ref_height: #62 found the *total* mismatch
    ! nulling near 44 m while latent and sensible null at different heights, so
    ! the global zero is already known to be a cancellation between components.
    ! This asks whether it is also one across latitude.
    call band_report()

    !--- top of atmosphere, when the emulator predicts it.  Global, not
    !--- covered-area: the TOA budget is not a surface-type quantity.
    if (ix_out_flut > 0 .and. ix_out_fsutoa > 0) then
      write(logunit_atm,'(a,f12.4)') '    TOA net, global (SOLIN-FSUTOA-FLUT)       ', &
           glob_in(ix_in_solin) - glob_out(ix_out_fsutoa) - glob_out(ix_out_flut)
    end if

    call shr_sys_flush(logunit_atm)

  contains

    ! emulator channel, weighted by covered fraction
    real(R8) function emu_mean(k)
      integer, intent(in) :: k
      integer :: i, j
      emu_mean = 0.0_R8
      if (k <= 0) return
      do j = 1, lsize_y
        do i = 1, lsize_x
          emu_mean = emu_mean + w_cov(i, j) * real(net_outputs(1, k, i, j), R8)
        end do
      end do
      emu_mean = emu_mean / cov_total
    end function emu_mean

    ! Accumulated merged coupler field: a plain area integral over the covered
    ! area, since the fraction weighting is already inside the field itself.
    ! The sum is over acc_n coupling steps, so divide that out to get the
    ! interval mean -- the same kind of quantity as the emulator's channel.
    subroutine band_report()
      ! Latent, sensible and their total, emulator against coupler, in six
      ! latitude bands.  Weighted exactly as the global figures are: area times
      ! the mean covered fraction over the accumulation window.
      integer, parameter :: nb = 6
      real(R8), parameter :: edge(nb+1) = &
           (/ -90.0_R8, -60.0_R8, -30.0_R8, 0.0_R8, 30.0_R8, 60.0_R8, 90.0_R8 /)
      character(len=8), parameter :: nm(nb) = &
           (/ '  60-90S', '  30-60S', '   0-30S', '   0-30N', '  30-60N', '  60-90N' /)
      real(R8) :: wsum(nb), elh(nb), esh(nb), clh(nb), csh(nb)
      real(R8) :: wq, la
      integer  :: i, j, b

      wsum = 0.0_R8; elh = 0.0_R8; esh = 0.0_R8; clh = 0.0_R8; csh = 0.0_R8

      do j = 1, lsize_y
        do i = 1, lsize_x
          la = cell_lat(i, j)
          b  = 1
          do while (b < nb .and. la >= edge(b+1))
            b = b + 1
          end do
          wq = w_cov(i, j)
          wsum(b) = wsum(b) + wq
          if (ix_out_lhflx > 0) elh(b) = elh(b) + wq * real(net_outputs(1, ix_out_lhflx, i, j), R8)
          if (ix_out_shflx > 0) esh(b) = esh(b) + wq * real(net_outputs(1, ix_out_shflx, i, j), R8)
          clh(b) = clh(b) - cell_area(i, j) * acc_lhf(i, j) / real(acc_n, R8)
          csh(b) = csh(b) + cell_area(i, j) * acc_shf(i, j) / real(acc_n, R8)
        end do
      end do

      write(logunit_atm,'(a)') '    by latitude (emulator, coupler, mismatch), W/m2:'
      write(logunit_atm,'(a)') &
           '        band     area%      latent-emu  latent-cpl  latent-mis' // &
           '   sens-emu   sens-cpl    sens-mis    TOTAL-mis'
      do b = 1, nb
        if (wsum(b) <= 0.0_R8) cycle
        write(logunit_atm,'(a,f9.1,7f12.3)') nm(b), 100.0_R8 * wsum(b) / cov_total, &
             elh(b)/wsum(b), clh(b)/wsum(b), clh(b)/wsum(b) - elh(b)/wsum(b), &
             esh(b)/wsum(b), csh(b)/wsum(b), csh(b)/wsum(b) - esh(b)/wsum(b), &
             (clh(b)+csh(b))/wsum(b) - (elh(b)+esh(b))/wsum(b)
      end do

    end subroutine band_report

    real(R8) function cpl_mean(f)
      real(R8), intent(in) :: f(:,:)
      integer :: i, j
      cpl_mean = 0.0_R8
      do j = 1, lsize_y
        do i = 1, lsize_x
          cpl_mean = cpl_mean + cell_area(i, j) * f(i, j)
        end do
      end do
      cpl_mean = cpl_mean / (cov_total * real(acc_n, R8))
    end function cpl_mean

    real(R8) function glob_out(k)
      integer, intent(in) :: k
      integer :: i, j
      glob_out = 0.0_R8
      if (k <= 0) return
      do j = 1, lsize_y
        do i = 1, lsize_x
          glob_out = glob_out + cell_area(i, j) * real(net_outputs(1, k, i, j), R8)
        end do
      end do
      glob_out = glob_out / area_total
    end function glob_out

    real(R8) function glob_in(k)
      integer, intent(in) :: k
      integer :: i, j
      glob_in = 0.0_R8
      if (k <= 0) return
      do j = 1, lsize_y
        do i = 1, lsize_x
          glob_in = glob_in + cell_area(i, j) * real(net_inputs(1, k, i, j), R8)
        end do
      end do
      glob_in = glob_in / area_total
    end function glob_in

  end subroutine ace_flux_budget_report

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
    real(R8) :: sif       ! emulator sea ice fraction, of the sea surface
    real(R8) :: land_true ! the land fraction EATM actually reports
    real(R8) :: ocn_true  ! open-ocean fraction, the reference prescriber's OCNFRAC
    logical  :: use_emul_ice
    real(R8) :: ifrac_flat(lsize_x*lsize_y)
    real(R8) :: sst_flat(lsize_x*lsize_y)
    logical  :: use_ocn_ts
    integer  :: n
    real(R8) :: fo, fi, fl  ! individually bounded ocean / ice / land fractions
    integer  :: n_clip_frac, n_norm_frac   ! cells corrected, by kind
    real(R8) :: worst_clip, worst_norm     ! and by how much

    ! A merged surface temperature the coupler built from the *original*
    ! fractions is about to be paired with corrected ones.  If the correction is
    ! material the two describe different mixtures, so report it rather than let
    ! it pass silently.  The threshold is not round-off: the coupler's own
    ! fraction limiter works to about 1e-3 (see #15b, where 77 of 64800 cells
    ! differed by roughly eps_fraclim), so anything at that level is expected
    ! and only a gross violation means something upstream is wrong.
    real(R8), parameter :: frac_tol = 0.05_R8

    do k = 1, n_input_channels
      if (in_from_out(k) > 0) then
        do j = 1, lsize_y
          do i = 1, lsize_x
            net_inputs(1, k, i, j) = state(in_from_out(k), i, j)
          enddo
        enddo
      end if
    end do

    n_clip_frac = 0
    n_norm_frac = 0
    worst_clip  = 0.0_R8
    worst_norm  = 0.0_R8

    ! SamudrACE's ocean predicts the sea ice fraction and its coupler splits
    ! the non-land fraction with it:
    !     ICEFRAC = ocean_sea_ice_fraction * (1 - LANDFRAC)
    !     OCNFRAC = max(1 - LANDFRAC - ICEFRAC, 0)
    ! E3SM writes the same identity as lfrac + ifrac + ofrac = 1 but fills
    ! ifrac from a sea ice component, so with a stub ice the atmosphere is told
    ! the polar ocean is open water.  Diagnostic only; see shr_emul_ice_mod.
    use_emul_ice = eatm_icefrac_from_ocn .and. &
                   shr_emul_ice_avail(lsize_x*lsize_y)
    if (use_emul_ice) then
      call shr_emul_ice_get(ifrac_flat)
    else if (eatm_icefrac_from_ocn) then
      write(logunit_atm,'(a)') '(ace_eatm_import) WARNING: '// &
           'eatm_icefrac_from_ocn is set but the ocean emulator has published '// &
           'no sea ice fraction of the right size; leaving ICEFRAC as the '// &
           'coupler gave it'
    end if
    ! SamudrACE's atmosphere declares a prescribed ocean with interpolate:true,
    ! so its reference coupler builds
    !     TS = OCNFRAC*sst + (1 - OCNFRAC)*TS_atm,  OCNFRAC = (1-LANDFRAC)(1-sif)
    ! and the atmosphere keeps its own predicted surface temperature over land
    ! *and over sea ice*.  The coupler's merged Sx_t is a different quantity: it
    ! fills the ice share with the ice component's own surface temperature,
    ! which for EICE is a fabricated seasonal cycle.  Where the ocean has
    ! published its unmerged sst, use the reference's formula instead.
    use_ocn_ts = eatm_ts_from_ocn .and. eatm_land_deficit .and. &
                 shr_emul_ice_avail(lsize_x*lsize_y) .and. &
                 shr_emul_ice_sst_avail(lsize_x*lsize_y)
    if (use_ocn_ts) then
      call shr_emul_ice_get(ifrac_flat)
      call shr_emul_ice_get_sst(sst_flat)
    else if (eatm_ts_from_ocn) then
      write(logunit_atm,'(a)') '(ace_eatm_import) NOTE: eatm_ts_from_ocn is set '// &
           'but the ocean emulator has published no sea surface temperature of '// &
           'the right size; using the coupler''s merged Sx_t'
    end if

    n = 0

    do j = 1, lsize_y
      do i = 1, lsize_x

        ! Bound each fraction on its own before combining them.  Clipping only
        ! the sum lets an individual field arrive outside [0,1] -- a negative
        ! ICEFRAC compensated by an ocean fraction above one still sums to a
        ! plausible total, and the emulator was never shown such a state.  The
        ! sum is then held at or below one so the deficit stays non-negative.
        n  = n + 1
        fo = min(max(ocnfrac(i, j), 0.0_R8), 1.0_R8)
        fi = min(max(icefrac(i, j), 0.0_R8), 1.0_R8)
        fl = min(max(lndfrac(i, j), 0.0_R8), 1.0_R8)

        worst_clip = max(worst_clip, &
             max(abs(fo - ocnfrac(i, j)), &
                 max(abs(fi - icefrac(i, j)), abs(fl - lndfrac(i, j)))))
        if (fo /= ocnfrac(i, j) .or. fi /= icefrac(i, j) .or. fl /= lndfrac(i, j)) &
             n_clip_frac = n_clip_frac + 1

        covered = fo + fi + fl
        if (covered > 1.0_R8) then
          n_norm_frac = n_norm_frac + 1
          worst_norm  = max(worst_norm, covered - 1.0_R8)
          fo = fo / covered
          fi = fi / covered
          fl = fl / covered
          covered = 1.0_R8
        end if
        deficit = 1.0_R8 - covered

        if (eatm_land_deficit) then

          net_inputs(1, ix_in_landfrac, i, j) = real(fl + deficit, R4)
          net_inputs(1, ix_in_ocnfrac,  i, j) = real(fo, R4)
          net_inputs(1, ix_in_icefrac,  i, j) = real(fi, R4)

          ! Re-split the non-land part between ice and open ocean the way
          ! SamudrACE does.  It has to key off `fl + deficit`, not `fl`: with a
          ! stub land Sf_lfrac is zero everywhere and the real land fraction
          ! arrives as the deficit.  Keying off fl alone drives the deficit to
          ! zero, which hands the emulator LANDFRAC = 0 and a 0 K surface
          ! temperature over every continent.  LANDFRAC and TS are left exactly
          ! as computed above; only the split of what is left over changes.
          if (use_emul_ice) then
            land_true = min(max(fl + deficit, 0.0_R8), 1.0_R8)
            sif = min(max(ifrac_flat(n), 0.0_R8), 1.0_R8)
            net_inputs(1, ix_in_icefrac, i, j) = &
                 real(sif * (1.0_R8 - land_true), R4)
            net_inputs(1, ix_in_ocnfrac, i, j) = real(max( &
                 1.0_R8 - land_true - sif * (1.0_R8 - land_true), 0.0_R8), R4)
          end if

          if (ix_in_ts > 0) then
            if (use_ocn_ts) then
              land_true = min(max(fl + deficit, 0.0_R8), 1.0_R8)
              sif       = min(max(ifrac_flat(n), 0.0_R8), 1.0_R8)
              ocn_true  = max((1.0_R8 - land_true) * (1.0_R8 - sif), 0.0_R8)
              net_inputs(1, ix_in_ts, i, j) = real( &
                   ocn_true * sst_flat(n) + &
                   (1.0_R8 - ocn_true) * real(state(ix_out_ts, i, j), R8), R4)
            else
              net_inputs(1, ix_in_ts, i, j) = real( &
                   ts(i, j) + deficit * real(state(ix_out_ts, i, j), R8), R4)
            end if
          end if

        else

          ! Ablation only (#15b, #25): the coupler's fractions and merged
          ! surface temperature straight through, weighted exactly as the code
          ! did at 23dd0c1b97.  With a stub land model that is LANDFRAC = 0
          ! everywhere and TS = 0 K over every land point.
          net_inputs(1, ix_in_landfrac, i, j) = real(lndfrac(i, j), R4)
          net_inputs(1, ix_in_ocnfrac,  i, j) = real(ocnfrac(i, j), R4)
          net_inputs(1, ix_in_icefrac,  i, j) = real(icefrac(i, j), R4)

          if (ix_in_ts > 0) then
            net_inputs(1, ix_in_ts, i, j) = real( &
                 (1.0_R8 - lndfrac(i, j)) * ts(i, j) + &
                 lndfrac(i, j) * real(state(ix_out_ts, i, j), R8), R4)
          end if

        end if

      enddo
    enddo

    if (n_clip_frac > 0 .or. n_norm_frac > 0) then
      write(logunit_atm,'(a,i0,a,es9.2,a,i0,a,es9.2)') &
           '  frac fix  clipped=', n_clip_frac, ' (max ', worst_clip, &
           ')  renormalized=', n_norm_frac, ' (max excess ', worst_norm
      ! Anything beyond round-off means the merged Sx_t handed over alongside
      ! these fractions was built from a different mixture than the emulator is
      ! now being told about.
      if (max(worst_clip, worst_norm) > frac_tol) call shr_sys_abort( &
           '(ace_eatm_import) ERROR: coupler surface fractions are outside '// &
           '[0,1] or sum above one by more than round-off; the merged Sx_t '// &
           'no longer matches the fractions passed to the emulator')
    end if

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
    real(R8) :: raw, qsat
    logical  :: do_log
    logical  :: use_near_surface   ! export at eatm_ref_height, not the layer midpoint
    integer  :: n_cap_shum         ! cells where humidity was capped at saturation
    real(R8) :: rh_max             ! largest relative humidity seen before capping

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
    n_cap_shum    = 0
    rh_max        = 0.0_R8

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

        !--- Cap the exported humidity at saturation.
        !---
        !--- Sa_shum is consumed by shr_flux_atmOcn as the *vapour* mixing
        !--- ratio: the latent flux is proportional to (q_sat(SST) - Sa_shum),
        !--- so a supersaturated value drives evaporation towards zero or
        !--- reverses it into condensation onto the ocean.
        !---
        !--- In the lowest_level configuration the field is the emulator's
        !--- specific *total* water, condensate included, and there is no
        !--- channel that separates the two: 19.3% of ocean cells arrive
        !--- supersaturated, with relative humidities up to 2.82.  Clipping
        !--- the condensate off at saturation is the closest estimate of the
        !--- vapour part the emulator's own output supports.
        !---
        !--- In the near_surface configuration Qat2m is a genuine vapour
        !--- humidity and the cap is a guard rather than a correction; it
        !--- fires on a few hundred cells out of 64800.
        if (.not. eatm_legacy_surface .and. eatm_cap_shum) then
          esat = datm_shr_esat(tbot(i, j), tbot(i, j))
          qsat = (0.622_R8 * esat) / max(pbot(i, j) - 0.378_R8 * esat, 1.0_R8)
          if (shum(i, j) > qsat) then
            n_cap_shum = n_cap_shum + 1
            if (qsat > 0.0_R8) rh_max = max(rh_max, shum(i, j) / qsat)
            shum(i, j) = qsat
          end if
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

        !--- put the interval-mean shortwave back on the real diurnal cycle ---
        if (eatm_sw_diurnal .and. solin_scale_ready) then
          if (solin_win(i, j) > 1.0_R8) then
            fsds_dn = fsds_dn * solin_now(i, j) / solin_win(i, j)
          else
            fsds_dn = 0.0_R8   ! polar night: no sun in the window, none now
          end if
        end if

        swvdr(i, j) = fsds_dn * frac_swvdr
        swndr(i, j) = fsds_dn * frac_swndr
        swvdf(i, j) = fsds_dn * frac_swvdf
        swndf(i, j) = fsds_dn * frac_swndf

        if (ix_out_fsus > 0 .and. .not. eatm_legacy_surface) then
          ! FSUS is a window mean too, so scale it the same way to keep the
          ! reported net consistent with the bands actually exported
          raw = real(net_outputs(1, ix_out_fsus, i, j), R8)
          if (eatm_sw_diurnal .and. solin_scale_ready) then
            if (solin_win(i, j) > 1.0_R8) then
              raw = raw * solin_now(i, j) / solin_win(i, j)
            else
              raw = 0.0_R8
            end if
          end if
          raw = fsds_dn - max(raw, 0.0_R8)
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
    if (n_cap_shum > 0) then
      write(logunit_atm, '(a,i0,a,i0,a,f7.3)') '  capped   shum=', n_cap_shum, &
           ' of ', lsize_x*lsize_y, ' cells at saturation, max RH before cap ', rh_max
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
  subroutine ace_solin_now(EClock, ggrid)

    ! Instantaneous TOA insolation at the current coupler time, and (on an
    ! emulator boundary) a copy of the window mean the emulator was handed.
    ! Their ratio is the diurnal shape that ace_eatm_export puts back onto the
    ! interval-mean shortwave.

    implicit none
    type(ESMF_Clock), intent(in) :: EClock
    type(mct_gGrid),  intent(in), pointer :: ggrid

    integer(IN)       :: CurrentYMD, CurrentTOD
    character(len=CS) :: calendar
    real(R8)          :: julday, delta, eccf, lat_r, lon_r
    real(R8), parameter :: degtorad = SHR_CONST_PI / 180.0_R8
    integer           :: klat, klon, n, i, j
    real(R8), pointer :: yc(:), xc(:)

    call seq_timemgr_EClockGetData(EClock, curr_ymd=CurrentYMD, curr_tod=CurrentTOD)
    call seq_timemgr_EClockGetData(EClock, calendar=calendar)
    call shr_cal_date2julian(CurrentYMD, CurrentTOD, julday, calendar)

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
        solin_now(i, j) = solar_const * eccf * &
             max(0.0_R8, shr_orb_cosz(julday, lat_r, lon_r, delta))
      end do
    end do

    deallocate(yc, xc)

  end subroutine ace_solin_now

  !===============================================================================
  subroutine ace_capture_solin_window()
    ! Keep the window mean the emulator is about to be driven with, as the
    ! denominator of the diurnal rescaling for the interval it covers.
    implicit none
    integer :: i, j
    do j = 1, lsize_y
      do i = 1, lsize_x
        solin_win(i, j) = real(net_inputs(1, ix_in_solin, i, j), R8)
      end do
    end do
    solin_scale_ready = .true.
  end subroutine ace_capture_solin_window

  !===============================================================================
  subroutine ace_compute_solin(EClock, ggrid)
    !----------------------------------------------------------------
    ! Compute SOLIN (solar insolation at TOA) from orbital mechanics, as the
    ! *time mean over the emulator step about to be taken*:
    !
    !   SOLIN = (1/dt) * integral over (T, T+dt] of S0 * eccf * max(0, cosz)
    !
    ! Two separate properties of the training data fix this form.
    !
    ! First, the emulator declares SOLIN in next_step_forcing_names, and fme
    ! feeds such a channel from time index step+1 rather than step
    ! (fme/ace/stepper/single_module.py:1139-1145).  The value belongs to the
    ! prediction target, not to the input state's own time.
    !
    ! Second -- and this is what the window is for -- SOLIN in the E3SMv3
    ! training stream carries cell_methods = "time: mean", not "time: point".
    ! It is the mean insolation over the 6 h leading up to its timestamp, which
    ! is exactly the interval (T, T+dt] the model is stepping across.
    !
    ! Using the instantaneous value at T+dt instead is not a small error.  The
    ! two fields have the same global mean (342 W/m2 either way -- the lit
    ! hemisphere is always the same fraction of the globe), so a global budget
    ! cannot see it, but they are different fields point by point: the
    ! instantaneous field is a cosine bullseye at the subsolar point, the
    ! 6-hourly mean is a smeared band 90 degrees of longitude wide.  Their RMS
    ! difference is 330 W/m2 against a field whose own global mean is 342.
    ! Every step was handing the emulator a radiative forcing pattern unlike
    ! anything in its training set.
    !
    ! The window mean is evaluated by the midpoint rule on n_solin_sub
    ! sub-intervals.  The integrand is smooth apart from the kink at sunrise
    ! and sunset, so convergence is fast: against a 2400-point reference, 48
    ! sub-steps (7.5 min) leave an RMS error of 0.03 W/m2 and a maximum of
    ! 0.1 W/m2.  The cost is 48 cosz evaluations per cell per emulator step,
    ! which is nothing next to one SFNO forward pass.
    !----------------------------------------------------------------
    implicit none
    type(ESMF_Clock), intent(in) :: EClock
    type(mct_gGrid),  intent(in), pointer :: ggrid

    integer(IN)       :: CurrentYMD, CurrentTOD
    character(len=CS) :: calendar
    real(R8)          :: julday, jsub
    real(R8)          :: delta, eccf
    real(R8)          :: lat_r, lon_r
    real(R8)          :: cosz_val, dt_days
    real(R8), parameter :: degtorad = SHR_CONST_PI / 180.0_R8

    ! Sub-intervals used for the midpoint-rule window mean.  Reduced to a
    ! single endpoint evaluation when eatm_solin_window is off, which
    ! reproduces the instantaneous SOLIN(T+dt) the code used before #35.
    integer, parameter :: n_solin_sub_mean = 48
    integer            :: n_solin_sub

    integer     :: klat, klon, n, i, j, m
    real(R8), pointer :: yc(:), xc(:)
    real(R8), allocatable :: accum(:,:)

    call seq_timemgr_EClockGetData(EClock, curr_ymd=CurrentYMD, curr_tod=CurrentTOD)
    call seq_timemgr_EClockGetData(EClock, calendar=calendar)

    call shr_cal_date2julian(CurrentYMD, CurrentTOD, julday, calendar)

    dt_days = real(eatm_model_dt, R8) / SHR_CONST_CDAY

    if (eatm_solin_window) then
      n_solin_sub = n_solin_sub_mean
    else
      n_solin_sub = 1
    end if

    allocate(yc(lsize), xc(lsize))
    klat = mct_aVect_indexRA(ggrid%data, 'lat')
    klon = mct_aVect_indexRA(ggrid%data, 'lon')
    yc(:) = ggrid%data%rAttr(klat, :)
    xc(:) = ggrid%data%rAttr(klon, :)

    allocate(accum(lsize_x, lsize_y))
    accum(:,:) = 0.0_R8

    do m = 1, n_solin_sub

      if (eatm_solin_window) then
        ! midpoint of sub-interval m within (T, T+dt]
        jsub = julday + dt_days * (real(m, R8) - 0.5_R8) / real(n_solin_sub, R8)
      else
        ! Ablation only (#35): the instantaneous value at the target time.
        jsub = julday + dt_days
      end if

      ! declination and the earth-sun distance factor drift slowly, but they
      ! are per-time not per-cell, so there is no reason to hold them fixed
      call shr_orb_decl(jsub, orb_eccen, orb_mvelpp, orb_lambm0, orb_obliqr, delta, eccf)

      n = 0
      do j = 1, lsize_y
        do i = 1, lsize_x
          n = n + 1
          lat_r = yc(n) * degtorad
          lon_r = xc(n) * degtorad
          cosz_val = shr_orb_cosz(jsub, lat_r, lon_r, delta)
          accum(i, j) = accum(i, j) + solar_const * eccf * max(0.0_R8, cosz_val)
        end do
      end do

    end do

    do j = 1, lsize_y
      do i = 1, lsize_x
        net_inputs(1, ix_in_solin, i, j) = &
             real(accum(i, j) / real(n_solin_sub, R8), R4)
      end do
    end do

    deallocate(accum)
    deallocate(yc, xc)

  end subroutine ace_compute_solin

end module ace_comp_mod
