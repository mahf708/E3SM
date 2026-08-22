module ace_comp_mod

  use esmf
  use eatmMod
  use eatmIO
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
  ! Public module data
  !--------------------------------------------------------------------------
  integer, parameter, public :: n_input_channels=39  ! number of input channels to emulator
  integer, parameter, public :: n_output_channels=44 ! number of output channels to emulator
  integer, parameter, public :: eatm_idt=6 * 60 * 60 ! eatm timestep (6hr) in seconds
  ! number of eatm steps in a year
  integer, parameter, public :: eatm_spy=(365 * 24 * 60 * 60) / eatm_idt
  ! TODO (AN): Parse from namelist
  integer(IN), parameter, public :: iradsw=1    ! radiation interval

  !--------------------------------------------------------------------------
  ! Private module data
  !--------------------------------------------------------------------------
  real(R8), parameter :: rdair  = SHR_CONST_RDAIR  ! dry air gas constant   ~ J/K/kg
  real(R8), parameter :: tKFrz  = SHR_CONST_TKFRZ

  real(R8), parameter :: solar_const = 1368.22_R8  ! total solar irradiance (W/m2), matches RRTMG

  ! Set up Torch data structures
  type(torch_model) :: ace_model
  type(torch_tensor), dimension(1) :: input_tensor
  type(torch_tensor), dimension(1) :: output_tensor

  ! ACE2-EAMv3 output channels 35-44 (fluxes, precipitation) are means over
  ! the emulator step; 1-34 are snapshots at the end of it.
  integer, parameter :: first_mean_channel = 35

  ! TOA insolation: the window mean the emulator was driven with, and the
  ! instantaneous value now.  Their ratio restores the diurnal cycle on the
  ! interval-mean shortwave.
  real(R8), allocatable :: solin_win(:,:)
  real(R8), allocatable :: solin_now(:,:)
  logical               :: solin_ready = .false.

  ! time of the last emulator advance, so a repeated call at the same model
  ! time does not step the emulator twice
  integer(IN) :: last_adv_ymd = -1
  integer(IN) :: last_adv_tod = -1

  integer(c_int) :: tensor_layout(4)
  integer(c_int64_t) :: input_tensor_shape(4)
  integer(c_int64_t) :: output_tensor_shape(4)

  ! TODO (AN): Parse from namelist
  character(len=*), parameter :: torchscript_file="/pscratch/sd/m/mahf708/test_ace_repo/test_trace_cuda.pt"
  ! character(len=*), parameter :: norm_file="/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/ace2_EAMv3_normalize.nc"
  ! character(len=*), parameter :: denorm_file="/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/ace2_EAMv3_denormalize.nc"
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

    allocate(solin_win(lsize_x, lsize_y), solin_now(lsize_x, lsize_y))
    solin_win(:,:) = 0.0_R8
    solin_now(:,:) = 0.0_R8

    call seq_timemgr_EClockGetData( EClock, curr_ymd=CurrentYMD, curr_tod=CurrentTOD )

    input_tensor_shape = [ &
      int(1, kind=c_int64_t), &
      int(n_input_channels, kind=c_int64_t), &
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

    ! call init_normalizer(normalizer, norm_file, n_input_channels)
    ! call init_normalizer(denormalizer, denorm_file, n_output_channels)

    ! load the traced model
    call torch_model_load(ace_model, torchscript_file, torch_kCUDA)

    if (read_restart) then

      ! int remainder (in sec) of coupler timestep relative to ACE timestep
      t_modulo = mod(CurrentTOD, eatm_idt)
      ! turn integer remainder into fraction through ACE timestep
      t_frac = real(t_modulo, kind=R8) / real(eatm_idt, kind=R8)

      call ace_bracket_blend(t_frac)
      call ace_capture_solin_window()   ! SOLIN comes back from the restart file

    else
      call ace_compute_solin(EClock, ggrid)
      call ace_capture_solin_window()

      net_inputs_nn = net_inputs
      ! normalize, can probably happen after tensor is made becuase it's a pointer
      ! call normalizer%normalize(net_inputs_nn)

      ! create input/output tensors based off net input/output arrays
      call torch_tensor_from_blob(&
        input_tensor(1), &
        c_loc(net_inputs_nn), &
        ndims=4_c_int, &
        tensor_shape=input_tensor_shape, &
        layout=tensor_layout, &
        dtype=torch_kFloat32, &
        device_type=torch_kCUDA &
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

      ! run inference
      call torch_model_forward(ace_model, input_tensor, output_tensor)

      ! Clean up C++ pointers
      call torch_delete(input_tensor)
      call torch_delete(output_tensor)

      ! denormalize
      ! call denormalizer%denormalize(net_outputs)

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

    last_adv_ymd = CurrentYMD
    last_adv_tod = CurrentTOD

    call ace_solin_now(EClock, ggrid)

    ! using restart data from ACE set the fields passed to the coupler
    call ace_eatm_export(ggrid)

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

    write(logunit_atm, *) "stepno: ", stepno
    write(logunit_atm, *) "cpl_idt: ", cpl_idt
    write(logunit_atm, *) "eatm_idt: ", eatm_idt
    write(logunit_atm, *) "CurrentYMD: ", CurrentYMD
    write(logunit_atm, *) "CurrentTOD: ", CurrentTOD
    call shr_sys_flush(logunit_atm)

    ! integer remainder (in sec) of coupler timestep relative to ACE timestep
    t_modulo = mod(CurrentTOD, eatm_idt)

    ! An emulator step is due at every multiple of eatm_idt, but only once per
    ! model time: the driver's phase-2 initialization call runs this routine
    ! again at the time the startup inference already covered.
    if (t_modulo .eq. 0 .and. &
        .not. (CurrentYMD .eq. last_adv_ymd .and. CurrentTOD .eq. last_adv_tod)) then

      ! Feed the emulator its own prediction for this time.  net_outputs still
      ! holds the field the coupler was handed at the previous coupler step,
      ! which is a partial interpolation towards t_ip1 and not a state the
      ! emulator was trained to consume.
      net_outputs(1, :, :, :) = eatm_intrp%t_ip1(:, :, :)

      call ace_eatm_import()
      call ace_compute_solin(EClock, ggrid)
      call ace_capture_solin_window()

      net_inputs_nn = net_inputs
      ! normalize, can probably happen after tensor is made becuase it's a pointer
      ! call normalizer%normalize(net_inputs_nn)

      ! create input/output tensors based off net input/output arrays
      call torch_tensor_from_blob(&
        input_tensor(1), &
        c_loc(net_inputs_nn), &
        ndims=4_c_int, &
        tensor_shape=input_tensor_shape, &
        layout=tensor_layout, &
        dtype=torch_kFloat32, &
        device_type=torch_kCUDA &
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

      ! run inference
      call torch_model_forward(ace_model, input_tensor, output_tensor)

      ! Clean up C++ pointers
      call torch_delete(input_tensor)
      call torch_delete(output_tensor)

      ! denormalize
      ! call denormalizer%denormalize(net_outputs)

      ! advance the time levels: old t_ip1 (state at current time T)
      ! becomes t_im1; new inference (state at T+6h) becomes t_ip1.
      ! Interpolation between them gives smooth transition over [T, T+6h].
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

    t_frac = real(t_modulo, kind=r8) / real(eatm_idt, kind=r8)

    call ace_bracket_blend(t_frac)

    ! diurnal shape for this coupler step, applied to the shortwave on export
    call ace_solin_now(EClock, ggrid)

    call ace_eatm_export(ggrid)

  end subroutine ace_comp_run

  subroutine ace_bracket_blend(t_frac)
    !----------------------------------------------------------------
    ! Combine the two bracketing emulator states into the field handed to the
    ! coupler at a fraction t_frac through the current emulator interval.
    !
    ! Snapshot channels are instantaneous values at the bracket times and are
    ! interpolated.  Interval-mean channels are already the mean over the
    ! interval being stepped across, so t_ip1 is the answer everywhere inside
    ! it; interpolating them from the previous interval's mean lags the surface
    ! radiation and precipitation by half an emulator step.
    !----------------------------------------------------------------
    implicit none

    real(R8), intent(in) :: t_frac

    integer  :: i, j, k

    do k = 1, n_output_channels
      do j = 1, lsize_y
        do i = 1, lsize_x
          if (k >= first_mean_channel) then
            net_outputs(1, k, i, j) = eatm_intrp%t_ip1(k, i, j)
          else
            net_outputs(1, k, i, j) = eatm_intrp%t_im1(k, i, j) + &
                t_frac * (eatm_intrp%t_ip1(k, i, j) - eatm_intrp%t_im1(k, i, j))
          end if
        end do
      end do
    end do

  end subroutine ace_bracket_blend

  subroutine ace_comp_finalize()
    call torch_delete(ace_model)
    if (allocated(solin_win)) deallocate(solin_win, solin_now)
    ! call finalize_normalizer(normalizer)
    ! call finalize_normalizer(denormalizer)
  end subroutine ace_comp_finalize

  subroutine ace_eatm_import()
    !----------------------------------------------------------------
    ! !DESCRIPTION:
    ! Set net_inputs from coupler imports and previous model outputs.
    ! PHIS (channel 4) persists from init. SOLIN (channel 5) is
    ! computed separately by ace_compute_solin.
    !----------------------------------------------------------------
    implicit none

    ! !LOCAL VARIABLES:
    integer  :: i, j
    real(R8) :: deficit   ! cell fraction no surface model covers

    do j = 1, lsize_y
      do i = 1, lsize_x
        ! lndfrac is the coupler's Sf_lfrac, the fraction claimed by the *land
        ! model*, which is identically zero with a stub land model.  That left
        ! LANDFRAC at zero everywhere and the coupler's merged surface
        ! temperature at 0 K over every land point.  Report the uncovered
        ! fraction as land and let the emulator own the surface there; with a
        ! land model running the deficit is zero and both lines pass through.
        deficit = max(0.0_R8, 1.0_R8 - ocnfrac(i, j) - icefrac(i, j) - lndfrac(i, j))
        net_inputs(1,  1, i, j) = lndfrac(i, j) + deficit  ! ACE2-EAMv3: LANDFRAC
        net_inputs(1,  2, i, j) = ocnfrac(i, j)            ! ACE2-EAMv3: OCNFRAC
        net_inputs(1,  3, i, j) = icefrac(i, j)            ! ACE2-EAMv3: ICEFRAC
        net_inputs(1,  6, i, j) = net_outputs(1, 1, i, j)  ! ACE2-EAMv3: PS
        net_inputs(1,  7, i, j) = ts(i, j) + deficit * net_outputs(1, 2, i, j)  ! ACE2-EAMv3: TS
        ! For 3D fields just advance through with time
        net_inputs(1,  8, i, j) = net_outputs(1,  3, i, j) ! ACE2-EAMv3: T_0
        net_inputs(1,  9, i, j) = net_outputs(1,  4, i, j) ! ACE2-EAMv3: T_1
        net_inputs(1, 10, i, j) = net_outputs(1,  5, i, j) ! ACE2-EAMv3: T_2
        net_inputs(1, 11, i, j) = net_outputs(1,  6, i, j) ! ACE2-EAMv3: T_3
        net_inputs(1, 12, i, j) = net_outputs(1,  7, i, j) ! ACE2-EAMv3: T_4
        net_inputs(1, 13, i, j) = net_outputs(1,  8, i, j) ! ACE2-EAMv3: T_5
        net_inputs(1, 14, i, j) = net_outputs(1,  9, i, j) ! ACE2-EAMv3: T_6
        net_inputs(1, 15, i, j) = net_outputs(1, 10, i, j) ! ACE2-EAMv3: T_7
        net_inputs(1, 16, i, j) = net_outputs(1, 11, i, j) ! ACE2-EAMv3: specific_total_water_0
        net_inputs(1, 17, i, j) = net_outputs(1, 12, i, j) ! ACE2-EAMv3: specific_total_water_1
        net_inputs(1, 18, i, j) = net_outputs(1, 13, i, j) ! ACE2-EAMv3: specific_total_water_2
        net_inputs(1, 19, i, j) = net_outputs(1, 14, i, j) ! ACE2-EAMv3: specific_total_water_3
        net_inputs(1, 20, i, j) = net_outputs(1, 15, i, j) ! ACE2-EAMv3: specific_total_water_4
        net_inputs(1, 21, i, j) = net_outputs(1, 16, i, j) ! ACE2-EAMv3: specific_total_water_5
        net_inputs(1, 22, i, j) = net_outputs(1, 17, i, j) ! ACE2-EAMv3: specific_total_water_6
        net_inputs(1, 23, i, j) = net_outputs(1, 18, i, j) ! ACE2-EAMv3: specific_total_water_7
        net_inputs(1, 24, i, j) = net_outputs(1, 19, i, j) ! ACE2-EAMv3: U_0
        net_inputs(1, 25, i, j) = net_outputs(1, 20, i, j) ! ACE2-EAMv3: U_1
        net_inputs(1, 26, i, j) = net_outputs(1, 21, i, j) ! ACE2-EAMv3: U_2
        net_inputs(1, 27, i, j) = net_outputs(1, 22, i, j) ! ACE2-EAMv3: U_3
        net_inputs(1, 28, i, j) = net_outputs(1, 23, i, j) ! ACE2-EAMv3: U_4
        net_inputs(1, 29, i, j) = net_outputs(1, 24, i, j) ! ACE2-EAMv3: U_5
        net_inputs(1, 30, i, j) = net_outputs(1, 25, i, j) ! ACE2-EAMv3: U_6
        net_inputs(1, 31, i, j) = net_outputs(1, 26, i, j) ! ACE2-EAMv3: U_7
        net_inputs(1, 32, i, j) = net_outputs(1, 27, i, j) ! ACE2-EAMv3: V_0
        net_inputs(1, 33, i, j) = net_outputs(1, 28, i, j) ! ACE2-EAMv3: V_1
        net_inputs(1, 34, i, j) = net_outputs(1, 29, i, j) ! ACE2-EAMv3: V_2
        net_inputs(1, 35, i, j) = net_outputs(1, 30, i, j) ! ACE2-EAMv3: V_3
        net_inputs(1, 36, i, j) = net_outputs(1, 31, i, j) ! ACE2-EAMv3: V_4
        net_inputs(1, 37, i, j) = net_outputs(1, 32, i, j) ! ACE2-EAMv3: V_5
        net_inputs(1, 38, i, j) = net_outputs(1, 33, i, j) ! ACE2-EAMv3: V_6
        net_inputs(1, 39, i, j) = net_outputs(1, 34, i, j) ! ACE2-EAMv3: V_7
      enddo
    enddo

    write(logunit_atm, *) "----------------------------------------------------------------"
    write(logunit_atm, *) "ace_eatm_import"
    write(logunit_atm, *) "----------------------------------------------------------------"
    write(logunit_atm, *) "ts  (min, max):   ( ", minval(ts(:, :)),  maxval(ts(:, :)), " )"
    call shr_sys_flush(logunit_atm)

  end subroutine ace_eatm_import

  subroutine ace_eatm_export(ggrid)
    ! !LOCAL VARIABLES:
    type(mct_gGrid), pointer :: ggrid
    real(R8),        pointer :: yc(:)

    integer(IN) :: klat
    integer     :: i, j, n
    real(R8)    :: e, avg_alb
    real(R8)    :: p_int, tv, fsds_dn, fsus_up, sw_scale
    real(R8), parameter :: ak_7 = 2328.4749
    real(R8), parameter :: bk_7 = 0.8722759
    real(R8), parameter :: degtorad = SHR_CONST_PI/180.0_R8

    allocate(yc(lsize))

    klat = mct_aVect_indexRA(ggrid%data,'lat')
    yc(:) = ggrid%data%rAttr(klat,:)

    n = 0
    do j = 1, lsize_y
      do i = 1, lsize_x
        n = n + 1

        pslv(i, j) = net_outputs(1,  1, i, j) ! PS (Surface pressure)
        ubot(i, j) = net_outputs(1, 26, i, j) ! U_7
        vbot(i, j) = net_outputs(1, 34, i, j) ! V_7
        tbot(i, j) = net_outputs(1, 10, i, j) ! T_7

        !--- T_7/U_7/V_7/STW_7 are means over the lowest layer, so the state
        !--- handed to the coupler belongs at the layer midpoint, and Sa_z is a
        !--- height above the surface rather than a pressure altitude above sea
        !--- level.
        p_int = ak_7 + bk_7 * pslv(i, j)          ! top of the lowest layer
        pbot(i, j) = 0.5_R8 * (pslv(i, j) + p_int)

        !--- specific humidity: the emulator's own STW_7.  shr_flux_atmOcn
        !--- reads Sa_shum as vapour, so cap the condensate off at saturation.
        e = datm_shr_esat(tbot(i, j), tbot(i, j))
        shum(i, j) = max(real(net_outputs(1, 18, i, j), R8), 0.0_R8)  ! STW_7
        shum(i, j) = min(shum(i, j), &
             (0.622_R8 * e) / max(pbot(i, j) - 0.378_R8 * e, 1.0_R8))

        tv = tbot(i, j) * (1.0_R8 + 0.608_R8 * shum(i, j))
        zbot(i, j) = (rdair * tv / SHR_CONST_G) * log(pslv(i, j) / pbot(i, j))

        ptem(i, j) = tbot(i,j) * (pslv(i,j)/pbot(i,j))**(rdair/SHR_CONST_CPDAIR)
        lwdn(i, j) = net_outputs(1, 40, i, j) ! FLDS (Downwelling longwave flux at surface)

        !--- density ---
        dens(i, j) = pbot(i, j)  / (rdair * tbot(i, j) * (1 + 0.608_R8 * shum(i, j)))

        snowc(i, j) = 0.0_R8
        rainc(i, j) = 0.0_R8
        if (tbot(i, j) < tKFrz) then
          rainl(i, j) = 0.0_R8
          snowl(i, j) = max(net_outputs(1, 37, i, j), 0.0_R8)
        else
          rainl(i, j) = max(net_outputs(1, 37, i, j), 0.0_R8)
          snowl(i, j) = 0.0_R8
        endif

        !--- FSDS and FSUS are means over the emulator step.  Put the diurnal
        !--- cycle back on them with the ratio of the instantaneous insolation
        !--- now to the window mean the emulator was driven with.
        fsds_dn = max(real(net_outputs(1, 41, i, j), R8), 0.0_R8)  ! FSDS
        fsus_up = max(real(net_outputs(1, 42, i, j), R8), 0.0_R8)  ! FSUS
        if (solin_ready) then
          if (solin_win(i, j) > 1.0_R8) then
            sw_scale = solin_now(i, j) / solin_win(i, j)
          else
            sw_scale = 0.0_R8   ! polar night: no sun in the window, none now
          end if
          fsds_dn = fsds_dn * sw_scale
          fsus_up = fsus_up * sw_scale
        end if

        !--- fabricate required sw[n,v]d[r,f] components from the downwelling ---
        swvdr(i, j) = fsds_dn * 0.28_R8
        swndr(i, j) = fsds_dn * 0.31_R8
        swvdf(i, j) = fsds_dn * 0.24_R8
        swndf(i, j) = fsds_dn * 0.17_R8
        swnet(i, j) = max(fsds_dn - fsus_up, 0.0_R8)

        ! avg_alb = ( 0.069 - 0.011*cos(2.0_R8*yc(n)*degtorad ) )
        ! swnet(i, j) = swnet(i, j) * (1.0_R4 - REAL(avg_alb, R4))
      enddo
    enddo

    deallocate(yc)

    write(logunit_atm, *) "----------------------------------------------------------------"
    write(logunit_atm, *) "ace_eatm_export"
    write(logunit_atm, *) "----------------------------------------------------------------"
    write(logunit_atm, *) "zbot  (min, max):   ( ", minval(zbot(:, :)),  maxval(zbot(:, :)), " )"
    write(logunit_atm, *) "tbot   (min, max):  ( ", minval(tbot(:, :)),  maxval(tbot(:, :)), " )"
    write(logunit_atm, *) "pbot   (min, max):  ( ", minval(pbot(:, :)),  maxval(pbot(:, :)), " )"
    write(logunit_atm, *) "ubot   (min, max):  ( ", minval(ubot(:, :)),  maxval(ubot(:, :)), " )"
    write(logunit_atm, *) "vbot   (min, max):  ( ", minval(vbot(:, :)),  maxval(vbot(:, :)), " )"
    write(logunit_atm, *) "swnet  (min, max):  ( ", minval(swnet(:, :)), maxval(swnet(:, :)), " )"
    write(logunit_atm, *) "rainl (min, max):   ( ", minval(rainl(:, :)), maxval(rainl(:, :)), " )"
    write(logunit_atm, *) "snowl (min, max):   ( ", minval(snowl(:, :)), maxval(snowl(:, :)), " )"
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

  subroutine ace_solin_now(EClock, ggrid)
    !----------------------------------------------------------------
    ! Instantaneous TOA insolation at the current coupler time.  Divided by the
    ! window mean the emulator was driven with, this is the diurnal shape that
    ! ace_eatm_export puts back onto the interval-mean shortwave.
    !----------------------------------------------------------------
    implicit none
    type(ESMF_Clock), intent(in) :: EClock
    type(mct_gGrid),  intent(in), pointer :: ggrid

    integer(IN)       :: CurrentYMD, CurrentTOD
    character(len=CS) :: calendar
    real(R8)          :: julday, delta, eccf, lat_r, lon_r
    real(R8), parameter :: degtorad = SHR_CONST_PI / 180.0_R8

    integer     :: klat, klon, n, i, j
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

  subroutine ace_capture_solin_window()
    ! Keep the window mean the emulator is about to be driven with, as the
    ! denominator of the diurnal rescaling over the interval it covers.
    implicit none
    integer :: i, j
    do j = 1, lsize_y
      do i = 1, lsize_x
        solin_win(i, j) = real(net_inputs(1, 5, i, j), R8)
      end do
    end do
    solin_ready = .true.
  end subroutine ace_capture_solin_window

  subroutine ace_compute_solin(EClock, ggrid)
    !----------------------------------------------------------------
    ! Compute SOLIN (solar insolation at TOA) from orbital mechanics, as the
    ! *mean over the emulator step about to be taken*:
    !
    !   SOLIN = (1/dt) * integral over (T, T+dt] of S0 * eccf * max(0, cosz)
    !
    ! SOLIN is a next-step forcing channel, so it belongs to the prediction
    ! target rather than to the input state's own time; and it carries
    ! cell_methods = "time: mean" in the training stream, so it is the mean
    ! over the 6 h leading up to that timestamp.  The instantaneous field at
    ! T+dt has the same global mean but is a different field point by point --
    ! a cosine bullseye at the subsolar point rather than a band 90 degrees of
    ! longitude wide -- with an RMS difference of 330 W/m2.
    !
    ! Evaluated by the midpoint rule on n_solin_sub sub-intervals.  Against a
    ! 2400-point reference, 48 sub-steps leave an RMS error of 0.03 W/m2.
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
    integer,  parameter :: n_solin_sub = 48

    integer     :: klat, klon, n, i, j, m
    real(R8), pointer :: yc(:), xc(:)
    real(R8), allocatable :: accum(:,:)

    call seq_timemgr_EClockGetData(EClock, curr_ymd=CurrentYMD, curr_tod=CurrentTOD)
    call seq_timemgr_EClockGetData(EClock, calendar=calendar)

    call shr_cal_date2julian(CurrentYMD, CurrentTOD, julday, calendar)

    dt_days = real(eatm_idt, R8) / SHR_CONST_CDAY

    allocate(yc(lsize), xc(lsize))
    klat = mct_aVect_indexRA(ggrid%data, 'lat')
    klon = mct_aVect_indexRA(ggrid%data, 'lon')
    yc(:) = ggrid%data%rAttr(klat, :)
    xc(:) = ggrid%data%rAttr(klon, :)

    allocate(accum(lsize_x, lsize_y))
    accum(:,:) = 0.0_R8

    do m = 1, n_solin_sub

      ! midpoint of sub-interval m within (T, T+dt]
      jsub = julday + dt_days * (real(m, R8) - 0.5_R8) / real(n_solin_sub, R8)

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
        net_inputs(1, 5, i, j) = real(accum(i, j) / real(n_solin_sub, R8), R4)
      end do
    end do

    deallocate(accum)
    deallocate(yc, xc)

  end subroutine ace_compute_solin

  ! Define here becasue we need the eatmIO mod and trying to avoid circular imports
  subroutine init_normalizer(norm, norm_file, n)
    implicit none
    class(t_normalization_struct), intent(out) :: norm
    character(len=*),   intent(in)  :: norm_file
    integer,            intent (in) :: n  ! number of variables
    ! !LOCAL VARIABLES:
    type(file_desc_t)          :: ncid    ! netcdf file id
    logical                    :: found
    character(len=*),parameter :: subname = '(init_normalizer) '

    allocate(norm%stds(n))
    allocate(norm%means(n))

    call ncd_pio_openfile(ncid, trim(norm_file), 0)

    call ncd_io(varname='stds', data=norm%stds, flag='read', ncid=ncid, readvar=found)
    if ( .not. found ) call shr_sys_abort( trim(subname)//' ERROR: reading -- stds -- from ' // trim(norm_file))

    call ncd_io(varname='means', data=norm%means, flag='read', ncid=ncid, readvar=found)
    if ( .not. found ) call shr_sys_abort( trim(subname)//' ERROR: reading -- means -- from ' // trim(norm_file))

    call ncd_pio_closefile(ncid)
  end subroutine init_normalizer

  subroutine finalize_normalizer(norm)
    implicit none
    class(t_normalization_struct), intent(inout) :: norm

    deallocate(norm%stds)
    deallocate(norm%means)

  end subroutine finalize_normalizer

end module ace_comp_mod
