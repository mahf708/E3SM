module samudra_comp_mod

  !-----------------------------------------------------------------------------
  ! Drives a traced Samudra-family ocean emulator through FTorch and translates
  ! between its channel layout and the fields the MCT coupler exchanges.
  !
  ! The emulator advances on its own timestep (5 days for SamudrACE-E3SMv3).
  ! The coupler runs far faster than that, so inference happens only on
  ! emulator-step boundaries and the coupler is handed a linear interpolation
  ! between the two bracketing emulator states.
  !
  ! Forcing.  The checkpoint declares its ten atmospheric flux channels as
  ! next-step forcing: in reference inference the model is handed the fluxes
  ! over the interval it is about to predict.  Online there is no such thing --
  ! the atmosphere has not run those five days yet.  EOCN therefore drives each
  ! step with the mean over the interval that just closed, which is a five-day
  ! persistence assumption on the forcing and the only causal choice available.
  ! The state time labels stay exact: t_im1 is valid at the boundary, t_ip1 one
  ! emulator step later, and the coupler sees the interval between them.
  !-----------------------------------------------------------------------------

  use esmf
  use eocnMod
  use eocnIO
  use eocn_channels_mod
  use mct_mod
  use seq_timemgr_mod, only: seq_timemgr_EClockGetData
  use shr_const_mod
  use shr_kind_mod, only: R4=>SHR_KIND_R4, R8=>SHR_KIND_R8, CS=>SHR_KIND_CS, CL=>SHR_KIND_CL, IN=>SHR_KIND_IN
  use shr_sys_mod,  only: shr_sys_flush, shr_sys_abort
  use shr_emul_ice_mod, only: shr_emul_ice_put

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
  private

  !--------------------------------------------------------------------------
  ! Seed libtorch's generators (eocn_torch_seed.cpp).  FTorch has no interface
  ! to them.  Samudra is deterministic, but the shim costs nothing and makes a
  ! coupled EATM+EOCN run reproducible as a whole.
  !--------------------------------------------------------------------------
  interface
    subroutine eocn_torch_manual_seed(seed) bind(C, name='eocn_torch_manual_seed')
      import :: c_int64_t
      integer(c_int64_t), value :: seed
    end subroutine eocn_torch_manual_seed
  end interface

  public :: samudra_comp_init
  public :: samudra_comp_run
  public :: samudra_comp_finalize

  !--- physical constants ---
  real(R8), parameter :: rhofw   = SHR_CONST_RHOFW    ! density of fresh water, kg/m3
  real(R8), parameter :: rearth  = SHR_CONST_REARTH   ! radius of the earth, m
  real(R8), parameter :: deg2rad = SHR_CONST_PI / 180.0_R8
  real(R8), parameter :: tfrz_sw = SHR_CONST_TKFRZSW  ! freezing point of sea water, K

  !--- torch handles ---
  type(torch_model) :: samudra_model
  type(torch_tensor), dimension(1) :: input_tensor
  type(torch_tensor), dimension(1) :: output_tensor

  integer(c_int)     :: tensor_layout(4)
  integer(c_int64_t) :: input_tensor_shape(4)
  integer(c_int64_t) :: output_tensor_shape(4)

  integer(c_int) :: model_device
  integer        :: n_tensor_in

  !--------------------------------------------------------------------------
  ! Model time of the last emulator advance.  Guarding on the clock alone is
  ! not enough: the driver can call a component's run method more than once at
  ! the same model time, and an emulator advanced twice there ends up
  ! permanently one step ahead of the coupler.  Recording the time makes the
  ! advance idempotent in model time.
  !--------------------------------------------------------------------------
  integer(IN) :: last_adv_ymd = -1
  integer(IN) :: last_adv_tod = -1

  logical :: first_inference = .true.

contains

  !===============================================================================
  subroutine samudra_comp_init(EClock, ggrid, read_restart)

    implicit none

    type(ESMF_Clock), intent(in)          :: EClock
    type(mct_gGrid),  intent(in), pointer :: ggrid
    logical,          intent(in)          :: read_restart

    integer     :: i, j, k
    real(R8)    :: t_frac
    integer(IN) :: CurrentYMD, CurrentTOD, stepno
    character(len=*), parameter :: subname = '(samudra_comp_init) '

    n_tensor_in = n_input_channels + n_forcing_channels

    input_tensor_shape = [ &
      int(1, kind=c_int64_t), &
      int(n_tensor_in, kind=c_int64_t), &
      int(lsize_y, kind=c_int64_t), &
      int(lsize_x, kind=c_int64_t) ]

    output_tensor_shape = [ &
      int(1, kind=c_int64_t), &
      int(n_output_channels, kind=c_int64_t), &
      int(lsize_y, kind=c_int64_t), &
      int(lsize_x, kind=c_int64_t) ]

    tensor_layout = [1_c_int, 2_c_int, 4_c_int, 3_c_int]

    select case (trim(eocn_model_device))
    case ('gpu', 'GPU', 'cuda', 'CUDA')
       model_device = torch_kCUDA
    case ('cpu', 'CPU')
       model_device = torch_kCPU
    case default
       call shr_sys_abort(trim(subname)//' ERROR: eocn_model_device must be '// &
            '"cpu" or "gpu", got "'//trim(eocn_model_device)//'"')
    end select

    if (len_trim(eocn_model_file) == 0) &
         call shr_sys_abort(trim(subname)//' ERROR: eocn_model_file is not set')

    write(logunit_ocn,*) trim(subname)//'loading traced model ', trim(eocn_model_file)
    write(logunit_ocn,*) trim(subname)//'device                = ', trim(eocn_model_device)
    write(logunit_ocn,*) trim(subname)//'input tensor channels = ', n_tensor_in
    call shr_sys_flush(logunit_ocn)

    call torch_model_load(samudra_model, trim(eocn_model_file), model_device)

    call samudra_cache_grid(ggrid)

    call seq_timemgr_EClockGetData( EClock, curr_ymd=CurrentYMD, curr_tod=CurrentTOD )
    call seq_timemgr_EClockGetData( EClock, stepno=stepno )

    if (eocn_rng_seed >= 0) then
      call eocn_torch_manual_seed(int(eocn_rng_seed, c_int64_t) + int(stepno, c_int64_t))
    end if

    if (read_restart) then

      t_frac = real(eocn_elapsed, kind=R8) / real(eocn_model_dt, kind=R8)
      call samudra_bracket_blend(t_frac)

    else

      ! The initial condition is the emulator state, and its own flux channels,
      ! at T0.  One inference carries it to T0 + dt, which is the upper bracket
      ! of the first interpolation interval.
      call samudra_inference()

      do k = 1, n_output_channels
        do j = 1, lsize_y
          do i = 1, lsize_x
            eocn_intrp%t_im1(k, i, j) = net_inputs(1, 12 + k, i, j)
            eocn_intrp%t_ip1(k, i, j) = net_outputs(1, k, i, j)
          end do
        end do
      end do

      eocn_elapsed = 0
      call samudra_bracket_blend(0.0_R8)

    endif

    ! The emulator now covers the interval starting at the current model time;
    ! a second initialization call at the same time must not advance it again.
    last_adv_ymd = CurrentYMD
    last_adv_tod = CurrentTOD

    call samudra_export(verbose=.true.)

  end subroutine samudra_comp_init

  !===============================================================================
  subroutine samudra_comp_run(EClock, ggrid)

    implicit none

    type(ESMF_Clock), intent(in)          :: EClock
    type(mct_gGrid),  intent(in), pointer :: ggrid

    integer     :: i, j, k
    integer     :: idt
    real(R8)    :: t_frac
    integer(IN) :: CurrentYMD, CurrentTOD

    call seq_timemgr_EClockGetData( EClock, curr_ymd=CurrentYMD, curr_tod=CurrentTOD )
    call seq_timemgr_EClockGetData( EClock, dtime=idt )

    ! The driver can call a component's run method more than once at the same
    ! model time.  Everything that moves the emulator forward -- including the
    ! elapsed-time counter itself -- sits behind this guard, so a repeated call
    ! re-exports the same state instead of stepping twice.
    if (.not. (CurrentYMD == last_adv_ymd .and. CurrentTOD == last_adv_tod)) then

      eocn_elapsed = eocn_elapsed + idt
      last_adv_ymd = CurrentYMD
      last_adv_tod = CurrentTOD

    end if

    if (eocn_elapsed >= eocn_model_dt) then

      eocn_elapsed = eocn_elapsed - eocn_model_dt

      ! close the flux window and hand its mean to the emulator
      call samudra_import_forcing()

      ! the state block of the input tensor is the state at this boundary
      do k = 1, n_output_channels
        do j = 1, lsize_y
          do i = 1, lsize_x
            net_inputs(1, 12 + k, i, j) = eocn_intrp%t_ip1(k, i, j)
            eocn_intrp%t_im1(k, i, j)   = eocn_intrp%t_ip1(k, i, j)
          end do
        end do
      end do

      call samudra_inference()

      do k = 1, n_output_channels
        do j = 1, lsize_y
          do i = 1, lsize_x
            eocn_intrp%t_ip1(k, i, j) = net_outputs(1, k, i, j)
          end do
        end do
      end do

      call samudra_reset_accumulators()

    end if

    t_frac = real(eocn_elapsed, kind=R8) / real(eocn_model_dt, kind=R8)
    call samudra_bracket_blend(t_frac)
    call samudra_export(verbose=.false.)

  end subroutine samudra_comp_run


  !===============================================================================
  subroutine samudra_import_forcing()

    ! Turn the running sums the coupler import has been feeding into interval
    ! means, in Samudra's units and sign convention.
    !
    ! The coupler's surface fluxes are positive *into* the surface; the
    ! emulator's FLUS/FSUS/LHFLX/SHFLX are positive upward, and its TAUX/TAUY
    ! are the stress on the atmosphere, so all of those change sign here.
    ! Precipitation arrives as kg/m2/s and the emulator wants m/s of liquid
    ! water equivalent.

    implicit none

    integer  :: i, j
    real(R8) :: w

    if (acc_n <= 0) then
      write(logunit_ocn,'(a)') '(samudra_import_forcing) WARNING: no coupler '// &
           'fluxes accumulated for this interval; holding the previous forcing'
      return
    end if

    w = 1.0_R8 / real(acc_n, R8)

    do j = 1, lsize_y
      do i = 1, lsize_x
        net_inputs(1, ix_in_taux,  i, j) = real(-acc_taux(i,j) * w, R4)
        net_inputs(1, ix_in_tauy,  i, j) = real(-acc_tauy(i,j) * w, R4)
        net_inputs(1, ix_in_prec,  i, j) = real(max(acc_prec(i,j) * w, 0.0_R8) / rhofw, R4)
        net_inputs(1, ix_in_snow,  i, j) = real(max(acc_snow(i,j) * w, 0.0_R8) / rhofw, R4)
        net_inputs(1, ix_in_flus,  i, j) = real(-acc_flus(i,j) * w, R4)
        net_inputs(1, ix_in_fsus,  i, j) = real( acc_fsus(i,j) * w, R4)
        net_inputs(1, ix_in_flds,  i, j) = real( acc_flds(i,j) * w, R4)
        net_inputs(1, ix_in_fsds,  i, j) = real( acc_fsds(i,j) * w, R4)
        net_inputs(1, ix_in_lhflx, i, j) = real(-acc_lhflx(i,j) * w, R4)
        net_inputs(1, ix_in_shflx, i, j) = real(-acc_shflx(i,j) * w, R4)
      end do
    end do

    write(logunit_ocn,'(a,i0,a)') '(samudra_import_forcing) closed a ', acc_n, &
         ' step flux window'
    write(logunit_ocn,'(a,4f12.4)') '(samudra_import_forcing) means TAUX TAUY FSDS FLDS: ', &
         gmean(net_inputs(1, ix_in_taux,:,:)), gmean(net_inputs(1, ix_in_tauy,:,:)), &
         gmean(net_inputs(1, ix_in_fsds,:,:)), gmean(net_inputs(1, ix_in_flds,:,:))
    write(logunit_ocn,'(a,4f12.4)') '(samudra_import_forcing) means FLUS FSUS LHFLX SHFLX: ', &
         gmean(net_inputs(1, ix_in_flus,:,:)), gmean(net_inputs(1, ix_in_fsus,:,:)), &
         gmean(net_inputs(1, ix_in_lhflx,:,:)), gmean(net_inputs(1, ix_in_shflx,:,:))
    call shr_sys_flush(logunit_ocn)

  end subroutine samudra_import_forcing

  !===============================================================================
  subroutine samudra_reset_accumulators()

    implicit none

    acc_taux(:,:)  = 0.0_R8
    acc_tauy(:,:)  = 0.0_R8
    acc_prec(:,:)  = 0.0_R8
    acc_snow(:,:)  = 0.0_R8
    acc_flus(:,:)  = 0.0_R8
    acc_fsus(:,:)  = 0.0_R8
    acc_flds(:,:)  = 0.0_R8
    acc_fsds(:,:)  = 0.0_R8
    acc_lhflx(:,:) = 0.0_R8
    acc_shflx(:,:) = 0.0_R8
    acc_n = 0

  end subroutine samudra_reset_accumulators

  !===============================================================================
  subroutine samudra_bracket_blend(t_frac)

    ! Linear interpolation between the bracketing emulator states.  Every
    ! Samudra output channel is a prognostic state, so unlike the atmosphere
    ! emulator there are no interval-mean channels to hold flat here.

    implicit none
    real(R8), intent(in) :: t_frac

    integer  :: i, j, k
    real(R4) :: f

    f = real(min(max(t_frac, 0.0_R8), 1.0_R8), R4)
    if (.not. eocn_interp_state) f = 0.0_R4

    do k = 1, n_output_channels
      do j = 1, lsize_y
        do i = 1, lsize_x
          net_outputs(1, k, i, j) = eocn_intrp%t_im1(k, i, j) + &
              f * (eocn_intrp%t_ip1(k, i, j) - eocn_intrp%t_im1(k, i, j))
        end do
      end do
    end do

  end subroutine samudra_bracket_blend

  !===============================================================================
  subroutine samudra_inference()

    implicit none

    integer :: k

    net_inputs_nn(1, 1:n_input_channels, :, :) = net_inputs(1, :, :, :)

    ! The traced graph slices a trailing next-step forcing block off the input
    ! tensor.  It does not read it -- the ocean checkpoint has no prescribed
    ! SST step -- but the slice has to be there for the shapes to line up, and
    ! the forcing values themselves already sit in their state-channel slots.
    do k = 1, n_forcing_channels
      net_inputs_nn(1, n_input_channels + k, :, :) = net_inputs(1, 2 + k, :, :)
    end do

    call samudra_sanitize_inputs()

    call torch_tensor_from_blob( &
      input_tensor(1), c_loc(net_inputs_nn), ndims=4_c_int, &
      tensor_shape=input_tensor_shape, layout=tensor_layout, &
      dtype=torch_kFloat32, device_type=model_device )
    call torch_tensor_from_blob( &
      output_tensor(1), c_loc(net_outputs), ndims=4_c_int, &
      tensor_shape=output_tensor_shape, layout=tensor_layout, &
      dtype=torch_kFloat32, device_type=torch_kCPU )

    call torch_model_forward(samudra_model, input_tensor, output_tensor)

    call torch_delete(input_tensor)
    call torch_delete(output_tensor)

    call samudra_validate_outputs()

  end subroutine samudra_inference

  !===============================================================================
  subroutine samudra_sanitize_inputs()

    ! Replace non-finite values in the input block with zero.
    !
    ! This is not defensive tidying.  Samudra's published initial conditions
    ! carry NaN over every land cell -- roughly a third of the grid -- because
    ! an ocean state is simply undefined there, and the convolutional stencil
    ! spreads a single NaN outward on every layer.  Because the emulator is
    ! autoregressive, one NaN at step zero is a NaN field for the rest of the
    ! run.  Zero is what the fme data loader substitutes, and the network was
    ! trained against that same substitution.
    !
    ! Anything non-finite arriving from the coupler instead means a flux went
    ! bad upstream, so that is counted and reported rather than passed over.

    implicit none

    integer :: k, i, j, nbad

    nbad = 0
    do k = 1, n_tensor_in
      do j = 1, lsize_y
        do i = 1, lsize_x
          if (net_inputs_nn(1, k, i, j) /= net_inputs_nn(1, k, i, j) .or. &
              abs(net_inputs_nn(1, k, i, j)) > huge(0.0_R4) * 0.5_R4) then
            net_inputs_nn(1, k, i, j) = 0.0_R4
            if (k <= 12) nbad = nbad + 1
          end if
        end do
      end do
    end do

    if (nbad > 0) then
      write(logunit_ocn,'(a,i0,a)') '(samudra_sanitize_inputs) WARNING: ', nbad, &
           ' non-finite values in the forcing channels were zeroed; these came '// &
           'from the coupler, not from the land mask'
      call shr_sys_flush(logunit_ocn)
    end if

  end subroutine samudra_sanitize_inputs

  !===============================================================================
  subroutine samudra_validate_outputs()

    ! A non-finite output is fatal.  The emulator is autoregressive and its
    ! convolutions have a growing receptive field, so one NaN becomes the whole
    ! field within a few steps and stays there.  Nothing downstream catches it:
    ! the export clamps compare against a bound, and every comparison with NaN
    ! is false, so a NaN sails through max() and reaches the atmosphere.
    !
    ! On the first inference the surface channels are also range-checked, which
    ! is the one moment a wrong channel table or a mis-traced graph is cheap to
    ! detect.

    implicit none

    integer  :: k, i, j, nbad
    real(R8) :: v, vmin, vmax
    character(len=*), parameter :: subname = '(samudra_validate_outputs) '

    nbad = 0
    do k = 1, n_output_channels
      do j = 1, lsize_y
        do i = 1, lsize_x
          if (net_outputs(1, k, i, j) /= net_outputs(1, k, i, j) .or. &
              abs(net_outputs(1, k, i, j)) > huge(0.0_R4) * 0.5_R4) nbad = nbad + 1
        end do
      end do
      if (nbad > 0) then
        write(logunit_ocn,'(a,i0,a)') trim(subname)//'ERROR: channel '// &
             trim(out_names(k))//' returned ', nbad, ' non-finite values'
        call shr_sys_abort(trim(subname)//' ERROR: the ocean emulator returned '// &
             'non-finite output in channel '//trim(out_names(k)))
      end if
    end do

    if (first_inference) then
      first_inference = .false.
      call range_check('sst', ix_out_sst,  260.0_R8, 320.0_R8)
      call range_check('ssh', ix_out_ssh,   -10.0_R8,  10.0_R8)
      call range_check('sss', ix_out_sss,     0.0_R8,  60.0_R8)
      call range_check('ocean_sea_ice_fraction', ix_out_sifrac, 0.0_R8, 1.0_R8)
    end if

  contains

    subroutine range_check(label, ix, lo, hi)
      character(len=*), intent(in) :: label
      integer,          intent(in) :: ix
      real(R8),         intent(in) :: lo, hi
      if (ix <= 0) return
      vmin =  huge(1.0_R8)
      vmax = -huge(1.0_R8)
      do j = 1, lsize_y
        do i = 1, lsize_x
          if (ocn_mask(i,j) < 0.5_R8) cycle
          v = real(net_outputs(1, ix, i, j), R8)
          vmin = min(vmin, v)
          vmax = max(vmax, v)
        end do
      end do
      write(logunit_ocn,'(a,2es13.5)') trim(subname)//trim(label)//' spans ', vmin, vmax
      if (vmin < lo .or. vmax > hi) call shr_sys_abort(trim(subname)// &
           ' ERROR: '//trim(label)//' is outside its physical range on the first '// &
           'inference -- check the channel table against the traced model metadata')
    end subroutine range_check

  end subroutine samudra_validate_outputs


  !===============================================================================
  subroutine samudra_cache_grid(ggrid)

    ! Cache cell centres so the sea surface height slope can be differenced.

    implicit none
    type(mct_gGrid), intent(in), pointer :: ggrid

    integer :: klat, klon, n, i, j

    klat = mct_aVect_indexRA(ggrid%data, 'lat')
    klon = mct_aVect_indexRA(ggrid%data, 'lon')

    n = 0
    do j = 1, lsize_y
      do i = 1, lsize_x
        n = n + 1
        cell_lat(i,j) = ggrid%data%rAttr(klat, n)
        cell_lon(i,j) = ggrid%data%rAttr(klon, n)
      end do
    end do

  end subroutine samudra_cache_grid

  !===============================================================================
  subroutine samudra_export(verbose)

    ! Pull the surface state the coupler wants out of the blended emulator
    ! state, clamp it to something physical, and fill land with a finite value.
    !
    ! Land is not merely cosmetic.  The coupler multiplies by the ocean
    ! fraction, so a land cell contributes nothing to the merge, but the
    ! surface flux scheme still evaluates its bulk formulae over the whole
    ! vector before that weighting.  A 0 K sea surface there produces
    ! infinities in the saturation humidity long before anyone multiplies by
    ! zero.

    implicit none
    logical, intent(in) :: verbose

    integer  :: i, j, n, ip, im, jp, jm
    real(R8) :: dx, dy, coslat
    real(R8) :: ifrac_flat(lsize_x*lsize_y)

    do j = 1, lsize_y
      do i = 1, lsize_x
        if (ocn_mask(i,j) > 0.5_R8) then
          so_t(i,j)     = max(real(net_outputs(1, ix_out_sst, i, j), R8), tfrz_sw)
          so_s(i,j)     = min(max(real(net_outputs(1, ix_out_sss, i, j), R8), 0.0_R8), 60.0_R8)
          so_u(i,j)     = real(net_outputs(1, ix_out_uvel, i, j), R8)
          so_v(i,j)     = real(net_outputs(1, ix_out_vvel, i, j), R8)
          so_ssh(i,j)   = real(net_outputs(1, ix_out_ssh,  i, j), R8)
          so_ifrac(i,j) = min(max(real(net_outputs(1, ix_out_sifrac, i, j), R8), 0.0_R8), 1.0_R8)
        else
          so_t(i,j)     = tfrz_sw
          so_s(i,j)     = 34.7_R8
          so_u(i,j)     = 0.0_R8
          so_v(i,j)     = 0.0_R8
          so_ssh(i,j)   = 0.0_R8
          so_ifrac(i,j) = 0.0_R8
        end if
      end do
    end do

    ! Sea surface height slope, centred differences, periodic in longitude.
    ! The poles get a one-sided difference; the grid has no cell there anyway.
    do j = 1, lsize_y
      jp = min(j + 1, lsize_y)
      jm = max(j - 1, 1)
      coslat = max(cos(cell_lat(1,j) * deg2rad), 1.0e-3_R8)
      dy = (cell_lat(1,jp) - cell_lat(1,jm)) * deg2rad * rearth
      do i = 1, lsize_x
        ip = i + 1
        if (ip > lsize_x) ip = 1
        im = i - 1
        if (im < 1) im = lsize_x
        dx = (360.0_R8 / real(lsize_x, R8)) * 2.0_R8 * deg2rad * rearth * coslat
        so_dhdx(i,j) = (so_ssh(ip,j) - so_ssh(im,j)) / dx
        if (abs(dy) > 0.0_R8) then
          so_dhdy(i,j) = (so_ssh(i,jp) - so_ssh(i,jm)) / dy
        else
          so_dhdy(i,j) = 0.0_R8
        end if
        if (ocn_mask(i,j) < 0.5_R8) then
          so_dhdx(i,j) = 0.0_R8
          so_dhdy(i,j) = 0.0_R8
        end if
      end do
    end do

    ! Publish the sea ice fraction for the emulator atmosphere.  The MCT
    ! coupler has no o2x field for it and, with a stub ice component, no ifrac
    ! of its own -- see shr_emul_ice_mod for why this exists and what replaces
    ! it.  Published as a fraction of the sea surface, which is what
    ! ocean_sea_ice_fraction means and what the atmosphere's formula expects.
    n = 0
    do j = 1, lsize_y
      do i = 1, lsize_x
        n = n + 1
        ifrac_flat(n) = so_ifrac(i,j)
      end do
    end do
    call shr_emul_ice_put(ifrac_flat)

    if (verbose) then
      write(logunit_ocn,'(a,4f12.5)') &
           '(samudra_export) ocean-mean SST, SSS, ice frac, |SSH|: ', &
           omean(so_t), omean(so_s), omean(so_ifrac), omean(abs(so_ssh))
      call shr_sys_flush(logunit_ocn)
    end if

  end subroutine samudra_export

  !===============================================================================
  real(R8) function gmean(field)

    ! Unweighted grid mean of a single input channel.  Only used for log lines,
    ! where the point is to see whether a forcing is the right order of
    ! magnitude and the right sign, not to close a budget.

    implicit none
    real(R4), intent(in) :: field(:,:)

    gmean = real(sum(real(field, R8)) / real(size(field), R8), R8)

  end function gmean

  !===============================================================================
  real(R8) function omean(field)

    ! Same, restricted to ocean cells.

    implicit none
    real(R8), intent(in) :: field(:,:)

    real(R8) :: s, w
    integer  :: i, j

    s = 0.0_R8
    w = 0.0_R8
    do j = 1, lsize_y
      do i = 1, lsize_x
        if (ocn_mask(i,j) > 0.5_R8) then
          s = s + field(i,j)
          w = w + 1.0_R8
        end if
      end do
    end do
    omean = s / max(w, 1.0_R8)

  end function omean

  !===============================================================================
  subroutine samudra_comp_finalize()

    implicit none
    call torch_delete(samudra_model)

  end subroutine samudra_comp_finalize

end module samudra_comp_mod
