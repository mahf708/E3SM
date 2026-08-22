module eocn_comp_mod

  ! !USES:

  use mct_mod
  use esmf
  use perf_mod
  use shr_const_mod
  use shr_sys_mod
  use shr_kind_mod   , only: IN=>SHR_KIND_IN, R4=>SHR_KIND_R4, R8=>SHR_KIND_R8, CS=>SHR_KIND_CS, CL=>SHR_KIND_CL
  use shr_file_mod   , only: shr_file_getunit, shr_file_freeunit
  use shr_cal_mod    , only: shr_cal_ymdtod2string
  use shr_mpi_mod    , only: shr_mpi_bcast
  use seq_timemgr_mod, only: seq_timemgr_EClockGetData, seq_timemgr_RestartAlarmIsOn
  use eocnMod
  use eocnSpmdMod
  use eocnIO
  use eocn_channels_mod
  use eocn_restart_mod
  use samudra_comp_mod

  implicit none
  private

  public :: eocn_comp_alloc
  public :: eocn_comp_init
  public :: eocn_comp_run
  public :: eocn_comp_final

  save

contains

  !===============================================================================
  subroutine eocn_comp_alloc(read_restart)

    ! Resolve the channel table, size everything to it, and load the emulator
    ! state.  Separate from eocn_comp_init because the coupler's ocean domain
    ! needs the sea surface fraction, and that is one of the emulator's own
    ! input channels -- so the state has to be on disk-to-memory before the
    ! domain is built.

    implicit none

    logical, intent(in) :: read_restart

    logical       :: exists
    integer(IN)   :: nu
    character(*), parameter :: F00 = "('(eocn_comp_alloc) ',8a)"
    character(*), parameter :: subName = "(eocn_comp_alloc) "

    call t_startf('EOCN_ALLOC')

    call eocn_channels_init(eocn_emulator, logunit_ocn)

    allocate(acc_taux(lsize_x,lsize_y))
    allocate(acc_tauy(lsize_x,lsize_y))
    allocate(acc_prec(lsize_x,lsize_y))
    allocate(acc_snow(lsize_x,lsize_y))
    allocate(acc_flus(lsize_x,lsize_y))
    allocate(acc_fsus(lsize_x,lsize_y))
    allocate(acc_flds(lsize_x,lsize_y))
    allocate(acc_fsds(lsize_x,lsize_y))
    allocate(acc_lhflx(lsize_x,lsize_y))
    allocate(acc_shflx(lsize_x,lsize_y))
    acc_taux = 0.0_R8 ; acc_tauy = 0.0_R8 ; acc_prec = 0.0_R8 ; acc_snow = 0.0_R8
    acc_flus = 0.0_R8 ; acc_fsus = 0.0_R8 ; acc_flds = 0.0_R8 ; acc_fsds = 0.0_R8
    acc_lhflx = 0.0_R8 ; acc_shflx = 0.0_R8
    acc_n = 0

    allocate(so_t(lsize_x,lsize_y))
    allocate(so_s(lsize_x,lsize_y))
    allocate(so_u(lsize_x,lsize_y))
    allocate(so_v(lsize_x,lsize_y))
    allocate(so_ssh(lsize_x,lsize_y))
    allocate(so_dhdx(lsize_x,lsize_y))
    allocate(so_dhdy(lsize_x,lsize_y))
    allocate(so_ifrac(lsize_x,lsize_y))
    ! so_ifrac is read every coupling step by the flux un-weighting, and
    ! published to EICE, but assigned only inside samudra_export.  In practice
    ! samudra_comp_init calls that routine before the first coupling step on
    ! both startup and restart, so the value is always defined by the time
    ! anyone reads it -- this only removes the dependence on that being true.
    so_ifrac = 0.0_R8

    allocate(cell_lat(lsize_x,lsize_y))
    allocate(cell_lon(lsize_x,lsize_y))
    allocate(ocn_mask(lsize_x,lsize_y))
    ocn_mask = 1.0_R8

    allocate(net_inputs(1, n_input_channels, lsize_x, lsize_y))
    allocate(net_outputs(1, n_output_channels, lsize_x, lsize_y))
    allocate(net_inputs_nn(1, n_input_channels + n_forcing_channels, lsize_x, lsize_y))
    net_inputs    = 0.0_R4
    net_outputs   = 0.0_R4
    net_inputs_nn = 0.0_R4

    allocate(eocn_intrp%t_im1(n_output_channels, lsize_x, lsize_y))
    allocate(eocn_intrp%t_ip1(n_output_channels, lsize_x, lsize_y))

    !----------------------------------------------------------------------
    ! Restart or initial condition
    !----------------------------------------------------------------------
    if (read_restart) then
       if (masterproc) then
          inquire(file=trim(rpfile)//trim(inst_suffix),exist=exists)
          if (.not.exists) call shr_sys_abort(trim(subname)// &
               ' ERROR: rpointer file missing')
          nu = shr_file_getUnit()
          open(nu,file=trim(rpfile)//trim(inst_suffix),form='formatted')
          read(nu,'(a)') restart_file
          close(nu)
          call shr_file_freeUnit(nu)
       endif
       call shr_mpi_bcast(restart_file,mpicom_ocn,'restart_file')
       if (masterproc) call eocn_restart_file_read(restart_file)
    else
       if (masterproc) then
          if (len_trim(eocn_ic_file) == 0) call shr_sys_abort(trim(subname)// &
               ' ERROR: eocn_ic_file must be set for a startup run')
          call eocn_initial_condition_file_read(eocn_ic_file)
       endif
    endif

    call t_stopf('EOCN_ALLOC')

  end subroutine eocn_comp_alloc


  !===============================================================================
  subroutine eocn_comp_init(Eclock, x2o, o2x, &
       seq_flds_x2o_fields, seq_flds_o2x_fields, &
       gsmap, ggrid, read_restart)

    implicit none

    type(ESMF_Clock)       , intent(in)    :: EClock
    type(mct_aVect)        , intent(inout) :: x2o, o2x
    character(len=*)       , intent(in)    :: seq_flds_x2o_fields
    character(len=*)       , intent(in)    :: seq_flds_o2x_fields
    type(mct_gsMap)        , pointer       :: gsMap
    type(mct_gGrid)        , pointer       :: ggrid
    logical                , intent(in)    :: read_restart

    call t_startf('EOCN_INIT')
    call samudra_comp_init(EClock, ggrid, read_restart)
    call t_stopf('EOCN_INIT')

  end subroutine eocn_comp_init

  !===============================================================================
  subroutine eocn_comp_run(EClock, x2o, o2x, gsmap, ggrid)

    implicit none

    type(ESMF_Clock)       , intent(in)    :: EClock
    type(mct_aVect)        , intent(inout) :: x2o
    type(mct_aVect)        , intent(inout) :: o2x
    type(mct_gsMap)        , pointer       :: gsMap
    type(mct_gGrid)        , pointer       :: ggrid

    integer(IN)   :: CurrentTOD, yy, mm, dd, nu, stepno
    logical       :: write_restart
    character(len=18) :: date_str
    character(*), parameter :: subName = "(eocn_comp_run) "

    call t_startf('EOCN_RUN')

    call seq_timemgr_EClockGetData( EClock, curr_tod=CurrentTOD)
    call seq_timemgr_EClockGetData( EClock, curr_yr=yy, curr_mon=mm, curr_day=dd)
    call seq_timemgr_EClockGetData( EClock, stepno=stepno)
    write_restart = seq_timemgr_RestartAlarmIsOn(EClock)

    call t_barrierf('eocn_BARRIER',mpicom_ocn)
    call t_startf('eocn_datamode')
    call samudra_comp_run(EClock, ggrid)
    call t_stopf('eocn_datamode')

    if (write_restart) then
       call t_startf('eocn_restart')
       call shr_cal_ymdtod2string(date_str, yy,mm,dd,currentTOD)

       write(restart_file,"(6a)") &
            trim(case_name), '.eocn',trim(inst_suffix),'.r.', trim(date_str), '.nc'
       if (masterproc) then
          nu = shr_file_getUnit()
          open(nu,file=trim(rpfile)//trim(inst_suffix),form='formatted')
          write(nu,'(a)') restart_file
          close(nu)
          call shr_file_freeUnit(nu)
          call eocn_restart_file_write(restart_file, date_str, stepno)
       endif
       call shr_sys_flush(logunit_ocn)
       call t_stopf('eocn_restart')
    endif

    call t_stopf('EOCN_RUN')

  end subroutine eocn_comp_run

  !===============================================================================
  subroutine eocn_comp_final()

    implicit none
    character(*), parameter :: F00 = "('(eocn_comp_final) ',8a)"

    call t_startf('EOCN_FINAL')

    deallocate(acc_taux, acc_tauy, acc_prec, acc_snow, acc_flus, acc_fsus)
    deallocate(acc_flds, acc_fsds, acc_lhflx, acc_shflx)
    deallocate(so_t, so_s, so_u, so_v, so_ssh, so_dhdx, so_dhdy, so_ifrac)
    deallocate(cell_lat, cell_lon, ocn_mask)
    deallocate(net_inputs, net_inputs_nn, net_outputs)
    deallocate(eocn_intrp%t_im1, eocn_intrp%t_ip1)

    call samudra_comp_finalize()
    call eocn_channels_final()

    if (masterproc) write(logunit_ocn,F00) trim(myModelName),': end of main integration loop'

    call t_stopf('EOCN_FINAL')

  end subroutine eocn_comp_final

end module eocn_comp_mod
