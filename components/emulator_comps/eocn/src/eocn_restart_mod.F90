module eocn_restart_mod

  !-----------------------------------------------------------------------------
  ! Restart and initial-condition I/O for EOCN.
  !
  ! Three things have to survive a restart for the run to continue exactly:
  !
  !   * both bracketing emulator states (the `time` dimension has length 2 and
  !     indexes t_im1 / t_ip1), so the interpolation the coupler sees is the
  !     same one a continuous run would have produced;
  !   * the static boundary inputs EOCN persists rather than re-derives
  !     (LANDFRAC and the sea surface fraction);
  !   * the partially accumulated coupler fluxes.  The emulator step is 5 days
  !     and restarts do not have to land on one, so dropping the accumulators
  !     would restart the interval's mean from whatever fraction of it follows
  !     the restart.
  !-----------------------------------------------------------------------------

  use shr_kind_mod      , only : r8 => shr_kind_r8
  use shr_sys_mod       , only : shr_sys_abort
  use pio               , only : pio_offset_kind, pio_set_buffer_size_limit

  use eocnIO
  use eocnMod
  use eocn_channels_mod

  implicit none
  save

  public :: eocn_restart_file_write
  public :: eocn_restart_file_read
  public :: eocn_initial_condition_file_read

  private

  contains

    subroutine eocn_restart_file_write( file, rdate, stepno )

      implicit none
      character(len=*), intent(in) :: file
      character(len=*), intent(in) :: rdate
      integer, intent(in) :: stepno

      type(file_desc_t) :: ncid
      integer :: i
      integer(pio_offset_kind) :: prev_buffer_limit

      write(logunit_ocn,'(72a1)') ("-",i=1,60)
      write(logunit_ocn, *) 'restart_file_open: writing EOCN restart dataset '

      ! EOCN is serial, so it writes whole global fields with pio_put_var
      ! rather than a distributed darray.  Under pnetcdf those become buffered
      ! puts that are not flushed until the file is closed, and this restart
      ! holds both bracketing emulator states -- around 90 MB, comfortably over
      ! the default limit, which aborts the run with "Attached buffer is too
      ! small" after the science has already finished.  Serial netcdf has no
      ! such buffer, which is why the restart test in VERIFICATION.md section 3
      ! never caught it.
      !
      ! Raise the limit for the write and put it back afterwards, rather than
      ! leaving it raised for every other component sharing the library.
      call pio_set_buffer_size_limit(int(256*1024*1024, pio_offset_kind), &
                                     prev_limit=prev_buffer_limit)

      call ncd_pio_createfile(ncid, trim(file))
      call set_restart_file_dimensions(ncid)
      call eocn_restart(ncid, 'define')
      call ncd_enddef(ncid)

      call eocn_restart(ncid, 'write')
      call ncd_pio_closefile(ncid)

      call pio_set_buffer_size_limit(prev_buffer_limit)

      write(logunit_ocn, *) 'Successfully wrote out restart data at nstep = ',stepno
      write(logunit_ocn,'(72a1)') ("-",i=1,60)

    end subroutine eocn_restart_file_write

    subroutine eocn_restart_file_read( file )

      implicit none
      character(len=*), intent(in) :: file

      type(file_desc_t) :: ncid
      integer :: i

      write(logunit_ocn, *) 'Reading restart dataset ', trim(file)
      call ncd_pio_openfile(ncid, trim(file), 0)
      call eocn_restart(ncid, 'read')
      call ncd_pio_closefile(ncid)

      write(logunit_ocn, *) 'Successfully read restart data for restart run'
      write(logunit_ocn,'(72a1)') ("-",i=1,60)

    end subroutine eocn_restart_file_read

    subroutine eocn_initial_condition_file_read( file )

      implicit none
      character(len=*), intent(in) :: file

      type(file_desc_t) :: ncid
      integer :: i

      write(logunit_ocn, *) 'Reading initial condition dataset ', trim(file)
      call ncd_pio_openfile(ncid, trim(file), 0)
      call eocn_initial_condition(ncid)
      call ncd_pio_closefile(ncid)

      write(logunit_ocn, *) 'Successfully read initial condition data for startup run'
      write(logunit_ocn,'(72a1)') ("-",i=1,60)

    end subroutine eocn_initial_condition_file_read

    subroutine set_restart_file_dimensions( ncid )

      implicit none
      type(file_desc_t) :: ncid
      integer :: dimid

      ! 'time' of length 2 indexes the two emulator time levels
      ! (1 => t_im1, 2 => t_ip1); it is not a calendar axis.
      call ncd_defdim(ncid, 'lon', lsize_x, dimid)
      call ncd_defdim(ncid, 'lat', lsize_y, dimid)
      call ncd_defdim(ncid, 'time', 2, dimid)

    end subroutine set_restart_file_dimensions

    subroutine eocn_restart( ncid, flag )

      implicit none
      type(file_desc_t), intent(inout) :: ncid
      character(len=*) , intent(in)    :: flag

      integer :: c
      logical :: readvar
      character(len=eocn_nlen)  :: vname
      character(len=eocn_ulen)  :: uname
      character(len=eocn_llen)  :: lname
      character(len=*), parameter :: subname = '(eocn_restart) '

      do c = 1, n_output_channels

        vname = out_names(c)
        call eocn_channel_metadata(vname, lname, uname)

        if ( flag == 'define' ) then
          call ncd_defvar(ncid=ncid, varname=trim(vname), xtype=ncd_double, &
               dim1name='lon', dim2name='lat', dim3name='time', &
               long_name=trim(lname), units=trim(uname))
        elseif (flag == 'read' .or. flag == 'write') then
          call ncd_io(varname=trim(vname), data=eocn_intrp%t_im1(c, :, :), &
               ncid=ncid, flag=flag, nt=1, readvar=readvar)
          if (flag == 'read' .and. .not. readvar) call shr_sys_abort( &
               trim(subname)//' ERROR: restart is missing channel '//trim(vname))
          call ncd_io(varname=trim(vname), data=eocn_intrp%t_ip1(c, :, :), &
               ncid=ncid, flag=flag, nt=2, readvar=readvar)
        else
          call shr_sys_abort(trim(subname)//' ERROR: unknown flag '//trim(flag))
        endif
      enddo

      !--- static boundary inputs EOCN owns ---
      call restart_input_channel(ncid, flag, ix_in_landfrac)
      call restart_input_channel(ncid, flag, ix_in_ssfrac)

      !--- Samudra's own ocean mask; not an emulator channel, but the only
      !--- thing that says where its state means anything
      if ( flag == 'define' ) then
        call ncd_defvar(ncid=ncid, varname='mask_2d', xtype=ncd_double, &
             dim1name='lon', dim2name='lat', &
             long_name='ocean mask of the emulator', units='1')
      elseif (flag == 'read' .or. flag == 'write') then
        call ncd_io(varname='mask_2d', data=ocn_mask, ncid=ncid, flag=flag, &
             readvar=readvar)
        if (flag == 'read' .and. .not. readvar) call shr_sys_abort( &
             trim(subname)//' ERROR: restart is missing mask_2d')
      endif

      if ( flag == 'define' ) then
        call ncd_defvar(ncid=ncid, varname='mask_ocean_sea_ice_fraction', &
             xtype=ncd_double, dim1name='lon', dim2name='lat', &
             long_name='mask of the emulator sea ice channels', units='1')
      elseif (flag == 'read' .or. flag == 'write') then
        call ncd_io(varname='mask_ocean_sea_ice_fraction', data=ice_mask, &
             ncid=ncid, flag=flag, readvar=readvar)
        if (flag == 'read' .and. .not. readvar) ice_mask(:,:) = ocn_mask(:,:)
      endif

      !--- partially accumulated coupler fluxes ---
      call restart_accumulator(ncid, flag, 'acc_taux',  acc_taux)
      call restart_accumulator(ncid, flag, 'acc_tauy',  acc_tauy)
      call restart_accumulator(ncid, flag, 'acc_prec',  acc_prec)
      call restart_accumulator(ncid, flag, 'acc_snow',  acc_snow)
      call restart_accumulator(ncid, flag, 'acc_flus',  acc_flus)
      call restart_accumulator(ncid, flag, 'acc_fsus',  acc_fsus)
      call restart_accumulator(ncid, flag, 'acc_flds',  acc_flds)
      call restart_accumulator(ncid, flag, 'acc_fsds',  acc_fsds)
      call restart_accumulator(ncid, flag, 'acc_lhflx', acc_lhflx)
      call restart_accumulator(ncid, flag, 'acc_shflx', acc_shflx)

      !--- and the same window taken from the atmosphere emulator directly ---
      call restart_accumulator(ncid, flag, 'raw_taux',  raw_taux, required=.false.)
      call restart_accumulator(ncid, flag, 'raw_tauy',  raw_tauy, required=.false.)
      call restart_accumulator(ncid, flag, 'raw_prec',  raw_prec, required=.false.)
      call restart_accumulator(ncid, flag, 'raw_snow',  raw_snow, required=.false.)
      call restart_accumulator(ncid, flag, 'raw_flus',  raw_flus, required=.false.)
      call restart_accumulator(ncid, flag, 'raw_fsus',  raw_fsus, required=.false.)
      call restart_accumulator(ncid, flag, 'raw_flds',  raw_flds, required=.false.)
      call restart_accumulator(ncid, flag, 'raw_fsds',  raw_fsds, required=.false.)
      call restart_accumulator(ncid, flag, 'raw_lhflx', raw_lhflx, required=.false.)
      call restart_accumulator(ncid, flag, 'raw_shflx', raw_shflx, required=.false.)

      if ( flag == 'define' ) then
        call ncd_defvar(ncid=ncid, varname='acc_n', xtype=ncd_int, &
             long_name='coupling steps accumulated into acc_*', units='1')
        call ncd_defvar(ncid=ncid, varname='eocn_elapsed', xtype=ncd_int, &
             long_name='seconds since the emulator last advanced', units='s')
        call ncd_defvar(ncid=ncid, varname='raw_n', xtype=ncd_int, &
             long_name='coupling steps accumulated into raw_*', units='1')
      elseif (flag == 'read' .or. flag == 'write') then
        call ncd_io(varname='acc_n', data=acc_n, ncid=ncid, flag=flag, &
             readvar=readvar)
        if (flag == 'read' .and. .not. readvar) call shr_sys_abort( &
             trim(subname)//' ERROR: restart is missing acc_n')
        call ncd_io(varname='eocn_elapsed', data=eocn_elapsed, ncid=ncid, &
             flag=flag, readvar=readvar)
        if (flag == 'read' .and. .not. readvar) call shr_sys_abort( &
             trim(subname)//' ERROR: restart is missing eocn_elapsed')
        ! Not fatal when missing: a restart written before the atmosphere's raw
        ! forcing channels existed is still a valid restart of the coupler path.
        call ncd_io(varname='raw_n', data=raw_n, ncid=ncid, flag=flag, &
             readvar=readvar)
        if (flag == 'read' .and. .not. readvar) raw_n = 0
      endif

    end subroutine eocn_restart

    subroutine restart_input_channel( ncid, flag, ix )

      implicit none
      type(file_desc_t), intent(inout) :: ncid
      character(len=*) , intent(in)    :: flag
      integer          , intent(in)    :: ix

      logical :: readvar
      character(len=eocn_nlen) :: vname
      character(len=eocn_ulen) :: uname
      character(len=eocn_llen) :: lname
      character(len=*), parameter :: subname = '(restart_input_channel) '

      if (ix <= 0) return

      vname = in_names(ix)
      call eocn_channel_metadata(vname, lname, uname)

      if ( flag == 'define' ) then
        call ncd_defvar(ncid=ncid, varname=trim(vname), xtype=ncd_double, &
             dim1name='lon', dim2name='lat', &
             long_name=trim(lname), units=trim(uname))
      elseif (flag == 'read' .or. flag == 'write') then
        call ncd_io(varname=trim(vname), data=net_inputs(1, ix, :, :), &
             ncid=ncid, flag=flag, readvar=readvar)
        if (flag == 'read' .and. .not. readvar) call shr_sys_abort( &
             trim(subname)//' ERROR: restart is missing '//trim(vname))
      endif

    end subroutine restart_input_channel

    subroutine restart_accumulator( ncid, flag, vname, field, required )

      implicit none
      type(file_desc_t), intent(inout) :: ncid
      character(len=*) , intent(in)    :: flag
      character(len=*) , intent(in)    :: vname
      real(r8)         , intent(inout) :: field(:,:)
      logical, optional, intent(in)    :: required

      logical :: readvar, must
      character(len=*), parameter :: subname = '(restart_accumulator) '

      must = .true.
      if (present(required)) must = required

      if ( flag == 'define' ) then
        call ncd_defvar(ncid=ncid, varname=trim(vname), xtype=ncd_double, &
             dim1name='lon', dim2name='lat', &
             long_name='running sum of an imported coupler flux', units='1')
      elseif (flag == 'read' .or. flag == 'write') then
        call ncd_io(varname=trim(vname), data=field, ncid=ncid, flag=flag, &
             readvar=readvar)
        if (flag == 'read' .and. .not. readvar) then
          if (must) call shr_sys_abort( &
               trim(subname)//' ERROR: restart is missing '//trim(vname))
          field(:,:) = 0.0_r8
        end if
      endif

    end subroutine restart_accumulator

    subroutine eocn_initial_condition( ncid )


      !-----------------------------------------------------------------------
      ! Populate the emulator's input state from an initial-condition file --
      ! one 2D field per input channel, named exactly as the channel is named
      ! in the checkpoint.
      !
      ! The published SamudrACE ocean initial conditions have a time dimension
      ! of length 3 (three initial conditions); the first record is read.

      implicit none
      type(file_desc_t), intent(inout) :: ncid

      integer :: c, varid, ndims
      logical :: readvar
      type(var_desc_t) :: vardesc
      character(len=*), parameter :: subname = '(eocn_initial_condition) '

      call ncd_io(varname='mask_2d', data=ocn_mask, ncid=ncid, flag='read', &
           readvar=readvar)
      if (.not. readvar) call shr_sys_abort(trim(subname)// &
           ' ERROR: initial condition file has no variable mask_2d; rebuild it '// &
           'with tools/make_eocn_input.py')

      ! The sea ice channels are masked more tightly than the ocean.  An
      ! initial condition built before this mask was carried has only mask_2d;
      ! fall back to it, which is the behaviour that put sea ice in the
      ! tropics, and say so rather than doing it silently.
      call ncd_io(varname='mask_ocean_sea_ice_fraction', data=ice_mask, &
           ncid=ncid, flag='read', readvar=readvar)
      if (.not. readvar) then
        ice_mask(:,:) = ocn_mask(:,:)
        write(logunit_ocn,'(a)') trim(subname)//' WARNING: initial condition '// &
             'has no mask_ocean_sea_ice_fraction; the emulator sea ice fraction '// &
             'will be exported wherever there is ocean, including where the '// &
             'network was never trained to predict it.  Rebuild the initial '// &
             'condition with tools/make_eocn_input.py.'
      end if

      do c = 1, n_input_channels
        call ncd_inqvid(ncid, trim(in_names(c)), varid, vardesc, readvar=readvar)
        if (.not. readvar) call shr_sys_abort(trim(subname)// &
             ' ERROR: initial condition file has no variable '//trim(in_names(c)))
        call ncd_inqvdims(ncid, ndims, vardesc)
        if (ndims == 3) then
          ! the published SamudrACE ocean IC bundles three initial conditions
          ! on a leading record dimension; take the first
          call ncd_io(varname=trim(in_names(c)), data=net_inputs(1, c, :, :), &
               ncid=ncid, flag='read', nt=1, readvar=readvar)
        else
          call ncd_io(varname=trim(in_names(c)), data=net_inputs(1, c, :, :), &
               ncid=ncid, flag='read', readvar=readvar)
        end if
      enddo

    end subroutine eocn_initial_condition

end module eocn_restart_mod
