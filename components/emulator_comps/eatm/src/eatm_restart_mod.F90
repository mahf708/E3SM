module eatm_restart_mod

  !-----------------------------------------------------------------------------
  ! Restart and initial-condition I/O for EATM.
  !
  ! The restart holds both bracketing emulator states (the `time` dimension has
  ! length 2 and indexes t_im1 / t_ip1) plus the boundary inputs EATM persists
  ! rather than re-derives -- currently PHIS and the last SOLIN.  Writing both
  ! time levels is what makes a restart reproduce the interpolated state the
  ! coupler would have seen in a continuous run.
  !
  ! Variable names come from eatm_channels_mod, so this file does not need to
  ! change when a different emulator with a different channel layout is used.
  !-----------------------------------------------------------------------------

  use shr_kind_mod      , only : r8 => shr_kind_r8
  use shr_sys_mod       , only : shr_sys_abort

  use eatmIO
  use eatmMod
  use eatm_channels_mod

  implicit none
  save

  public :: eatm_restart_file_write
  public :: eatm_restart_file_read
  public :: eatm_initial_condition_file_read

  private

  contains
    subroutine eatm_restart_file_write( file, rdate, stepno )

      !----------------------------------------------------------------
      ! !DESCRIPTION:
      ! Create the restart file and write both emulator time levels to it.
      implicit none
      ! !ARGUMENTS
      character(len=*), intent(in) :: file
      character(len=*), intent(in) :: rdate
      integer, intent(in) :: stepno

      ! !LOCAL VARIABLES:
      type(file_desc_t) :: ncid ! netcdf id
      integer :: i ! index
      !----------------------------------------------------------------

      write(logunit_atm,'(72a1)') ("-",i=1,60)
      write(logunit_atm, *) 'restart_file_open: writing EATM restart dataset '
      write(logunit_atm, *)

      ! Define dimensions and variables
      call ncd_pio_createfile(ncid, trim(file))
      call set_restart_file_dimensions(ncid)
      call eatm_restart(ncid, 'define')
      call ncd_enddef(ncid)

      ! Write restart file variables
      call eatm_restart(ncid, 'write')
      call ncd_pio_closefile(ncid)

      write(logunit_atm, *)
      write(logunit_atm, *) 'Successfully wrote out restart data at nstep = ',stepno
      write(logunit_atm,'(72a1)') ("-",i=1,60)

    end subroutine eatm_restart_file_write

    subroutine eatm_restart_file_read( file )
      !----------------------------------------------------------------
      ! !DESCRIPTION:
      ! Read both emulator time levels and the persisted boundary inputs.
      implicit none
      ! !ARGUMENTS
      character(len=*), intent(in) :: file

      ! !LOCAL VARIABLES:
      type(file_desc_t) :: ncid ! netcdf id
      integer :: i ! index
      !----------------------------------------------------------------

      write(logunit_atm, *) 'Reading restart dataset'
      call ncd_pio_openfile(ncid, trim(file), 0)
      call eatm_restart(ncid, 'read')
      call ncd_pio_closefile(ncid)

      write(logunit_atm, *)
      write(logunit_atm, *) 'Successfully read restart data for restart run'
      write(logunit_atm,'(72a1)') ("-",i=1,60)

    end subroutine eatm_restart_file_read

    subroutine eatm_initial_condition_file_read( file )
      !----------------------------------------------------------------
      ! !DESCRIPTION:
      ! Populate the emulator's input state from an initial-condition file.
      implicit none
      ! !ARGUMENTS
      character(len=*), intent(in) :: file

      ! !LOCAL VARIABLES:
      type(file_desc_t) :: ncid ! netcdf id
      integer :: i ! index
      !----------------------------------------------------------------

      write(logunit_atm, *) 'Reading initial condition dataset ', trim(file)
      call ncd_pio_openfile(ncid, trim(file), 0)
      call eatm_initial_condition(ncid)
      call ncd_pio_closefile(ncid)

      write(logunit_atm, *)
      write(logunit_atm, *) 'Successfully read initial condition data for startup run'
      write(logunit_atm,'(72a1)') ("-",i=1,60)

    end subroutine eatm_initial_condition_file_read

    subroutine set_restart_file_dimensions( ncid )

      !----------------------------------------------------------------
      ! !DESCRIPTION:
      ! ...
      implicit none
      ! !ARGUMENTS
      type(file_desc_t) :: ncid ! netcdf id

      ! !LOCAL VARIABLES:
      integer :: dimid               ! netCDF dimension id
      !----------------------------------------------------------------

      ! Define dimensions.  'time' of length 2 indexes the two emulator time
      ! levels (1 => t_im1, 2 => t_ip1), it is not a calendar axis.
      call ncd_defdim(ncid, 'lon', lsize_x, dimid)
      call ncd_defdim(ncid, 'lat', lsize_y, dimid)
      call ncd_defdim(ncid, 'time', 2, dimid)

    end subroutine set_restart_file_dimensions

    subroutine eatm_restart( ncid, flag )
      !-----------------------------------------------------------------------
      ! DESCRIPTION:
      ! define/read/write eatm restart data.

      ! ARGUMENTS:
      implicit none
      type(file_desc_t), intent(inout) :: ncid ! netcdf id
      character(len=*) , intent(in)    :: flag ! 'define' or 'read' or 'write'

      ! LOCAL VARIABLES:
      integer :: c ! channel index
      logical :: readvar ! determine if variable is read
      character(len=eatm_nlen)  :: vname
      character(len=eatm_ulen)  :: uname
      character(len=eatm_llen)  :: lname
      character(len=*), parameter :: subname = '(eatm_restart) '

      do c = 1, n_output_channels

        vname = out_names(c)
        call eatm_channel_metadata(vname, lname, uname)

        if ( flag == 'define' ) then
          call ncd_defvar(&
            ncid=ncid, &
            varname=trim(vname), &
            xtype=ncd_double, &
            dim1name='lon', &
            dim2name='lat', &
            dim3name='time', &
            long_name=trim(lname), &
            units=trim(uname) &
        )
        elseif (flag == 'read' .or. flag == 'write') then
          call ncd_io(&
            varname=trim(vname), &
            data=eatm_intrp%t_im1(c, :, :), &
            ncid=ncid, &
            flag=flag, &
            nt=1, &
            readvar=readvar &
          )
          if (flag == 'read' .and. .not. readvar) call shr_sys_abort( &
               trim(subname)//' ERROR: restart is missing channel '//trim(vname))
          call ncd_io(&
            varname=trim(vname), &
            data=eatm_intrp%t_ip1(c, :, :), &
            ncid=ncid, &
            flag=flag, &
            nt=2, &
            readvar=readvar &
          )
        else
          call shr_sys_abort(trim(subname)//' ERROR: unknown flag '//trim(flag))
        endif
      enddo

      !--- boundary inputs EATM owns and must carry across a restart ---
      call restart_input_channel(ncid, flag, ix_in_phis)
      call restart_input_channel(ncid, flag, ix_in_solin)

    end subroutine eatm_restart

    subroutine restart_input_channel( ncid, flag, ix )
      !-----------------------------------------------------------------------
      ! define/read/write a single (time-invariant) emulator input channel.
      implicit none
      type(file_desc_t), intent(inout) :: ncid
      character(len=*) , intent(in)    :: flag
      integer          , intent(in)    :: ix

      logical :: readvar
      character(len=eatm_nlen) :: vname
      character(len=eatm_ulen) :: uname
      character(len=eatm_llen) :: lname
      character(len=*), parameter :: subname = '(restart_input_channel) '

      if (ix <= 0) return

      vname = in_names(ix)
      call eatm_channel_metadata(vname, lname, uname)

      if ( flag == 'define' ) then
        call ncd_defvar(&
          ncid=ncid, &
          varname=trim(vname), &
          xtype=ncd_double, &
          dim1name='lon', &
          dim2name='lat', &
          long_name=trim(lname), &
          units=trim(uname) &
        )
      elseif (flag == 'read' .or. flag == 'write') then
        call ncd_io(&
          varname=trim(vname), &
          data=net_inputs(1, ix, :, :), &
          ncid=ncid, &
          flag=flag, &
          readvar=readvar &
        )
        if (flag == 'read' .and. .not. readvar) call shr_sys_abort( &
             trim(subname)//' ERROR: restart is missing '//trim(vname))
      endif

    end subroutine restart_input_channel

    subroutine eatm_initial_condition( ncid )
      !-----------------------------------------------------------------------
      ! DESCRIPTION:
      ! read eatm initial condition data -- one 2D field per emulator input
      ! channel, named exactly as the channel is named in the checkpoint.

      ! ARGUMENTS:
      implicit none
      type(file_desc_t), intent(inout) :: ncid ! netcdf id

      ! LOCAL VARIABLES:
      integer :: c ! channel index
      logical :: readvar ! determine if variable is read
      character(len=*), parameter :: subname = '(eatm_initial_condition) '

      do c = 1, n_input_channels
        call ncd_io(&
          varname=trim(in_names(c)), &
          data=net_inputs(1, c, :, :), &
          ncid=ncid, &
          flag='read', &
          readvar=readvar &
        )
        if (.not. readvar) call shr_sys_abort(trim(subname)// &
             ' ERROR: initial condition file has no variable '//trim(in_names(c)))
      enddo

    end subroutine eatm_initial_condition

end module eatm_restart_mod
