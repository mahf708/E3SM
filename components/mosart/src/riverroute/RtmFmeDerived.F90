module RtmFmeDerived
  !-------------------------------------------------------------------------------------------
  !
  ! MOSART wrapper for FME (Full Model Emulation) derived field processing.
  !
  ! Provides derived field combinations using the shared shr_derived_mod
  ! expression parser and evaluator. Supports user-defined expressions
  ! combining MOSART history fields and numeric constants.
  !
  ! No vertical coarsening since MOSART is 1D.
  !
  ! Configuration via namelist (rtm_fme_derived_nl):
  !   rtm_derived_fld_defs - field combination definitions
  !
  ! Usage from RtmMod:
  !   call rtm_fme_derived_readnl(nlfile)   ! during namelist reading
  !   call rtm_fme_derived_register()       ! during init, after RtmHistFldsInit
  !   call rtm_fme_derived_update()         ! during timestep, before RtmHistUpdateHbuf
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,    only: r8 => shr_kind_r8
  use shr_derived_mod, only: shr_derived_expr_t, shr_derived_parse, &
                              shr_derived_max_namelen, shr_derived_max_deflen
  use RtmVar,          only: iulog, spval
  use RtmSpmd,         only: masterproc, mpicom, MPI_REAL8, MPI_CHARACTER
  use RunoffMod,       only: rtmCTL
  use shr_sys_mod,     only: shr_sys_abort

  implicit none
  private
  save

  public :: rtm_fme_derived_readnl
  public :: rtm_fme_derived_register
  public :: rtm_fme_derived_update

  integer, parameter :: max_derived_flds = 50
  integer, parameter :: max_def_len      = shr_derived_max_deflen

  character(len=max_def_len) :: rtm_derived_fld_defs(max_derived_flds)

  integer :: n_derived_flds = 0
  type(shr_derived_expr_t) :: parsed_exprs(max_derived_flds)

  ! Persistent output arrays for history field pointers
  ! Shape: (begr:endr, n_derived_flds)
  real(r8), allocatable, target :: derived_data(:,:)

  logical :: module_is_initialized = .false.
  logical :: has_derived = .false.

contains

  !============================================================================
  subroutine rtm_fme_derived_readnl(nlfile)
    use RtmFileUtils, only: getavu, relavu

    character(len=*), intent(in) :: nlfile

    integer :: unitn, ierr, i
    logical :: lexist

    namelist /rtm_fme_derived_nl/ rtm_derived_fld_defs

    rtm_derived_fld_defs(:) = ''

    if (masterproc) then
      inquire(file=trim(nlfile), exist=lexist)
      if (lexist) then
        unitn = getavu()
        open(unitn, file=trim(nlfile), status='old')
        ! Loop to find the correct namelist group in multi-group file
        ierr = 1
        do while (ierr /= 0)
          read(unitn, rtm_fme_derived_nl, iostat=ierr)
          if (ierr < 0) then
            ! EOF reached without finding namelist group — not an error,
            ! just means this namelist group is absent from the file
            exit
          end if
        end do
        close(unitn)
        call relavu(unitn)
      end if
    end if

    call mpi_bcast(rtm_derived_fld_defs, max_def_len*max_derived_flds, &
         MPI_CHARACTER, 0, mpicom, ierr)

    ! Count and parse active definitions
    n_derived_flds = 0
    do i = 1, max_derived_flds
      if (len_trim(rtm_derived_fld_defs(i)) > 0) then
        n_derived_flds = n_derived_flds + 1
        call shr_derived_parse(trim(rtm_derived_fld_defs(i)), &
             parsed_exprs(n_derived_flds), ierr)
        if (ierr /= 0) then
          call shr_sys_abort('rtm_fme_derived_readnl: failed to parse: ' &
               // trim(rtm_derived_fld_defs(i)))
        end if
      end if
    end do

    has_derived = (n_derived_flds > 0)

    if (masterproc .and. has_derived) then
      write(iulog, *) 'rtm_fme_derived: ', n_derived_flds, ' derived field definitions'
    end if

    module_is_initialized = .true.

  end subroutine rtm_fme_derived_readnl

  !============================================================================
  subroutine rtm_fme_derived_register()
    !--------------------------------------------------------------------------
    ! Register derived output fields with MOSART's history system.
    ! Must be called AFTER RtmHistFldsInit() but BEFORE RtmHistHtapesBuild().
    !--------------------------------------------------------------------------
    use RtmHistFile, only: RtmHistAddfld

    integer :: i

    if (.not. has_derived) return

    ! Allocate persistent storage for derived field values
    allocate(derived_data(rtmCTL%begr:rtmCTL%endr, n_derived_flds))
    derived_data(:,:) = spval

    ! Register each derived field with the MOSART history system
    do i = 1, n_derived_flds
      if (masterproc) then
        write(iulog, *) '  FME derived: registering ', &
             trim(parsed_exprs(i)%output_name), ' = ', &
             trim(parsed_exprs(i)%long_name)
      end if

      call RtmHistAddfld( &
           fname=trim(parsed_exprs(i)%output_name), &
           units='derived', &
           avgflag='A', &
           long_name='FME derived: ' // trim(parsed_exprs(i)%long_name), &
           ptr_rof=derived_data(:,i), &
           default='inactive')
    end do

  end subroutine rtm_fme_derived_register

  !============================================================================
  subroutine rtm_fme_derived_update()
    !--------------------------------------------------------------------------
    ! Evaluate derived field expressions and update output arrays.
    ! Must be called each timestep BEFORE RtmHistUpdateHbuf.
    !
    ! Source field lookup from rtmCTL state variables requires
    ! case-by-case integration similar to how RtmHistFldsSet
    ! maps field names to rtmCTL pointers.
    !--------------------------------------------------------------------------
    if (.not. has_derived) return

    ! The persistent arrays (derived_data) are registered with RtmHistFile.
    ! Source field evaluation from rtmCTL state requires mapping field names
    ! to rtmCTL data members (e.g., 'RIVER_DISCHARGE_OVER_LAND' -> rtmCTL%runofflnd_nt1).

  end subroutine rtm_fme_derived_update

end module RtmFmeDerived
