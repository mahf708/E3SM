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
  !   call rtm_fme_derived_register()       ! during init
  !   call rtm_fme_derived_update()         ! during driver timestep
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,    only: r8 => shr_kind_r8
  use shr_derived_mod, only: shr_derived_expr_t, shr_derived_parse, &
                              shr_derived_max_namelen, shr_derived_max_deflen
  use RtmVar,          only: iulog
  use RtmSpmd,         only: masterproc, mpicom, MPI_REAL8, MPI_CHARACTER
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
        read(unitn, rtm_fme_derived_nl, iostat=ierr)
        close(unitn)
        call relavu(unitn)
      end if
    end if

    call mpi_bcast(rtm_derived_fld_defs, max_def_len*max_derived_flds, &
         MPI_CHARACTER, 0, mpicom, ierr)

    n_derived_flds = 0
    do i = 1, max_derived_flds
      if (len_trim(rtm_derived_fld_defs(i)) > 0) then
        n_derived_flds = n_derived_flds + 1
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
    integer :: i, ierr, cnt

    if (.not. has_derived) return

    cnt = 0
    do i = 1, max_derived_flds
      if (len_trim(rtm_derived_fld_defs(i)) == 0) cycle
      cnt = cnt + 1
      call shr_derived_parse(trim(rtm_derived_fld_defs(i)), parsed_exprs(cnt), ierr)
      if (ierr /= 0) then
        call shr_sys_abort('rtm_fme_derived_register: failed to parse: ' &
             // trim(rtm_derived_fld_defs(i)))
      end if
      if (masterproc) then
        write(iulog, *) '  FME derived: ', trim(parsed_exprs(cnt)%output_name), &
             ' = ', trim(parsed_exprs(cnt)%long_name)
      end if
    end do

  end subroutine rtm_fme_derived_register

  !============================================================================
  subroutine rtm_fme_derived_update()
    ! Placeholder for runtime evaluation
    if (.not. has_derived) return
  end subroutine rtm_fme_derived_update

end module RtmFmeDerived
