module elm_fme_derived
  !-------------------------------------------------------------------------------------------
  !
  ! ELM wrapper for FME (Full Model Emulation) derived field processing.
  !
  ! Provides derived field combinations using the shared shr_derived_mod
  ! expression parser and evaluator. Supports user-defined expressions
  ! combining ELM history fields and numeric constants.
  !
  ! Configuration via namelist (elm_fme_derived_nl):
  !   elm_derived_fld_defs - expression definitions, e.g. "TOTAL_SOIL_WATER=H2OSOI+SOILICE"
  !
  ! Usage from controlMod / elm_driver:
  !   call elm_fme_derived_readnl(nlfile)     ! during namelist reading
  !   call elm_fme_derived_register()         ! during init, after other addfld calls
  !   call elm_fme_derived_update(bounds)     ! during driver timestep
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,    only: r8 => shr_kind_r8
  use shr_derived_mod, only: shr_derived_expr_t, shr_derived_parse, &
                              shr_derived_max_namelen, shr_derived_max_deflen
  use elm_varctl,      only: iulog
  use abortutils,      only: endrun
  use spmdMod,         only: masterproc, mpicom, MPI_REAL8, MPI_INTEGER, MPI_CHARACTER

  implicit none
  private
  save

  public :: elm_fme_derived_readnl
  public :: elm_fme_derived_register
  public :: elm_fme_derived_update

  ! Parameters
  integer, parameter :: max_derived_flds = 50
  integer, parameter :: max_name_len     = shr_derived_max_namelen
  integer, parameter :: max_def_len      = shr_derived_max_deflen

  ! Namelist variables
  character(len=max_def_len) :: elm_derived_fld_defs(max_derived_flds)

  ! Parsed state
  integer :: n_derived_flds = 0
  type(shr_derived_expr_t) :: parsed_exprs(max_derived_flds)

  logical :: module_is_initialized = .false.
  logical :: has_derived = .false.

contains

  !============================================================================
  subroutine elm_fme_derived_readnl(nlfile)
    use shr_nl_mod,  only: shr_nl_find_group_name
    use fileutils,   only: getavu, relavu

    character(len=*), intent(in) :: nlfile

    integer :: unitn, ierr, i
    logical :: lexist

    namelist /elm_fme_derived_nl/ elm_derived_fld_defs

    elm_derived_fld_defs(:) = ''

    if (masterproc) then
      inquire(file=trim(nlfile), exist=lexist)
      if (lexist) then
        unitn = getavu()
        open(unitn, file=trim(nlfile), status='old')
        call shr_nl_find_group_name(unitn, 'elm_fme_derived_nl', status=ierr)
        if (ierr == 0) then
          read(unitn, elm_fme_derived_nl, iostat=ierr)
          if (ierr /= 0) then
            call endrun('elm_fme_derived_readnl: ERROR reading namelist')
          end if
        end if
        close(unitn)
        call relavu(unitn)
      end if
    end if

    call mpi_bcast(elm_derived_fld_defs, max_def_len*max_derived_flds, &
         MPI_CHARACTER, 0, mpicom, ierr)

    ! Count active definitions
    n_derived_flds = 0
    do i = 1, max_derived_flds
      if (len_trim(elm_derived_fld_defs(i)) > 0) then
        n_derived_flds = n_derived_flds + 1
      end if
    end do

    has_derived = (n_derived_flds > 0)

    if (masterproc .and. has_derived) then
      write(iulog, *) 'elm_fme_derived: ', n_derived_flds, ' derived field definitions'
    end if

    module_is_initialized = .true.

  end subroutine elm_fme_derived_readnl

  !============================================================================
  subroutine elm_fme_derived_register()
    ! Parse expressions (for validation). Actual field registration
    ! would require integration with histFileMod which is deferred
    ! until full ELM integration is completed.
    integer :: i, ierr, cnt

    if (.not. has_derived) return

    cnt = 0
    do i = 1, max_derived_flds
      if (len_trim(elm_derived_fld_defs(i)) == 0) cycle
      cnt = cnt + 1
      call shr_derived_parse(trim(elm_derived_fld_defs(i)), parsed_exprs(cnt), ierr)
      if (ierr /= 0) then
        call endrun('elm_fme_derived_register: failed to parse: ' &
             // trim(elm_derived_fld_defs(i)))
      end if
      if (masterproc) then
        write(iulog, *) '  FME derived: ', trim(parsed_exprs(cnt)%output_name), &
             ' = ', trim(parsed_exprs(cnt)%long_name)
      end if
    end do

  end subroutine elm_fme_derived_register

  !============================================================================
  subroutine elm_fme_derived_update()
    ! Placeholder for runtime evaluation.
    ! Full integration with ELM histFileMod requires field data access
    ! via ELM's internal data structures, which will be implemented
    ! when the ELM component wrapper is fully connected.

    if (.not. has_derived) return

    ! Evaluation deferred to full integration

  end subroutine elm_fme_derived_update

end module elm_fme_derived
