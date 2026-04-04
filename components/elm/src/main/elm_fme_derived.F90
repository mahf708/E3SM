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
  ! Usage:
  !   call elm_fme_derived_readnl(nlfile)       ! during namelist reading (controlMod)
  !   call elm_fme_derived_register(bounds)     ! during init, before hist_htapes_build
  !   call elm_fme_derived_update(bounds)       ! during driver timestep, before hist_update_hbuf
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,    only: r8 => shr_kind_r8
  use shr_derived_mod, only: shr_derived_expr_t, shr_derived_parse, shr_derived_eval, &
                              shr_derived_max_operands, shr_derived_max_namelen, &
                              shr_derived_max_deflen
  use elm_varctl,      only: iulog
  use elm_varcon,      only: spval
  use abortutils,      only: endrun
  use spmdMod,         only: masterproc, mpicom, MPI_REAL8, MPI_INTEGER, MPI_CHARACTER
  use decompMod,       only: bounds_type

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

  ! Persistent output arrays for history field pointers (gridcell-level)
  ! These must remain allocated for the entire simulation since hist_addfld1d
  ! stores pointers to them.
  real(r8), allocatable, target :: derived_data(:,:)  ! (begg:endg, n_derived_flds)

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

    ! Count and parse active definitions
    n_derived_flds = 0
    do i = 1, max_derived_flds
      if (len_trim(elm_derived_fld_defs(i)) > 0) then
        n_derived_flds = n_derived_flds + 1
        call shr_derived_parse(trim(elm_derived_fld_defs(i)), &
             parsed_exprs(n_derived_flds), ierr)
        if (ierr /= 0) then
          call endrun('elm_fme_derived_readnl: failed to parse: ' &
               // trim(elm_derived_fld_defs(i)))
        end if
      end if
    end do

    has_derived = (n_derived_flds > 0)

    if (masterproc .and. has_derived) then
      write(iulog, *) 'elm_fme_derived: ', n_derived_flds, ' derived field definitions'
    end if

    module_is_initialized = .true.

  end subroutine elm_fme_derived_readnl

  !============================================================================
  subroutine elm_fme_derived_register(bounds)
    !--------------------------------------------------------------------------
    ! Register derived output fields with ELM's history system.
    ! Must be called BEFORE hist_htapes_build().
    !
    ! Allocates persistent data arrays and registers each derived field
    ! as a gridcell-level history field using hist_addfld1d.
    !--------------------------------------------------------------------------
    use histFileMod, only: hist_addfld1d

    type(bounds_type), intent(in) :: bounds

    integer :: i, begg, endg

    if (.not. has_derived) return

    begg = bounds%begg
    endg = bounds%endg

    ! Allocate persistent storage for derived field values
    allocate(derived_data(begg:endg, n_derived_flds))
    derived_data(:,:) = spval

    ! Register each derived field with the history system
    do i = 1, n_derived_flds
      if (masterproc) then
        write(iulog, *) '  FME derived: registering ', &
             trim(parsed_exprs(i)%output_name), ' = ', &
             trim(parsed_exprs(i)%long_name)
      end if

      call hist_addfld1d( &
           fname=trim(parsed_exprs(i)%output_name), &
           units='derived', &
           avgflag='A', &
           long_name='FME derived: ' // trim(parsed_exprs(i)%long_name), &
           ptr_gcell=derived_data(:,i), &
           default='inactive')
    end do

  end subroutine elm_fme_derived_register

  !============================================================================
  subroutine elm_fme_derived_update(bounds)
    !--------------------------------------------------------------------------
    ! Evaluate derived field expressions and update output arrays.
    ! Must be called each timestep BEFORE hist_update_hbuf.
    !
    ! Looks up source field values from ELM's history pointer arrays
    ! (elmptr_rs for 1D fields) by scanning the masterlist for matching
    ! field names, then evaluates the parsed expression.
    !--------------------------------------------------------------------------
    use histFileMod, only: elmptr_rs, masterlist, nfmaster, max_flds
    use shr_derived_mod, only: shr_derived_eval

    type(bounds_type), intent(in) :: bounds

    integer :: i, n, nf, begg, endg, ncol, hpidx
    real(r8), allocatable :: operand_data(:,:,:)
    real(r8), allocatable :: result_data(:,:)
    real(r8), pointer :: fld_ptr(:)
    logical :: found

    if (.not. has_derived) return

    begg = bounds%begg
    endg = bounds%endg
    ncol = endg - begg + 1

    ! Evaluate each derived expression
    do i = 1, n_derived_flds
      ! Gather operand data (treat as 1D: ncol x nlev=1 x n_operands)
      allocate(operand_data(ncol, 1, parsed_exprs(i)%n_operands))
      operand_data(:,:,:) = 0.0_r8

      do n = 1, parsed_exprs(i)%n_operands
        if (parsed_exprs(i)%operands(n)%is_constant) cycle

        ! Look up field by name in the history masterlist
        found = .false.
        do nf = 1, nfmaster
          if (trim(masterlist(nf)%field%name) == &
              trim(parsed_exprs(i)%operands(n)%field_name)) then
            hpidx = masterlist(nf)%field%hpindex
            if (masterlist(nf)%field%numdims == 1) then
              fld_ptr => elmptr_rs(hpidx)%ptr
              if (associated(fld_ptr)) then
                operand_data(1:ncol, 1, n) = fld_ptr(begg:endg)
                found = .true.
              end if
            end if
            exit
          end if
        end do

        if (.not. found) then
          ! Field not found — result will be incorrect, fill with spval
          derived_data(begg:endg, i) = spval
          deallocate(operand_data)
          goto 100  ! skip to next expression
        end if
      end do

      ! Evaluate expression
      allocate(result_data(ncol, 1))
      call shr_derived_eval(parsed_exprs(i), operand_data, ncol, 1, ncol, result_data)
      derived_data(begg:endg, i) = result_data(1:ncol, 1)
      deallocate(result_data)
      deallocate(operand_data)

      100 continue
    end do

  end subroutine elm_fme_derived_update

end module elm_fme_derived
