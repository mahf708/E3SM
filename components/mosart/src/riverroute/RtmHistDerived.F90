module RtmHistDerived
  !-------------------------------------------------------------------------------------------
  !
  ! Derived field combinations for MOSART history output.
  !
  ! Parses field definition strings like "OUTPUT=INPUT1+INPUT2-INPUT3"
  ! and computes them at each history output step.
  !
  ! No vertical coarsening since MOSART is 1D.
  !
  ! Configuration is via namelist (rtm_derived_fields_nl):
  !   rtm_derived_fld_defs - field combination definitions
  !
  ! Usage from RtmMod:
  !   call rtm_derived_fields_readnl(nlfile)   ! during namelist reading
  !   call rtm_derived_fields_register()       ! during init, after other addfld calls
  !   call rtm_derived_fields_update()         ! during driver timestep
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,  only: r8 => shr_kind_r8
  use RtmVar,        only: iulog, spval
  use RtmSpmd,       only: masterproc, mpicom_rof, MPI_REAL8, MPI_CHARACTER
  use RunoffMod,     only: rtmCTL
  use shr_sys_mod,   only: shr_sys_abort

  implicit none
  private
  save

  public :: rtm_derived_fields_readnl
  public :: rtm_derived_fields_register
  public :: rtm_derived_fields_update

  ! Parameters
  integer, parameter :: max_derived_flds   = 50    ! max derived field definitions
  integer, parameter :: max_derived_inputs = 20    ! max input fields per derived definition
  integer, parameter :: max_name_len       = 32    ! matches max_namlen in RtmHistFile
  integer, parameter :: max_def_len        = 256   ! max length of a derived field definition string

  ! Namelist variables
  character(len=max_def_len) :: rtm_derived_fld_defs(max_derived_flds)

  ! Parsed state: derived field combinations
  integer :: n_derived_flds = 0

  type :: derived_field_t
    character(len=max_name_len) :: output_name                        ! output field name
    integer                     :: n_inputs                            ! number of input fields
    character(len=max_name_len) :: input_names(max_derived_inputs)     ! input field names
    character(len=1)            :: operators(max_derived_inputs)       ! '+', '-', '*', '/' per input
    character(len=max_def_len)  :: units                               ! units string
    character(len=max_def_len)  :: long_name                           ! long name for NetCDF
    real(r8), pointer           :: output_ptr(:) => null()             ! pointer for history output
  end type derived_field_t

  type(derived_field_t) :: derived_flds(max_derived_flds)

  ! Flag to track initialization
  logical :: module_is_initialized = .false.
  logical :: has_derived  = .false.

contains

  !============================================================================
  subroutine rtm_derived_fields_readnl(nlfile)
    !--------------------------------------------------------------------------
    ! Read the rtm_derived_fields_nl namelist group
    !--------------------------------------------------------------------------
    use RtmFileUtils, only: getavu, relavu

    character(len=*), intent(in) :: nlfile

    integer :: unitn, ierr, i
    logical :: lexist

    namelist /rtm_derived_fields_nl/ rtm_derived_fld_defs

    ! Initialize defaults
    rtm_derived_fld_defs(:) = ''

    if (masterproc) then
      inquire(file=trim(nlfile), exist=lexist)
      if (.not. lexist) return  ! No namelist file; use defaults

      unitn = getavu()
      open(unitn, file=trim(nlfile), status='old', iostat=ierr)
      if (ierr /= 0) then
        call relavu(unitn)
        return
      end if
      ! Search for the namelist group; if not found, just use defaults
      ierr = 1
      do while (ierr /= 0)
        read(unitn, rtm_derived_fields_nl, iostat=ierr)
        if (ierr < 0) then
          ! End of file reached without finding namelist; use defaults
          rtm_derived_fld_defs(:) = ''
          exit
        end if
      end do
      close(unitn)
      call relavu(unitn)
    end if

    ! Broadcast to all processors
    call mpi_bcast(rtm_derived_fld_defs, max_def_len*max_derived_flds, MPI_CHARACTER, &
         0, mpicom_rof, ierr)

    ! Parse derived field definitions
    n_derived_flds = 0
    do i = 1, max_derived_flds
      if (len_trim(rtm_derived_fld_defs(i)) == 0) exit
      n_derived_flds = n_derived_flds + 1
      call parse_derived_def(rtm_derived_fld_defs(i), derived_flds(n_derived_flds))
    end do
    has_derived = (n_derived_flds > 0)

    if (masterproc) then
      if (has_derived) then
        write(iulog,*) 'rtm_derived_fields_readnl: ', n_derived_flds, ' derived field(s) defined'
        do i = 1, n_derived_flds
          write(iulog,*) '  ', trim(derived_flds(i)%output_name), ' = ', &
               trim(derived_flds(i)%long_name)
        end do
      end if
    end if

  end subroutine rtm_derived_fields_readnl

  !============================================================================
  subroutine parse_derived_def(defstr, dfld)
    !--------------------------------------------------------------------------
    ! Parse a derived field definition string of the form:
    !   "OUTPUT_NAME=INPUT1+INPUT2-INPUT3"
    ! Supported operators between fields: +, -, *, /
    ! The first input is always added (implicit +).
    !--------------------------------------------------------------------------
    character(len=*), intent(in)       :: defstr
    type(derived_field_t), intent(out) :: dfld

    integer :: eq_pos, op_pos, start_pos, n, i
    character(len=max_def_len) :: rhs
    character(len=1) :: ch

    dfld%n_inputs = 0
    dfld%units = 'mixed'
    dfld%operators(:) = '+'
    nullify(dfld%output_ptr)

    ! Find '=' separator
    eq_pos = index(defstr, '=')
    if (eq_pos < 2) then
      call shr_sys_abort('parse_derived_def: invalid definition (no "="), got: '//trim(defstr))
    end if

    dfld%output_name = adjustl(defstr(1:eq_pos-1))
    rhs = adjustl(defstr(eq_pos+1:))

    ! Parse operator-separated input names
    n = 0
    start_pos = 1
    do
      ! Find next operator (+, -, *, /)
      op_pos = 0
      do i = start_pos, len_trim(rhs)
        ch = rhs(i:i)
        if (ch == '+' .or. ch == '-' .or. ch == '*' .or. ch == '/') then
          op_pos = i
          exit
        end if
      end do

      n = n + 1
      if (n > max_derived_inputs) then
        call shr_sys_abort('parse_derived_def: too many inputs in definition: '//trim(defstr))
      end if

      if (n == 1) then
        dfld%operators(n) = '+'  ! first input is always added
      end if

      if (op_pos > 0) then
        dfld%input_names(n) = adjustl(rhs(start_pos:op_pos-1))
        ! The operator belongs to the NEXT input
        if (n < max_derived_inputs) then
          dfld%operators(n+1) = rhs(op_pos:op_pos)
        end if
        start_pos = op_pos + 1
      else
        dfld%input_names(n) = adjustl(rhs(start_pos:))
        exit
      end if
    end do
    dfld%n_inputs = n

    ! Build long name describing the expression
    dfld%long_name = trim(dfld%input_names(1))
    do n = 2, dfld%n_inputs
      dfld%long_name = trim(dfld%long_name) // ' ' // dfld%operators(n) // ' ' // &
           trim(dfld%input_names(n))
    end do

  end subroutine parse_derived_def

  !============================================================================
  subroutine rtm_derived_fields_register()
    !--------------------------------------------------------------------------
    ! Register derived output fields with the history system via RtmHistAddfld.
    ! Must be called during init, after other addfld calls.
    !--------------------------------------------------------------------------
    use RtmHistFile, only: RtmHistAddfld

    integer :: i
    integer :: begr, endr

    if (.not. has_derived) return

    begr = rtmCTL%begr
    endr = rtmCTL%endr

    ! Allocate output arrays and register each derived field
    do i = 1, n_derived_flds
      allocate(derived_flds(i)%output_ptr(begr:endr))
      derived_flds(i)%output_ptr(:) = 0.0_r8

      call RtmHistAddfld( &
           fname=trim(derived_flds(i)%output_name), &
           units=trim(derived_flds(i)%units), &
           avgflag='A', &
           long_name=trim(derived_flds(i)%long_name), &
           ptr_rof=derived_flds(i)%output_ptr, &
           default='inactive')
    end do

    module_is_initialized = .true.

    if (masterproc) then
      write(iulog,*) 'rtm_derived_fields_register: registration complete'
    end if

  end subroutine rtm_derived_fields_register

  !============================================================================
  subroutine rtm_derived_fields_update()
    !--------------------------------------------------------------------------
    ! Compute derived fields for the current timestep.
    ! Source fields are looked up via rtmCTL pointers by name.
    ! Called each timestep from the driver.
    !--------------------------------------------------------------------------
    integer  :: i, n, r
    integer  :: begr, endr
    real(r8), pointer :: src_ptr(:)

    if (.not. has_derived) return

    begr = rtmCTL%begr
    endr = rtmCTL%endr

    do i = 1, n_derived_flds
      derived_flds(i)%output_ptr(begr:endr) = 0.0_r8

      do n = 1, derived_flds(i)%n_inputs
        call get_rtm_field_ptr(derived_flds(i)%input_names(n), src_ptr)

        if (.not. associated(src_ptr)) then
          ! Field not found; fill with spval and skip
          derived_flds(i)%output_ptr(begr:endr) = spval
          exit
        end if

        select case (derived_flds(i)%operators(n))
        case ('+')
          do r = begr, endr
            derived_flds(i)%output_ptr(r) = derived_flds(i)%output_ptr(r) + src_ptr(r)
          end do
        case ('-')
          do r = begr, endr
            derived_flds(i)%output_ptr(r) = derived_flds(i)%output_ptr(r) - src_ptr(r)
          end do
        case ('*')
          if (n == 1) then
            do r = begr, endr
              derived_flds(i)%output_ptr(r) = src_ptr(r)
            end do
          else
            do r = begr, endr
              derived_flds(i)%output_ptr(r) = derived_flds(i)%output_ptr(r) * src_ptr(r)
            end do
          end if
        case ('/')
          if (n == 1) then
            do r = begr, endr
              derived_flds(i)%output_ptr(r) = src_ptr(r)
            end do
          else
            do r = begr, endr
              if (src_ptr(r) /= 0.0_r8) then
                derived_flds(i)%output_ptr(r) = derived_flds(i)%output_ptr(r) / src_ptr(r)
              else
                derived_flds(i)%output_ptr(r) = 0.0_r8
              end if
            end do
          end if
        end select
      end do
    end do

  end subroutine rtm_derived_fields_update

  !============================================================================
  subroutine get_rtm_field_ptr(fname, ptr)
    !--------------------------------------------------------------------------
    ! Look up a MOSART field name and return a pointer to the corresponding
    ! rtmCTL data array. This supports the standard MOSART output field names.
    !--------------------------------------------------------------------------
    use rof_cpl_indices, only: nt_rtm, rtm_tracers

    character(len=*), intent(in)  :: fname
    real(r8), pointer, intent(out) :: ptr

    character(len=max_name_len) :: uname

    uname = adjustl(fname)
    nullify(ptr)

    ! Match against known MOSART field names
    ! River discharge over land
    if (trim(uname) == 'RIVER_DISCHARGE_OVER_LAND_'//trim(rtm_tracers(1))) then
      ptr => rtmCTL%runofflnd_nt1
      return
    end if
    if (trim(uname) == 'RIVER_DISCHARGE_OVER_LAND_'//trim(rtm_tracers(2))) then
      ptr => rtmCTL%runofflnd_nt2
      return
    end if

    ! River discharge to ocean
    if (trim(uname) == 'RIVER_DISCHARGE_TO_OCEAN_'//trim(rtm_tracers(1))) then
      ptr => rtmCTL%runoffocn_nt1
      return
    end if
    if (trim(uname) == 'RIVER_DISCHARGE_TO_OCEAN_'//trim(rtm_tracers(2))) then
      ptr => rtmCTL%runoffocn_nt2
      return
    end if

    ! Total discharge to ocean
    if (trim(uname) == 'TOTAL_DISCHARGE_TO_OCEAN_'//trim(rtm_tracers(1))) then
      ptr => rtmCTL%runofftot_nt1
      return
    end if
    if (trim(uname) == 'TOTAL_DISCHARGE_TO_OCEAN_'//trim(rtm_tracers(2))) then
      ptr => rtmCTL%runofftot_nt2
      return
    end if

    ! If not found, warn and leave null
    if (masterproc) then
      write(iulog,*) 'get_rtm_field_ptr: WARNING - unknown field name: ', trim(uname)
    end if

  end subroutine get_rtm_field_ptr

end module RtmHistDerived
