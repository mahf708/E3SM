module cam_history_derived
  !-------------------------------------------------------------------------------------------
  !
  ! Online derived field processing for EAM history output.
  !
  ! Provides two capabilities:
  !   1. Vertical coarsening: pdel-weighted averaging of 3D fields onto a user-defined
  !      coarser pressure grid, output as individual 2D (scalar) fields per layer.
  !   2. Field combinations: summation of multiple constituent/state fields into a single
  !      derived field.
  !
  ! These are composable: a derived (combined) field can also be vertically coarsened.
  !
  ! Configuration is via namelist (derived_fields_nl):
  !   vcoarsen_pbounds  - pressure boundaries (Pa) defining coarsened layers, top to surface
  !   vcoarsen_flds     - field names to vertically coarsen
  !   derived_fld_defs  - field combination definitions, e.g. "TOTAL_N=NUMLIQ+NUMICE"
  !
  ! Usage from physpkg.F90:
  !   call derived_fields_readnl(nlfile)   ! during namelist reading
  !   call derived_fields_register()       ! during phys_init, after other addfld calls
  !   call derived_fields_writeout(state)  ! during physics timestep
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,    only: r8 => shr_kind_r8
  use ppgrid,          only: pcols, pver, pverp
  use cam_logfile,     only: iulog
  use cam_abortutils,  only: endrun
  use spmd_utils,      only: masterproc

  implicit none
  private
  save

  public :: derived_fields_readnl
  public :: derived_fields_register
  public :: derived_fields_writeout

  ! Parameters
  integer, parameter :: max_vcoarsen_bounds = 51   ! max pressure boundaries (=> max 50 layers)
  integer, parameter :: max_vcoarsen_flds   = 100  ! max fields to coarsen
  integer, parameter :: max_derived_flds    = 50   ! max derived field definitions
  integer, parameter :: max_derived_inputs  = 20   ! max input fields per derived definition
  integer, parameter :: max_name_len        = 34   ! matches fieldname_len in cam_history
  integer, parameter :: max_def_len         = 256  ! max length of a derived field definition string

  ! Namelist variables
  real(r8) :: vcoarsen_pbounds(max_vcoarsen_bounds)
  character(len=max_name_len) :: vcoarsen_flds(max_vcoarsen_flds)
  character(len=max_def_len)  :: derived_fld_defs(max_derived_flds)

  ! Parsed state: vertical coarsening
  integer :: n_vcoarsen_levs = 0                ! number of coarsened layers
  integer :: n_vcoarsen_flds = 0                ! number of fields to coarsen

  ! Parsed state: derived field combinations
  integer :: n_derived_flds = 0                 ! number of derived fields

  type :: derived_field_t
    character(len=max_name_len) :: output_name            ! output field name
    integer                     :: n_inputs                ! number of input fields
    character(len=max_name_len) :: input_names(max_derived_inputs) ! input field names
    character(len=1)            :: operators(max_derived_inputs)   ! '+', '-', '*', '/' per input
    character(len=max_def_len)  :: units                   ! units string
    character(len=max_def_len)  :: long_name               ! long name for NetCDF
    logical                     :: do_vcoarsen             ! also vertically coarsen this field?
  end type derived_field_t

  type(derived_field_t) :: derived_flds(max_derived_flds)

  ! Flag to track initialization
  logical :: module_is_initialized = .false.
  logical :: has_vcoarsen = .false.
  logical :: has_derived  = .false.

contains

  !============================================================================
  subroutine derived_fields_readnl(nlfile)
    !--------------------------------------------------------------------------
    ! Read the derived_fields_nl namelist group
    !--------------------------------------------------------------------------
    use namelist_utils, only: find_group_name
    use units,          only: getunit, freeunit
    use spmd_utils,     only: mpicom, masterprocid, mpi_real8, mpi_character, mpi_integer

    character(len=*), intent(in) :: nlfile

    integer :: unitn, ierr, i

    namelist /derived_fields_nl/ vcoarsen_pbounds, vcoarsen_flds, derived_fld_defs

    ! Initialize defaults
    vcoarsen_pbounds(:) = -1.0_r8
    vcoarsen_flds(:) = ''
    derived_fld_defs(:) = ''

    if (masterproc) then
      unitn = getunit()
      open(unitn, file=trim(nlfile), status='old')
      call find_group_name(unitn, 'derived_fields_nl', status=ierr)
      if (ierr == 0) then
        read(unitn, derived_fields_nl, iostat=ierr)
        if (ierr /= 0) then
          call endrun('derived_fields_readnl: ERROR reading namelist derived_fields_nl')
        end if
      end if
      close(unitn)
      call freeunit(unitn)
    end if

    ! Broadcast to all processors
    call mpi_bcast(vcoarsen_pbounds, max_vcoarsen_bounds, mpi_real8, masterprocid, mpicom, ierr)
    call mpi_bcast(vcoarsen_flds, max_name_len*max_vcoarsen_flds, mpi_character, masterprocid, mpicom, ierr)
    call mpi_bcast(derived_fld_defs, max_def_len*max_derived_flds, mpi_character, masterprocid, mpicom, ierr)

    ! Parse vertical coarsening bounds
    n_vcoarsen_levs = 0
    do i = 2, max_vcoarsen_bounds
      if (vcoarsen_pbounds(i) < 0.0_r8) exit
      n_vcoarsen_levs = n_vcoarsen_levs + 1
    end do
    has_vcoarsen = (n_vcoarsen_levs > 0)

    ! Validate pressure bounds are non-negative and strictly increasing
    if (has_vcoarsen) then
      if (vcoarsen_pbounds(1) < 0.0_r8) then
        call endrun('derived_fields_readnl: vcoarsen_pbounds(1) must be non-negative')
      end if
      do i = 1, n_vcoarsen_levs
        if (vcoarsen_pbounds(i+1) <= vcoarsen_pbounds(i)) then
          call endrun('derived_fields_readnl: vcoarsen_pbounds must be strictly increasing')
        end if
      end do
    end if

    ! Count fields to coarsen
    n_vcoarsen_flds = 0
    do i = 1, max_vcoarsen_flds
      if (len_trim(vcoarsen_flds(i)) == 0) exit
      n_vcoarsen_flds = n_vcoarsen_flds + 1
    end do

    ! Parse derived field definitions
    n_derived_flds = 0
    do i = 1, max_derived_flds
      if (len_trim(derived_fld_defs(i)) == 0) exit
      n_derived_flds = n_derived_flds + 1
      call parse_derived_def(derived_fld_defs(i), derived_flds(n_derived_flds))
    end do
    has_derived = (n_derived_flds > 0)

    ! Mark derived fields that should also be coarsened
    call mark_derived_for_coarsening()

    if (masterproc) then
      if (has_vcoarsen) then
        write(iulog,*) 'derived_fields_readnl: vertical coarsening enabled with ', &
             n_vcoarsen_levs, ' layers'
        write(iulog,*) '  Pressure boundaries (Pa):', vcoarsen_pbounds(1:n_vcoarsen_levs+1)
        write(iulog,*) '  Fields to coarsen:', n_vcoarsen_flds
      end if
      if (has_derived) then
        write(iulog,*) 'derived_fields_readnl: ', n_derived_flds, ' derived field(s) defined'
      end if
    end if

  end subroutine derived_fields_readnl

  !============================================================================
  subroutine parse_derived_def(defstr, dfld)
    !--------------------------------------------------------------------------
    ! Parse a derived field definition string of the form:
    !   "OUTPUT_NAME=INPUT1+INPUT2-INPUT3*INPUT4/INPUT5"
    ! Supported operators between fields: +, -, *, /
    ! The first input is always added (implicit +).
    ! Numeric constants (e.g., "PRECT*1000.0") are NOT supported;
    ! all operands must be field names.
    !--------------------------------------------------------------------------
    character(len=*), intent(in)       :: defstr
    type(derived_field_t), intent(out) :: dfld

    integer :: eq_pos, op_pos, start_pos, n, i
    character(len=max_def_len) :: rhs
    character(len=1) :: ch

    dfld%n_inputs = 0
    dfld%do_vcoarsen = .false.
    dfld%units = 'mixed'
    dfld%operators(:) = '+'

    ! Find '=' separator
    eq_pos = index(defstr, '=')
    if (eq_pos < 2) then
      call endrun('parse_derived_def: invalid definition (no "="), got: '//trim(defstr))
    end if

    dfld%output_name = adjustl(defstr(1:eq_pos-1))
    rhs = adjustl(defstr(eq_pos+1:))

    ! Parse operator-separated input names
    ! First input always has implicit '+' operator
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
        call endrun('parse_derived_def: too many inputs in definition: '//trim(defstr))
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
  subroutine mark_derived_for_coarsening()
    !--------------------------------------------------------------------------
    ! Check if any derived field names also appear in vcoarsen_flds.
    ! If so, mark them for vertical coarsening and remove from the
    ! vcoarsen_flds list (they will be handled in the derived path).
    !--------------------------------------------------------------------------
    integer :: i, j

    do i = 1, n_derived_flds
      do j = 1, n_vcoarsen_flds
        if (trim(derived_flds(i)%output_name) == trim(vcoarsen_flds(j))) then
          derived_flds(i)%do_vcoarsen = .true.
          ! Remove from vcoarsen_flds by blanking (skip during coarsen of state fields)
          vcoarsen_flds(j) = ''
          exit
        end if
      end do
    end do

  end subroutine mark_derived_for_coarsening

  !============================================================================
  subroutine derived_fields_register()
    !--------------------------------------------------------------------------
    ! Register derived output fields with the history system via addfld.
    ! Must be called during phys_init, after other addfld calls but before intht.
    !--------------------------------------------------------------------------
    use cam_history, only: addfld, horiz_only
    use constituents, only: cnst_get_ind

    integer :: i, k, n, idx
    character(len=max_name_len) :: fname
    character(len=max_def_len)  :: lname
    character(len=max_name_len) :: uname
    character(len=6), parameter :: known_state_vars(6) = &
         (/ 'T     ', 'U     ', 'V     ', 'OMEGA ', 'Z3    ', 'Q     ' /)
    logical :: found

    if (.not. has_derived .and. .not. has_vcoarsen) return

    ! Validate all input field names before registering
    ! Check derived field inputs
    do i = 1, n_derived_flds
      do n = 1, derived_flds(i)%n_inputs
        uname = adjustl(derived_flds(i)%input_names(n))
        found = .false.
        do k = 1, size(known_state_vars)
          if (trim(uname) == trim(known_state_vars(k))) then
            found = .true.
            exit
          end if
        end do
        if (.not. found) then
          call cnst_get_ind(trim(uname), idx, abrtf=.false.)
          found = (idx > 0)
        end if
        if (.not. found) then
          call endrun('derived_fields_register: unknown input field "'//trim(uname)// &
               '" in derived definition "'//trim(derived_flds(i)%output_name)// &
               '". Must be T, U, V, OMEGA, Z3, Q, or a registered constituent.')
        end if
      end do
    end do
    ! Check vcoarsen field names (skip blanked and derived-field names)
    do i = 1, n_vcoarsen_flds
      if (len_trim(vcoarsen_flds(i)) == 0) cycle
      uname = adjustl(vcoarsen_flds(i))
      found = .false.
      do k = 1, size(known_state_vars)
        if (trim(uname) == trim(known_state_vars(k))) then
          found = .true.
          exit
        end if
      end do
      if (.not. found) then
        call cnst_get_ind(trim(uname), idx, abrtf=.false.)
        found = (idx > 0)
      end if
      if (.not. found) then
        call endrun('derived_fields_register: unknown vcoarsen field "'//trim(uname)// &
             '". Must be T, U, V, OMEGA, Z3, Q, or a registered constituent.')
      end if
    end do

    ! Register derived (combined) fields as 3D fields on 'lev'
    do i = 1, n_derived_flds
      call addfld(trim(derived_flds(i)%output_name), (/ 'lev' /), 'A', &
           trim(derived_flds(i)%units), trim(derived_flds(i)%long_name))

      ! If this derived field also gets coarsened, register the scalar fields
      if (derived_flds(i)%do_vcoarsen) then
        do k = 1, n_vcoarsen_levs
          call make_vcoarsen_name(derived_flds(i)%output_name, k, fname)
          call make_vcoarsen_longname(derived_flds(i)%long_name, k, lname)
          call addfld(trim(fname), horiz_only, 'A', &
               trim(derived_flds(i)%units), trim(lname))
        end do
      end if
    end do

    ! Register vertically coarsened fields for state variables
    do i = 1, n_vcoarsen_flds
      if (len_trim(vcoarsen_flds(i)) == 0) cycle  ! skip blanked entries
      do k = 1, n_vcoarsen_levs
        call make_vcoarsen_name(vcoarsen_flds(i), k, fname)
        write(lname, '(A,A,I0,A,F0.1,A,F0.1,A)') &
             trim(vcoarsen_flds(i)), ' vcoarsen layer ', k, &
             ' (', vcoarsen_pbounds(k)/100.0_r8, '-', &
             vcoarsen_pbounds(k+1)/100.0_r8, ' hPa)'
        call addfld(trim(fname), horiz_only, 'A', 'varies', trim(lname))
      end do
    end do

    module_is_initialized = .true.

    if (masterproc) then
      write(iulog,*) 'derived_fields_register: registration complete'
    end if

  end subroutine derived_fields_register

  !============================================================================
  subroutine derived_fields_writeout(state)
    !--------------------------------------------------------------------------
    ! Compute and output derived fields for the current physics chunk.
    ! Called each timestep from physpkg.F90 with access to physics_state.
    !--------------------------------------------------------------------------
    use physics_types, only: physics_state
    use cam_history,   only: outfld
    use constituents,  only: cnst_get_ind, pcnst, cnst_name

    type(physics_state), intent(in) :: state

    real(r8) :: tmp_field(pcols, pver)     ! temporary for derived field computation
    real(r8) :: src_field(pcols, pver)     ! source field for coarsening
    real(r8) :: coarsened(pcols)           ! single coarsened layer output
    integer  :: i, k, n, ncol, lchnk
    integer  :: cnst_idx
    character(len=max_name_len) :: fname

    if (.not. has_derived .and. .not. has_vcoarsen) return

    ncol  = state%ncol
    lchnk = state%lchnk

    ! --- Step 1: Compute and output derived fields ---
    do i = 1, n_derived_flds
      tmp_field(:,:) = 0.0_r8

      do n = 1, derived_flds(i)%n_inputs
        call get_state_field(state, derived_flds(i)%input_names(n), src_field, ncol)
        select case (derived_flds(i)%operators(n))
        case ('+')
          tmp_field(1:ncol, 1:pver) = tmp_field(1:ncol, 1:pver) + src_field(1:ncol, 1:pver)
        case ('-')
          tmp_field(1:ncol, 1:pver) = tmp_field(1:ncol, 1:pver) - src_field(1:ncol, 1:pver)
        case ('*')
          ! For multiply, first input initializes, subsequent inputs multiply
          if (n == 1) then
            tmp_field(1:ncol, 1:pver) = src_field(1:ncol, 1:pver)
          else
            tmp_field(1:ncol, 1:pver) = tmp_field(1:ncol, 1:pver) * src_field(1:ncol, 1:pver)
          end if
        case ('/')
          ! For divide, first input initializes, subsequent inputs divide
          if (n == 1) then
            tmp_field(1:ncol, 1:pver) = src_field(1:ncol, 1:pver)
          else
            where (src_field(1:ncol, 1:pver) /= 0.0_r8)
              tmp_field(1:ncol, 1:pver) = tmp_field(1:ncol, 1:pver) / src_field(1:ncol, 1:pver)
            elsewhere
              tmp_field(1:ncol, 1:pver) = 0.0_r8
            end where
          end if
        end select
      end do

      ! Output the full-level combined field
      call outfld(trim(derived_flds(i)%output_name), tmp_field, pcols, lchnk)

      ! If this derived field also gets coarsened, do it now
      if (derived_flds(i)%do_vcoarsen) then
        do k = 1, n_vcoarsen_levs
          call vcoarsen_layer(tmp_field, state%pint, ncol, k, coarsened)
          call make_vcoarsen_name(derived_flds(i)%output_name, k, fname)
          call outfld(trim(fname), coarsened, pcols, lchnk)
        end do
      end if
    end do

    ! --- Step 2: Vertically coarsen state fields ---
    do i = 1, n_vcoarsen_flds
      if (len_trim(vcoarsen_flds(i)) == 0) cycle

      call get_state_field(state, vcoarsen_flds(i), src_field, ncol)

      do k = 1, n_vcoarsen_levs
        call vcoarsen_layer(src_field, state%pint, ncol, k, coarsened)
        call make_vcoarsen_name(vcoarsen_flds(i), k, fname)
        call outfld(trim(fname), coarsened, pcols, lchnk)
      end do
    end do

  end subroutine derived_fields_writeout

  !============================================================================
  subroutine vcoarsen_layer(field, pint, ncol, k_out, field_out)
    !--------------------------------------------------------------------------
    ! Compute pdel-weighted average of 'field' for coarsened layer k_out.
    !
    ! For each column, the coarsened value is:
    !   sum(field(k) * overlap(k)) / sum(overlap(k))
    ! where overlap(k) is the pressure thickness overlap between model level k
    ! and the target pressure range [pb(k_out), pb(k_out+1)].
    !--------------------------------------------------------------------------
    use cam_history_support, only: fillvalue
    real(r8), intent(in)  :: field(pcols, pver)      ! input 3D field
    real(r8), intent(in)  :: pint(pcols, pverp)      ! interface pressures
    integer,  intent(in)  :: ncol                    ! number of active columns
    integer,  intent(in)  :: k_out                   ! target coarsened layer index
    real(r8), intent(out) :: field_out(pcols)         ! output coarsened values

    real(r8) :: pb_top, pb_bot           ! target layer pressure bounds
    real(r8) :: p_top, p_bot, overlap    ! model level overlap computation
    real(r8) :: numerator, denominator
    integer  :: i, k

    pb_top = vcoarsen_pbounds(k_out)
    pb_bot = vcoarsen_pbounds(k_out + 1)

    do i = 1, ncol
      numerator   = 0.0_r8
      denominator = 0.0_r8

      do k = 1, pver
        p_top = max(pint(i, k),   pb_top)
        p_bot = min(pint(i, k+1), pb_bot)
        overlap = max(0.0_r8, p_bot - p_top)

        if (overlap > 0.0_r8) then
          numerator   = numerator   + field(i, k) * overlap
          denominator = denominator + overlap
        end if
      end do

      if (denominator > 0.0_r8) then
        field_out(i) = numerator / denominator
      else
        field_out(i) = fillvalue
      end if
    end do

    ! Zero out inactive columns
    field_out(ncol+1:pcols) = fillvalue

  end subroutine vcoarsen_layer

  !============================================================================
  subroutine get_state_field(state, fname, field_out, ncol)
    !--------------------------------------------------------------------------
    ! Look up a field name and extract the corresponding 2D slice from
    ! physics_state. Supports standard state variables and constituents.
    !--------------------------------------------------------------------------
    use physics_types, only: physics_state
    use constituents,  only: cnst_get_ind, pcnst, cnst_name

    type(physics_state), intent(in)  :: state
    character(len=*),    intent(in)  :: fname
    real(r8),            intent(out) :: field_out(pcols, pver)
    integer,             intent(in)  :: ncol

    integer :: idx
    character(len=max_name_len) :: uname

    uname = adjustl(fname)
    field_out(:,:) = 0.0_r8

    ! Check standard state variables first
    select case (trim(uname))
    case ('T')
      field_out(1:ncol, :) = state%t(1:ncol, :)
      return
    case ('U')
      field_out(1:ncol, :) = state%u(1:ncol, :)
      return
    case ('V')
      field_out(1:ncol, :) = state%v(1:ncol, :)
      return
    case ('OMEGA')
      field_out(1:ncol, :) = state%omega(1:ncol, :)
      return
    case ('Z3')
      field_out(1:ncol, :) = state%zm(1:ncol, :)
      return
    case ('Q')
      field_out(1:ncol, :) = state%q(1:ncol, :, 1)
      return
    case ('PS')
      ! Surface pressure is 2D; replicate across levels for consistency
      ! (not typical for coarsening, but handle gracefully)
      call endrun('get_state_field: PS is a 2D field, cannot be used for vertical processing')
    end select

    ! Try constituent lookup
    call cnst_get_ind(trim(uname), idx, abrtf=.false.)
    if (idx > 0) then
      field_out(1:ncol, :) = state%q(1:ncol, :, idx)
      return
    end if

    ! Field not found
    call endrun('get_state_field: unknown field name: '//trim(uname)// &
         '. Must be T, U, V, OMEGA, Z3, Q, or a registered constituent name.')

  end subroutine get_state_field

  !============================================================================
  subroutine make_vcoarsen_name(base_name, layer_idx, out_name)
    !--------------------------------------------------------------------------
    ! Construct the output field name for a coarsened layer.
    ! E.g., base_name="U", layer_idx=3 => out_name="U_3"
    !--------------------------------------------------------------------------
    character(len=*), intent(in)  :: base_name
    integer,          intent(in)  :: layer_idx
    character(len=*), intent(out) :: out_name

    character(len=4) :: idx_str

    write(idx_str, '(I0)') layer_idx
    out_name = trim(base_name) // '_' // trim(idx_str)

  end subroutine make_vcoarsen_name

  !============================================================================
  subroutine make_vcoarsen_longname(base_long, layer_idx, out_long)
    !--------------------------------------------------------------------------
    ! Construct a descriptive long name for a coarsened layer field.
    !--------------------------------------------------------------------------
    character(len=*), intent(in)  :: base_long
    integer,          intent(in)  :: layer_idx
    character(len=*), intent(out) :: out_long

    character(len=4) :: idx_str

    write(idx_str, '(I0)') layer_idx
    out_long = trim(base_long) // ' vcoarsen layer ' // trim(idx_str) // &
         ' (' // trim(pb_range_str(layer_idx)) // ')'

  end subroutine make_vcoarsen_longname

  !============================================================================
  function pb_range_str(k) result(rstr)
    !--------------------------------------------------------------------------
    ! Return a string like "0.0-250.0 hPa" for coarsened layer k.
    !--------------------------------------------------------------------------
    integer, intent(in) :: k
    character(len=64) :: rstr

    write(rstr, '(F0.1,A,F0.1,A)') &
         vcoarsen_pbounds(k)/100.0_r8, '-', &
         vcoarsen_pbounds(k+1)/100.0_r8, ' hPa'

  end function pb_range_str

end module cam_history_derived
