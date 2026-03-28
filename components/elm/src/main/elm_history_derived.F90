module elm_history_derived
  !-------------------------------------------------------------------------------------------
  !
  ! Online derived field processing for ELM history output.
  !
  ! Provides two capabilities:
  !   1. Soil depth coarsening: dz-weighted averaging of column-level soil fields onto
  !      user-defined coarser depth layers, output as individual 1D fields per layer.
  !   2. Field combinations: summation of multiple column-level fields into a single
  !      derived field.
  !
  ! These are composable: a derived (combined) field can also be vertically coarsened.
  !
  ! Configuration is via namelist (elm_derived_fields_nl):
  !   elm_vcoarsen_zbounds  - depth boundaries (m) defining coarsened layers
  !   elm_vcoarsen_flds     - field names to vertically coarsen
  !   elm_derived_fld_defs  - field combination definitions, e.g. "TOTAL_SOIL_WATER=H2OSOI_LIQ+H2OSOI_ICE"
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,    only: r8 => shr_kind_r8
  use elm_varpar,      only: nlevsoi
  use elm_varcon,      only: spval
  use elm_varctl,      only: iulog
  use abortutils,      only: endrun
  use spmdMod,         only: masterproc, mpicom, MPI_REAL8, MPI_CHARACTER, MPI_INTEGER
  use decompMod,       only: bounds_type, get_proc_bounds

  implicit none
  private
  save

  public :: elm_derived_fields_readnl
  public :: elm_derived_fields_register
  public :: elm_derived_fields_update

  ! Parameters
  integer, parameter :: max_vcoarsen_bounds = 51
  integer, parameter :: max_vcoarsen_flds   = 100
  integer, parameter :: max_derived_flds    = 50
  integer, parameter :: max_derived_inputs  = 20
  integer, parameter :: max_name_len        = 64
  integer, parameter :: max_def_len         = 256

  ! Namelist variables
  real(r8) :: elm_vcoarsen_zbounds(max_vcoarsen_bounds)
  character(len=max_name_len) :: elm_vcoarsen_flds(max_vcoarsen_flds)
  character(len=max_def_len)  :: elm_derived_fld_defs(max_derived_flds)

  ! Parsed state: vertical coarsening
  integer :: n_vcoarsen_levs = 0
  integer :: n_vcoarsen_flds = 0

  ! Parsed state: derived field combinations
  integer :: n_derived_flds = 0

  type :: derived_field_t
    character(len=max_name_len) :: output_name
    integer                     :: n_inputs
    character(len=max_name_len) :: input_names(max_derived_inputs)
    character(len=1)            :: operators(max_derived_inputs)
    character(len=max_def_len)  :: units
    character(len=max_def_len)  :: long_name
    logical                     :: do_vcoarsen
  end type derived_field_t

  type(derived_field_t) :: derived_flds(max_derived_flds)

  logical :: module_is_initialized = .false.
  logical :: has_vcoarsen = .false.
  logical :: has_derived  = .false.

contains

  !============================================================================
  subroutine elm_derived_fields_readnl(nlfile)
    use shr_log_mod, only: errMsg => shr_log_errMsg

    character(len=*), intent(in) :: nlfile

    integer :: unitn, ierr, i
    integer :: masterprocid

    namelist /elm_derived_fields_nl/ elm_vcoarsen_zbounds, elm_vcoarsen_flds, elm_derived_fld_defs

    masterprocid = 0

    ! Initialize defaults
    elm_vcoarsen_zbounds(:) = -1.0_r8
    elm_vcoarsen_flds(:) = ''
    elm_derived_fld_defs(:) = ''

    if (masterproc) then
      open(newunit=unitn, file=trim(nlfile), status='old', iostat=ierr)
      if (ierr == 0) then
        ! Try to find the namelist group
        read(unitn, nml=elm_derived_fields_nl, iostat=ierr)
        if (ierr > 0) then
          ! Namelist read error - not necessarily fatal, group may not exist
          ! Reset to defaults
          elm_vcoarsen_zbounds(:) = -1.0_r8
          elm_vcoarsen_flds(:) = ''
          elm_derived_fld_defs(:) = ''
        end if
        close(unitn)
      end if
    end if

    ! Broadcast to all processors
    call mpi_bcast(elm_vcoarsen_zbounds, max_vcoarsen_bounds, MPI_REAL8, masterprocid, mpicom, ierr)
    call mpi_bcast(elm_vcoarsen_flds, max_name_len*max_vcoarsen_flds, MPI_CHARACTER, masterprocid, mpicom, ierr)
    call mpi_bcast(elm_derived_fld_defs, max_def_len*max_derived_flds, MPI_CHARACTER, masterprocid, mpicom, ierr)

    ! Parse vertical coarsening bounds
    n_vcoarsen_levs = 0
    do i = 2, max_vcoarsen_bounds
      if (elm_vcoarsen_zbounds(i) < 0.0_r8) exit
      n_vcoarsen_levs = n_vcoarsen_levs + 1
    end do
    has_vcoarsen = (n_vcoarsen_levs > 0)

    ! Validate depth bounds are non-negative and strictly increasing
    if (has_vcoarsen) then
      if (elm_vcoarsen_zbounds(1) < 0.0_r8) then
        call endrun('elm_derived_fields_readnl: elm_vcoarsen_zbounds(1) must be non-negative')
      end if
      do i = 1, n_vcoarsen_levs
        if (elm_vcoarsen_zbounds(i+1) <= elm_vcoarsen_zbounds(i)) then
          call endrun('elm_derived_fields_readnl: elm_vcoarsen_zbounds must be strictly increasing')
        end if
      end do
    end if

    ! Count fields to coarsen
    n_vcoarsen_flds = 0
    do i = 1, max_vcoarsen_flds
      if (len_trim(elm_vcoarsen_flds(i)) == 0) exit
      n_vcoarsen_flds = n_vcoarsen_flds + 1
    end do

    ! Parse derived field definitions
    n_derived_flds = 0
    do i = 1, max_derived_flds
      if (len_trim(elm_derived_fld_defs(i)) == 0) exit
      n_derived_flds = n_derived_flds + 1
      call parse_derived_def(elm_derived_fld_defs(i), derived_flds(n_derived_flds))
    end do
    has_derived = (n_derived_flds > 0)

    ! Mark derived fields that should also be coarsened
    call mark_derived_for_coarsening()

    if (masterproc) then
      if (has_vcoarsen) then
        write(iulog,*) 'elm_derived_fields_readnl: soil depth coarsening enabled with ', &
             n_vcoarsen_levs, ' layers'
        write(iulog,*) '  Depth boundaries (m):', elm_vcoarsen_zbounds(1:n_vcoarsen_levs+1)
        write(iulog,*) '  Fields to coarsen:', n_vcoarsen_flds
      end if
      if (has_derived) then
        write(iulog,*) 'elm_derived_fields_readnl: ', n_derived_flds, ' derived field(s) defined'
      end if
    end if

  end subroutine elm_derived_fields_readnl

  !============================================================================
  subroutine parse_derived_def(defstr, dfld)
    character(len=*), intent(in)       :: defstr
    type(derived_field_t), intent(out) :: dfld

    integer :: eq_pos, op_pos, start_pos, n, i
    character(len=max_def_len) :: rhs
    character(len=1) :: ch

    dfld%n_inputs = 0
    dfld%do_vcoarsen = .false.
    dfld%units = 'mixed'
    dfld%operators(:) = '+'

    eq_pos = index(defstr, '=')
    if (eq_pos < 2) then
      call endrun('parse_derived_def: invalid definition (no "="), got: '//trim(defstr))
    end if

    dfld%output_name = adjustl(defstr(1:eq_pos-1))
    rhs = adjustl(defstr(eq_pos+1:))

    n = 0
    start_pos = 1
    do
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

      if (n == 1) dfld%operators(n) = '+'

      if (op_pos > 0) then
        dfld%input_names(n) = adjustl(rhs(start_pos:op_pos-1))
        if (n < max_derived_inputs) dfld%operators(n+1) = rhs(op_pos:op_pos)
        start_pos = op_pos + 1
      else
        dfld%input_names(n) = adjustl(rhs(start_pos:))
        exit
      end if
    end do
    dfld%n_inputs = n

    dfld%long_name = trim(dfld%input_names(1))
    do n = 2, dfld%n_inputs
      dfld%long_name = trim(dfld%long_name) // ' ' // dfld%operators(n) // ' ' // &
           trim(dfld%input_names(n))
    end do

  end subroutine parse_derived_def

  !============================================================================
  subroutine mark_derived_for_coarsening()
    integer :: i, j

    do i = 1, n_derived_flds
      do j = 1, n_vcoarsen_flds
        if (trim(derived_flds(i)%output_name) == trim(elm_vcoarsen_flds(j))) then
          derived_flds(i)%do_vcoarsen = .true.
          elm_vcoarsen_flds(j) = ''
          exit
        end if
      end do
    end do

  end subroutine mark_derived_for_coarsening

  !============================================================================
  subroutine elm_derived_fields_register()
    use histFileMod, only: hist_addfld1d, hist_addfld2d

    integer :: i, k
    character(len=max_name_len) :: fname
    character(len=max_def_len)  :: lname

    if (.not. has_derived .and. .not. has_vcoarsen) return

    ! Register derived (combined) fields as 2D fields on nlevsoi
    do i = 1, n_derived_flds
      call hist_addfld2d(fname=trim(derived_flds(i)%output_name), &
           units=trim(derived_flds(i)%units), type2d='levsoi', &
           avgflag='A', long_name=trim(derived_flds(i)%long_name))

      ! If also coarsened, register 1D fields per layer
      if (derived_flds(i)%do_vcoarsen) then
        do k = 1, n_vcoarsen_levs
          call make_vcoarsen_name(derived_flds(i)%output_name, k, fname)
          call make_vcoarsen_longname(derived_flds(i)%long_name, k, lname)
          call hist_addfld1d(fname=trim(fname), units=trim(derived_flds(i)%units), &
               avgflag='A', long_name=trim(lname))
        end do
      end if
    end do

    ! Register vertically coarsened 1D fields for state variables
    do i = 1, n_vcoarsen_flds
      if (len_trim(elm_vcoarsen_flds(i)) == 0) cycle
      do k = 1, n_vcoarsen_levs
        call make_vcoarsen_name(elm_vcoarsen_flds(i), k, fname)
        write(lname, '(A,A,I0,A,F0.3,A,F0.3,A)') &
             trim(elm_vcoarsen_flds(i)), ' vcoarsen layer ', k, &
             ' (', elm_vcoarsen_zbounds(k), '-', elm_vcoarsen_zbounds(k+1), ' m)'
        call hist_addfld1d(fname=trim(fname), units='varies', &
             avgflag='A', long_name=trim(lname))
      end do
    end do

    module_is_initialized = .true.

    if (masterproc) then
      write(iulog,*) 'elm_derived_fields_register: registration complete'
    end if

  end subroutine elm_derived_fields_register

  !============================================================================
  subroutine elm_derived_fields_update(bounds, &
       t_soisno_col, h2osoi_liq_col, h2osoi_ice_col, dz_col)
    !--------------------------------------------------------------------------
    ! Compute and output derived fields for the current timestep.
    ! Called each timestep with column-level soil data.
    !--------------------------------------------------------------------------
    use histFileMod, only: hist_addfld_1d

    type(bounds_type), intent(in) :: bounds
    real(r8), intent(in) :: t_soisno_col(bounds%begc:, 1:)    ! soil temperature [K]
    real(r8), intent(in) :: h2osoi_liq_col(bounds%begc:, 1:)  ! soil liquid water [kg/m2]
    real(r8), intent(in) :: h2osoi_ice_col(bounds%begc:, 1:)  ! soil ice [kg/m2]
    real(r8), intent(in) :: dz_col(bounds%begc:, 1:)          ! soil layer thickness [m]

    real(r8) :: tmp_field(bounds%begc:bounds%endc, 1:nlevsoi)
    real(r8) :: src_field(bounds%begc:bounds%endc, 1:nlevsoi)
    real(r8) :: coarsened(bounds%begc:bounds%endc)
    integer  :: i, k, n, c
    character(len=max_name_len) :: fname

    if (.not. has_derived .and. .not. has_vcoarsen) return

    ! --- Step 1: Compute and output derived fields ---
    do i = 1, n_derived_flds
      tmp_field(:,:) = 0.0_r8

      do n = 1, derived_flds(i)%n_inputs
        call get_soil_field(bounds, derived_flds(i)%input_names(n), src_field, &
             t_soisno_col, h2osoi_liq_col, h2osoi_ice_col)
        select case (derived_flds(i)%operators(n))
        case ('+')
          tmp_field(bounds%begc:bounds%endc, :) = &
               tmp_field(bounds%begc:bounds%endc, :) + src_field(bounds%begc:bounds%endc, :)
        case ('-')
          tmp_field(bounds%begc:bounds%endc, :) = &
               tmp_field(bounds%begc:bounds%endc, :) - src_field(bounds%begc:bounds%endc, :)
        case ('*')
          if (n == 1) then
            tmp_field(bounds%begc:bounds%endc, :) = src_field(bounds%begc:bounds%endc, :)
          else
            tmp_field(bounds%begc:bounds%endc, :) = &
                 tmp_field(bounds%begc:bounds%endc, :) * src_field(bounds%begc:bounds%endc, :)
          end if
        case ('/')
          if (n == 1) then
            tmp_field(bounds%begc:bounds%endc, :) = src_field(bounds%begc:bounds%endc, :)
          else
            where (src_field(bounds%begc:bounds%endc, :) /= 0.0_r8)
              tmp_field(bounds%begc:bounds%endc, :) = &
                   tmp_field(bounds%begc:bounds%endc, :) / src_field(bounds%begc:bounds%endc, :)
            elsewhere
              tmp_field(bounds%begc:bounds%endc, :) = 0.0_r8
            end where
          end if
        end select
      end do

      ! If this derived field also gets coarsened, do it now
      if (derived_flds(i)%do_vcoarsen) then
        do k = 1, n_vcoarsen_levs
          call vcoarsen_layer(tmp_field, dz_col, bounds, k, coarsened)
          call make_vcoarsen_name(derived_flds(i)%output_name, k, fname)
        end do
      end if
    end do

    ! --- Step 2: Vertically coarsen state fields ---
    do i = 1, n_vcoarsen_flds
      if (len_trim(elm_vcoarsen_flds(i)) == 0) cycle

      call get_soil_field(bounds, elm_vcoarsen_flds(i), src_field, &
           t_soisno_col, h2osoi_liq_col, h2osoi_ice_col)

      do k = 1, n_vcoarsen_levs
        call vcoarsen_layer(src_field, dz_col, bounds, k, coarsened)
        call make_vcoarsen_name(elm_vcoarsen_flds(i), k, fname)
      end do
    end do

  end subroutine elm_derived_fields_update

  !============================================================================
  subroutine vcoarsen_layer(field, dz, bounds, k_out, field_out)
    !--------------------------------------------------------------------------
    ! Compute dz-weighted average of 'field' for coarsened layer k_out.
    ! Uses soil depth boundaries instead of pressure boundaries.
    !--------------------------------------------------------------------------
    use elm_varpar, only: nlevsoi
    use elm_varcon, only: zsoi => zsoi_decomp

    real(r8), intent(in)  :: field(:,:)
    real(r8), intent(in)  :: dz(:,:)
    type(bounds_type), intent(in) :: bounds
    integer,  intent(in)  :: k_out
    real(r8), intent(out) :: field_out(bounds%begc:)

    real(r8) :: zb_top, zb_bot
    real(r8) :: z_top, z_bot, overlap
    real(r8) :: numerator, denominator
    integer  :: c, k

    zb_top = elm_vcoarsen_zbounds(k_out)
    zb_bot = elm_vcoarsen_zbounds(k_out + 1)

    do c = bounds%begc, bounds%endc
      numerator   = 0.0_r8
      denominator = 0.0_r8

      z_top = 0.0_r8
      do k = 1, nlevsoi
        z_bot = z_top + dz(c, k)
        overlap = max(0.0_r8, min(z_bot, zb_bot) - max(z_top, zb_top))

        if (overlap > 0.0_r8) then
          numerator   = numerator   + field(c, k) * overlap
          denominator = denominator + overlap
        end if

        z_top = z_bot
      end do

      if (denominator > 0.0_r8) then
        field_out(c) = numerator / denominator
      else
        field_out(c) = spval
      end if
    end do

  end subroutine vcoarsen_layer

  !============================================================================
  subroutine get_soil_field(bounds, fname, field_out, t_soisno, h2osoi_liq, h2osoi_ice)
    type(bounds_type), intent(in)  :: bounds
    character(len=*),  intent(in)  :: fname
    real(r8),          intent(out) :: field_out(bounds%begc:, 1:)
    real(r8),          intent(in)  :: t_soisno(bounds%begc:, 1:)
    real(r8),          intent(in)  :: h2osoi_liq(bounds%begc:, 1:)
    real(r8),          intent(in)  :: h2osoi_ice(bounds%begc:, 1:)

    character(len=max_name_len) :: uname

    uname = adjustl(fname)
    field_out(:,:) = 0.0_r8

    select case (trim(uname))
    case ('TSOI', 'T_SOISNO')
      field_out(bounds%begc:bounds%endc, :) = t_soisno(bounds%begc:bounds%endc, :)
    case ('H2OSOI_LIQ')
      field_out(bounds%begc:bounds%endc, :) = h2osoi_liq(bounds%begc:bounds%endc, :)
    case ('H2OSOI_ICE', 'SOILICE')
      field_out(bounds%begc:bounds%endc, :) = h2osoi_ice(bounds%begc:bounds%endc, :)
    case default
      call endrun('elm_derived: get_soil_field: unknown field name: '//trim(uname))
    end select

  end subroutine get_soil_field

  !============================================================================
  subroutine make_vcoarsen_name(base_name, layer_idx, out_name)
    character(len=*), intent(in)  :: base_name
    integer,          intent(in)  :: layer_idx
    character(len=*), intent(out) :: out_name
    character(len=4) :: idx_str

    write(idx_str, '(I0)') layer_idx
    out_name = trim(base_name) // '_' // trim(idx_str)

  end subroutine make_vcoarsen_name

  !============================================================================
  subroutine make_vcoarsen_longname(base_long, layer_idx, out_long)
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
    integer, intent(in) :: k
    character(len=64) :: rstr

    write(rstr, '(F0.3,A,F0.3,A)') &
         elm_vcoarsen_zbounds(k), '-', elm_vcoarsen_zbounds(k+1), ' m'

  end function pb_range_str

end module elm_history_derived
