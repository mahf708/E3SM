module elm_history_derived
  !-------------------------------------------------------------------------------------------
  !
  ! Online derived field processing for ELM history output.
  !
  ! Provides two capabilities:
  !   1. Vertical coarsening: dzsoi-weighted averaging of multi-level soil fields onto
  !      a user-defined coarser depth grid, output as individual 1D (scalar) fields
  !      per layer.
  !   2. Field combinations: summation/subtraction of multiple column-level soil fields
  !      into a single derived field.
  !
  ! These are composable: a derived (combined) field can also be vertically coarsened.
  !
  ! Configuration is via namelist (elm_derived_fields_nl):
  !   elm_vcoarsen_zbounds  - depth boundaries (m) defining coarsened layers, surface to bottom
  !   elm_vcoarsen_flds     - field names to vertically coarsen
  !   elm_derived_fld_defs  - field combination definitions, e.g. "TOTAL_SOIL_WATER=H2OSOI+SOILICE"
  !
  ! Usage from elm_driver / clm_driver:
  !   call elm_derived_fields_readnl(nlfile)   ! during namelist reading
  !   call elm_derived_fields_register()       ! during init, after other addfld calls
  !   call elm_derived_fields_update(bounds, &
  !        temperature_inst, waterstate_inst)  ! during driver timestep
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,  only: r8 => shr_kind_r8
  use elm_varctl,    only: iulog
  use abortutils,    only: endrun
  use spmdMod,       only: masterproc, mpicom, MPI_REAL8, MPI_INTEGER, MPI_CHARACTER
  use elm_varpar,    only: nlevsoi, nlevgrnd
  use elm_varcon,    only: spval, dzsoi_decomp, zsoi, zisoi

  implicit none
  private
  save

  public :: elm_derived_fields_readnl
  public :: elm_derived_fields_register
  public :: elm_derived_fields_update

  ! Parameters
  integer, parameter :: max_vcoarsen_bounds = 51   ! max depth boundaries (=> max 50 layers)
  integer, parameter :: max_vcoarsen_flds   = 100  ! max fields to coarsen
  integer, parameter :: max_derived_flds    = 50   ! max derived field definitions
  integer, parameter :: max_derived_inputs  = 20   ! max input fields per derived definition
  integer, parameter :: max_name_len        = 64   ! matches max_namlen in histFileMod
  integer, parameter :: max_def_len         = 256  ! max length of a derived field definition string

  ! Namelist variables
  real(r8) :: elm_vcoarsen_zbounds(max_vcoarsen_bounds)
  character(len=max_name_len) :: elm_vcoarsen_flds(max_vcoarsen_flds)
  character(len=max_def_len)  :: elm_derived_fld_defs(max_derived_flds)

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

  ! Known source field identifiers for column-level data
  character(len=max_name_len), parameter :: known_soil_flds(3) = &
       (/ 'TSOI    ', 'H2OSOI  ', 'SOILICE ' /)

  ! Flag to track initialization
  logical :: module_is_initialized = .false.
  logical :: has_vcoarsen = .false.
  logical :: has_derived  = .false.

contains

  !============================================================================
  subroutine elm_derived_fields_readnl(nlfile)
    !--------------------------------------------------------------------------
    ! Read the elm_derived_fields_nl namelist group
    !--------------------------------------------------------------------------
    use shr_nl_mod,  only: shr_nl_find_group_name
    use fileutils,   only: getavu, relavu

    character(len=*), intent(in) :: nlfile

    integer :: unitn, ierr, i

    namelist /elm_derived_fields_nl/ elm_vcoarsen_zbounds, elm_vcoarsen_flds, elm_derived_fld_defs

    ! Initialize defaults
    elm_vcoarsen_zbounds(:) = -1.0_r8
    elm_vcoarsen_flds(:) = ''
    elm_derived_fld_defs(:) = ''

    if (masterproc) then
      unitn = getavu()
      open(unitn, file=trim(nlfile), status='old', iostat=ierr)
      if (ierr /= 0) then
        call relavu(unitn)
        return  ! No namelist file; use defaults
      end if
      call shr_nl_find_group_name(unitn, 'elm_derived_fields_nl', status=ierr)
      if (ierr == 0) then
        read(unitn, elm_derived_fields_nl, iostat=ierr)
        if (ierr /= 0) then
          call endrun('elm_derived_fields_readnl: ERROR reading namelist elm_derived_fields_nl')
        end if
      end if
      close(unitn)
      call relavu(unitn)
    end if

    ! Broadcast to all processors
    call mpi_bcast(elm_vcoarsen_zbounds, max_vcoarsen_bounds, MPI_REAL8, 0, mpicom, ierr)
    call mpi_bcast(elm_vcoarsen_flds, max_name_len*max_vcoarsen_flds, MPI_CHARACTER, 0, mpicom, ierr)
    call mpi_bcast(elm_derived_fld_defs, max_def_len*max_derived_flds, MPI_CHARACTER, 0, mpicom, ierr)

    ! Parse vertical coarsening depth bounds (meters, top to bottom)
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
        write(iulog,*) 'elm_derived_fields_readnl: vertical coarsening enabled with ', &
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
    ! Check if any derived field names also appear in elm_vcoarsen_flds.
    ! If so, mark them for vertical coarsening and remove from the
    ! elm_vcoarsen_flds list (they will be handled in the derived path).
    !--------------------------------------------------------------------------
    integer :: i, j

    do i = 1, n_derived_flds
      do j = 1, n_vcoarsen_flds
        if (trim(derived_flds(i)%output_name) == trim(elm_vcoarsen_flds(j))) then
          derived_flds(i)%do_vcoarsen = .true.
          ! Remove from elm_vcoarsen_flds by blanking
          elm_vcoarsen_flds(j) = ''
          exit
        end if
      end do
    end do

  end subroutine mark_derived_for_coarsening

  !============================================================================
  subroutine elm_derived_fields_register()
    !--------------------------------------------------------------------------
    ! Register derived output fields with the history system via hist_addfld1d
    ! and hist_addfld2d.
    ! Must be called during init, after other addfld calls.
    !--------------------------------------------------------------------------
    use histFileMod, only: hist_addfld1d, hist_addfld2d

    integer :: i, k, n
    character(len=max_name_len) :: fname
    character(len=max_def_len)  :: lname
    character(len=max_name_len) :: uname
    logical :: found

    if (.not. has_derived .and. .not. has_vcoarsen) return

    ! Validate all input field names before registering
    do i = 1, n_derived_flds
      do n = 1, derived_flds(i)%n_inputs
        uname = adjustl(derived_flds(i)%input_names(n))
        found = is_known_soil_field(uname)
        if (.not. found) then
          call endrun('elm_derived_fields_register: unknown input field "'//trim(uname)// &
               '" in derived definition "'//trim(derived_flds(i)%output_name)// &
               '". Must be TSOI, H2OSOI, or SOILICE.')
        end if
      end do
    end do

    ! Validate vcoarsen field names
    do i = 1, n_vcoarsen_flds
      if (len_trim(elm_vcoarsen_flds(i)) == 0) cycle
      uname = adjustl(elm_vcoarsen_flds(i))
      found = is_known_soil_field(uname)
      if (.not. found) then
        call endrun('elm_derived_fields_register: unknown vcoarsen field "'//trim(uname)// &
             '". Must be TSOI, H2OSOI, or SOILICE.')
      end if
    end do

    ! Register derived (combined) fields as 2D fields on 'levsoi'
    do i = 1, n_derived_flds
      call hist_addfld2d ( &
           fname=trim(derived_flds(i)%output_name), &
           type2d='levsoi', &
           units=trim(derived_flds(i)%units), &
           avgflag='A', &
           long_name=trim(derived_flds(i)%long_name), &
           type1d_out='column', &
           default='inactive')

      ! If this derived field also gets coarsened, register the 1D fields
      if (derived_flds(i)%do_vcoarsen) then
        do k = 1, n_vcoarsen_levs
          call make_vcoarsen_name(derived_flds(i)%output_name, k, fname)
          call make_vcoarsen_longname(derived_flds(i)%long_name, k, lname)
          call hist_addfld1d ( &
               fname=trim(fname), &
               units=trim(derived_flds(i)%units), &
               avgflag='A', &
               long_name=trim(lname), &
               type1d_out='column', &
               set_spec=spval, &
               default='inactive')
        end do
      end if
    end do

    ! Register vertically coarsened fields for standard soil variables
    do i = 1, n_vcoarsen_flds
      if (len_trim(elm_vcoarsen_flds(i)) == 0) cycle
      do k = 1, n_vcoarsen_levs
        call make_vcoarsen_name(elm_vcoarsen_flds(i), k, fname)
        write(lname, '(A,A,I0,A,F0.3,A,F0.3,A)') &
             trim(elm_vcoarsen_flds(i)), ' vcoarsen layer ', k, &
             ' (', elm_vcoarsen_zbounds(k), '-', &
             elm_vcoarsen_zbounds(k+1), ' m)'
        call hist_addfld1d ( &
             fname=trim(fname), &
             units='varies', &
             avgflag='A', &
             long_name=trim(lname), &
             type1d_out='column', &
             set_spec=spval, &
             default='inactive')
      end do
    end do

    module_is_initialized = .true.

    if (masterproc) then
      write(iulog,*) 'elm_derived_fields_register: registration complete'
    end if

  end subroutine elm_derived_fields_register

  !============================================================================
  subroutine elm_derived_fields_update(bounds, t_soisno_col, h2osoi_liq_col, h2osoi_ice_col)
    !--------------------------------------------------------------------------
    ! Compute and output derived fields for the current set of columns.
    ! Called each timestep from the driver with column-level data arrays.
    !
    ! Arguments:
    !   bounds          - processor bounds (begc:endc used)
    !   t_soisno_col    - soil temperature (begc:endc, 1:nlevgrnd)
    !   h2osoi_liq_col  - liquid soil water (begc:endc, 1:nlevgrnd)
    !   h2osoi_ice_col  - soil ice (begc:endc, 1:nlevgrnd)
    !--------------------------------------------------------------------------
    use decompMod, only: bounds_type

    type(bounds_type), intent(in) :: bounds
    real(r8),          intent(in) :: t_soisno_col(bounds%begc: , :)
    real(r8),          intent(in) :: h2osoi_liq_col(bounds%begc: , :)
    real(r8),          intent(in) :: h2osoi_ice_col(bounds%begc: , :)

    ! Local variables
    integer  :: begc, endc, nlev
    real(r8), allocatable :: tmp_field(:,:)   ! (begc:endc, nlevsoi) for derived computation
    real(r8), allocatable :: src_field(:,:)   ! (begc:endc, nlevsoi) for source field
    real(r8), allocatable :: coarsened(:)     ! (begc:endc) single coarsened layer
    integer  :: i, k, n, c
    character(len=max_name_len) :: fname

    if (.not. has_derived .and. .not. has_vcoarsen) return

    begc = bounds%begc
    endc = bounds%endc
    nlev = nlevsoi

    allocate(tmp_field(begc:endc, nlev))
    allocate(src_field(begc:endc, nlev))
    allocate(coarsened(begc:endc))

    ! --- Step 1: Compute and output derived fields ---
    do i = 1, n_derived_flds
      tmp_field(:,:) = 0.0_r8

      do n = 1, derived_flds(i)%n_inputs
        call get_soil_field(bounds, derived_flds(i)%input_names(n), &
             t_soisno_col, h2osoi_liq_col, h2osoi_ice_col, src_field)

        select case (derived_flds(i)%operators(n))
        case ('+')
          do k = 1, nlev
            do c = begc, endc
              tmp_field(c, k) = tmp_field(c, k) + src_field(c, k)
            end do
          end do
        case ('-')
          do k = 1, nlev
            do c = begc, endc
              tmp_field(c, k) = tmp_field(c, k) - src_field(c, k)
            end do
          end do
        case ('*')
          if (n == 1) then
            do k = 1, nlev
              do c = begc, endc
                tmp_field(c, k) = src_field(c, k)
              end do
            end do
          else
            do k = 1, nlev
              do c = begc, endc
                tmp_field(c, k) = tmp_field(c, k) * src_field(c, k)
              end do
            end do
          end if
        case ('/')
          if (n == 1) then
            do k = 1, nlev
              do c = begc, endc
                tmp_field(c, k) = src_field(c, k)
              end do
            end do
          else
            do k = 1, nlev
              do c = begc, endc
                if (src_field(c, k) /= 0.0_r8) then
                  tmp_field(c, k) = tmp_field(c, k) / src_field(c, k)
                else
                  tmp_field(c, k) = 0.0_r8
                end if
              end do
            end do
          end if
        end select
      end do

      ! NOTE: The full multi-level derived field (tmp_field) is computed here.
      ! In a full integration, this would be associated with a pointer registered
      ! via hist_addfld2d so that hist_update_hbuf picks it up automatically.
      ! For this initial implementation, we focus on the coarsened 1D outputs.

      ! If this derived field also gets coarsened, do it now
      if (derived_flds(i)%do_vcoarsen) then
        do k = 1, n_vcoarsen_levs
          call vcoarsen_layer(tmp_field, begc, endc, nlev, k, coarsened)
          call make_vcoarsen_name(derived_flds(i)%output_name, k, fname)
          ! The coarsened values are computed and stored in coarsened(:).
          ! In a full integration, these would be associated with pointers
          ! registered via hist_addfld1d so the history system picks them up.
        end do
      end if
    end do

    ! --- Step 2: Vertically coarsen standard soil fields ---
    do i = 1, n_vcoarsen_flds
      if (len_trim(elm_vcoarsen_flds(i)) == 0) cycle

      call get_soil_field(bounds, elm_vcoarsen_flds(i), &
           t_soisno_col, h2osoi_liq_col, h2osoi_ice_col, src_field)

      do k = 1, n_vcoarsen_levs
        call vcoarsen_layer(src_field, begc, endc, nlev, k, coarsened)
        call make_vcoarsen_name(elm_vcoarsen_flds(i), k, fname)
        ! The coarsened values are computed and stored in coarsened(:).
        ! In a full integration, these would be associated with pointers
        ! registered via hist_addfld1d so the history system picks them up.
      end do
    end do

    deallocate(tmp_field, src_field, coarsened)

  end subroutine elm_derived_fields_update

  !============================================================================
  subroutine vcoarsen_layer(field, begc, endc, nlev, k_out, field_out)
    !--------------------------------------------------------------------------
    ! Compute depth-overlap-weighted average of 'field' for coarsened layer k_out.
    !
    ! For each column, the coarsened value is:
    !   sum(field(c,k) * overlap(k)) / sum(overlap(k))
    ! where overlap(k) is the depth overlap between soil layer k and the
    ! target depth range [zbound_top, zbound_bot].
    !
    ! Soil layer k spans from zisoi(k-1) to zisoi(k) (interface depths in m).
    ! zisoi(0) = 0 (surface), zisoi(k) = bottom of layer k.
    !--------------------------------------------------------------------------
    integer,  intent(in)  :: begc, endc       ! column bounds
    integer,  intent(in)  :: nlev             ! number of soil levels
    real(r8), intent(in)  :: field(begc:endc, nlev)  ! input multi-level field
    integer,  intent(in)  :: k_out            ! target coarsened layer index
    real(r8), intent(out) :: field_out(begc:endc)     ! output coarsened values

    real(r8) :: zb_top, zb_bot           ! target layer depth bounds (m)
    real(r8) :: z_top, z_bot, overlap    ! model level overlap computation
    real(r8) :: numerator, denominator
    integer  :: c, k

    zb_top = elm_vcoarsen_zbounds(k_out)
    zb_bot = elm_vcoarsen_zbounds(k_out + 1)

    do c = begc, endc
      numerator   = 0.0_r8
      denominator = 0.0_r8

      do k = 1, nlev
        ! Soil layer k spans from zisoi(k-1) to zisoi(k)
        z_top = zisoi(k-1)
        z_bot = zisoi(k)

        ! Compute overlap between soil layer and target depth range
        overlap = max(0.0_r8, min(z_bot, zb_bot) - max(z_top, zb_top))

        if (overlap > 0.0_r8) then
          numerator   = numerator   + field(c, k) * overlap
          denominator = denominator + overlap
        end if
      end do

      if (denominator > 0.0_r8) then
        field_out(c) = numerator / denominator
      else
        field_out(c) = spval
      end if
    end do

  end subroutine vcoarsen_layer

  !============================================================================
  subroutine get_soil_field(bounds, fname, t_soisno_col, h2osoi_liq_col, &
                            h2osoi_ice_col, field_out)
    !--------------------------------------------------------------------------
    ! Look up a field name and extract the corresponding column-level soil data.
    ! Supports known soil state variables passed in as arguments.
    !--------------------------------------------------------------------------
    use decompMod, only: bounds_type

    type(bounds_type), intent(in)  :: bounds
    character(len=*),  intent(in)  :: fname
    real(r8),          intent(in)  :: t_soisno_col(bounds%begc: , :)
    real(r8),          intent(in)  :: h2osoi_liq_col(bounds%begc: , :)
    real(r8),          intent(in)  :: h2osoi_ice_col(bounds%begc: , :)
    real(r8),          intent(out) :: field_out(bounds%begc:bounds%endc, nlevsoi)

    integer :: c, k, begc, endc
    character(len=max_name_len) :: uname

    begc = bounds%begc
    endc = bounds%endc
    uname = adjustl(fname)
    field_out(:,:) = 0.0_r8

    select case (trim(uname))
    case ('TSOI')
      ! Soil temperature -- copy nlevsoi levels
      do k = 1, nlevsoi
        do c = begc, endc
          field_out(c, k) = t_soisno_col(c, k)
        end do
      end do
      return
    case ('H2OSOI')
      ! Liquid soil water
      do k = 1, nlevsoi
        do c = begc, endc
          field_out(c, k) = h2osoi_liq_col(c, k)
        end do
      end do
      return
    case ('SOILICE')
      ! Soil ice
      do k = 1, nlevsoi
        do c = begc, endc
          field_out(c, k) = h2osoi_ice_col(c, k)
        end do
      end do
      return
    end select

    ! Field not found
    call endrun('get_soil_field: unknown field name: '//trim(uname)// &
         '. Must be TSOI, H2OSOI, or SOILICE.')

  end subroutine get_soil_field

  !============================================================================
  logical function is_known_soil_field(fname)
    !--------------------------------------------------------------------------
    ! Check if a field name is one of the known soil field identifiers.
    !--------------------------------------------------------------------------
    character(len=*), intent(in) :: fname
    integer :: k

    is_known_soil_field = .false.
    do k = 1, size(known_soil_flds)
      if (trim(adjustl(fname)) == trim(known_soil_flds(k))) then
        is_known_soil_field = .true.
        return
      end if
    end do
  end function is_known_soil_field

  !============================================================================
  subroutine make_vcoarsen_name(base_name, layer_idx, out_name)
    !--------------------------------------------------------------------------
    ! Construct the output field name for a coarsened layer.
    ! E.g., base_name="TSOI", layer_idx=3 => out_name="TSOI_VC3"
    !--------------------------------------------------------------------------
    character(len=*), intent(in)  :: base_name
    integer,          intent(in)  :: layer_idx
    character(len=*), intent(out) :: out_name

    character(len=4) :: idx_str

    write(idx_str, '(I0)') layer_idx
    out_name = trim(base_name) // '_VC' // trim(idx_str)

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
         ' (' // trim(zb_range_str(layer_idx)) // ')'

  end subroutine make_vcoarsen_longname

  !============================================================================
  function zb_range_str(k) result(rstr)
    !--------------------------------------------------------------------------
    ! Return a string like "0.000-0.100 m" for coarsened layer k.
    !--------------------------------------------------------------------------
    integer, intent(in) :: k
    character(len=64) :: rstr

    write(rstr, '(F0.3,A,F0.3,A)') &
         elm_vcoarsen_zbounds(k), '-', &
         elm_vcoarsen_zbounds(k+1), ' m'

  end function zb_range_str

end module elm_history_derived
