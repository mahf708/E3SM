module eam_derived
  !-------------------------------------------------------------------------------------------
  !
  ! EAM derived field wrapper.
  !
  ! Thin integration layer between EAM's physics state / history system and
  ! the shared shr_derived_mod routines. All expression parsing and evaluation
  ! math is delegated to the shared module.
  !
  ! Provides two capabilities:
  !   1. Derived fields: user-defined expressions combining state variables,
  !      constituents, physics buffer fields, and numeric constants.
  !   2. Automatic tendencies: cache previous timestep values and output
  !      time derivatives for any field (state, constituent, pbuf, or derived).
  !
  ! Configuration via namelist (eam_derived_nl):
  !   derived_fld_defs  - expression definitions, e.g. "TOTAL_WATER=Q+CLDICE+CLDLIQ+QRAIN"
  !   tend_flds         - field names for automatic tendency output
  !
  ! Expression syntax:
  !   OUTPUT_NAME=OPERAND1 op OPERAND2 op ...
  !   where op is +, -, *, / and each operand is a field name or numeric constant.
  !   Evaluation is strict left-to-right (no operator precedence).
  !   Examples:
  !     TOTAL_WATER=Q+CLDICE+CLDLIQ+QRAIN
  !     HGTsfc=PHIS/9.80616
  !     TOTAL_WATER_g=TOTAL_WATER*1000.0  (chaining: uses earlier derived field)
  !
  ! Chaining: definitions are processed in order; earlier derived fields are
  ! cached and available as inputs to later definitions.
  !
  ! Tendencies: for each field in tend_flds, outputs d{NAME}_dt = (curr - prev) / dt
  ! each physics timestep. The first timestep stores the initial value and outputs zero.
  !
  ! Usage from physpkg.F90:
  !   call eam_derived_readnl(nlfile)     ! during namelist reading
  !   call eam_derived_register()         ! during phys_init / phys_register
  !   call eam_derived_write(state, pbuf2d)  ! during physics timestep
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,    only: r8 => shr_kind_r8
  use shr_derived_mod, only: shr_derived_expr_t, shr_derived_operand_t, &
                              shr_derived_parse, shr_derived_eval, &
                              shr_derived_is_number, &
                              shr_derived_max_operands, shr_derived_max_namelen, &
                              shr_derived_max_deflen
  use ppgrid,          only: pcols, pver, pverp, begchunk, endchunk
  use cam_logfile,     only: iulog
  use cam_abortutils,  only: endrun
  use spmd_utils,      only: masterproc

  implicit none
  private
  save

  public :: eam_derived_readnl
  public :: eam_derived_register
  public :: eam_derived_write

  ! Parameters
  integer, parameter :: max_derived_flds = 50
  integer, parameter :: max_tend_flds    = 100
  integer, parameter :: max_name_len     = shr_derived_max_namelen
  integer, parameter :: max_def_len      = shr_derived_max_deflen

  ! Known 3D state variable names
  integer, parameter :: n_known_state = 6
  character(len=max_name_len), parameter :: known_state_vars(n_known_state) = &
       (/ 'T     ', 'U     ', 'V     ', 'OMEGA ', 'Z3    ', 'Q     ' /)

  ! Namelist variables
  character(len=max_def_len)  :: derived_fld_defs(max_derived_flds)
  character(len=max_name_len) :: tend_flds(max_tend_flds)

  ! Parsed state
  integer :: n_derived = 0
  type(shr_derived_expr_t) :: expressions(max_derived_flds)

  integer :: n_tend = 0
  character(len=max_name_len) :: tend_names(max_tend_flds)

  ! Derived field cache for chaining (allocated during register)
  real(r8), allocatable :: derived_cache(:,:,:,:)   ! (pcols, pver, n_derived, begchunk:endchunk)

  ! Tendency tracking (allocated during register)
  real(r8), allocatable :: tend_prev(:,:,:,:)        ! (pcols, pver, n_tend, begchunk:endchunk)
  logical :: tend_initialized = .false.

  ! Flags
  logical :: has_derived = .false.
  logical :: has_tend    = .false.
  logical :: module_is_initialized = .false.

contains

  !============================================================================
  subroutine eam_derived_readnl(nlfile)
    !--------------------------------------------------------------------------
    ! Read the eam_derived_nl namelist group.
    !--------------------------------------------------------------------------
    use namelist_utils, only: find_group_name
    use units,          only: getunit, freeunit
    use spmd_utils,     only: mpicom, masterprocid, mpi_character

    character(len=*), intent(in) :: nlfile

    integer :: unitn, ierr, i

    namelist /eam_derived_nl/ derived_fld_defs, tend_flds

    ! Initialize defaults
    derived_fld_defs(:) = ''
    tend_flds(:) = ''

    if (masterproc) then
      unitn = getunit()
      open(unitn, file=trim(nlfile), status='old')
      call find_group_name(unitn, 'eam_derived_nl', status=ierr)
      if (ierr == 0) then
        read(unitn, eam_derived_nl, iostat=ierr)
        if (ierr /= 0) then
          call endrun('eam_derived_readnl: ERROR reading namelist eam_derived_nl')
        end if
      end if
      close(unitn)
      call freeunit(unitn)
    end if

    ! Broadcast to all processors
    call mpi_bcast(derived_fld_defs, max_def_len*max_derived_flds, mpi_character, &
         masterprocid, mpicom, ierr)
    call mpi_bcast(tend_flds, max_name_len*max_tend_flds, mpi_character, &
         masterprocid, mpicom, ierr)

    ! Parse derived field definitions
    n_derived = 0
    do i = 1, max_derived_flds
      if (len_trim(derived_fld_defs(i)) == 0) exit
      n_derived = n_derived + 1
      call shr_derived_parse(derived_fld_defs(i), expressions(n_derived), ierr)
      if (ierr /= 0) then
        call endrun('eam_derived_readnl: failed to parse definition: '// &
             trim(derived_fld_defs(i)))
      end if
    end do
    has_derived = (n_derived > 0)

    ! Parse tendency field names
    n_tend = 0
    do i = 1, max_tend_flds
      if (len_trim(tend_flds(i)) == 0) exit
      n_tend = n_tend + 1
      tend_names(n_tend) = adjustl(tend_flds(i))
    end do
    has_tend = (n_tend > 0)

    if (masterproc) then
      if (has_derived) then
        write(iulog,*) 'eam_derived_readnl: ', n_derived, ' derived field(s) defined'
        do i = 1, n_derived
          write(iulog,*) '  ', trim(expressions(i)%output_name), ' = ', &
               trim(expressions(i)%long_name)
        end do
      end if
      if (has_tend) then
        write(iulog,*) 'eam_derived_readnl: ', n_tend, ' tendency field(s) requested'
        do i = 1, n_tend
          write(iulog,*) '  d', trim(tend_names(i)), '_dt'
        end do
      end if
    end if

  end subroutine eam_derived_readnl

  !============================================================================
  subroutine eam_derived_register()
    !--------------------------------------------------------------------------
    ! Register derived output fields and tendency fields with the history
    ! system. Must be called during phys_register, after other addfld calls.
    !--------------------------------------------------------------------------
    use cam_history, only: addfld

    integer :: i, n
    character(len=max_name_len) :: fname
    character(len=max_def_len)  :: lname

    if (.not. has_derived .and. .not. has_tend) return

    ! Validate and register derived fields
    do i = 1, n_derived
      ! Validate each operand
      do n = 1, expressions(i)%n_operands
        if (.not. expressions(i)%operands(n)%is_constant) then
          call validate_field_name(expressions(i)%operands(n)%field_name, i)
        end if
      end do

      ! Register as 3D field on 'lev'
      call addfld(trim(expressions(i)%output_name), (/ 'lev' /), 'A', &
           'derived', trim(expressions(i)%long_name))
    end do

    ! Validate and register tendency fields
    do i = 1, n_tend
      call validate_tend_field(tend_names(i))

      ! Register tendency as 3D field: d{NAME}_dt
      fname = 'd' // trim(tend_names(i)) // '_dt'
      lname = 'd(' // trim(tend_names(i)) // ')/dt'
      call addfld(trim(fname), (/ 'lev' /), 'A', '/s', trim(lname))
    end do

    ! Allocate derived field cache for chaining
    if (has_derived) then
      allocate(derived_cache(pcols, pver, n_derived, begchunk:endchunk))
      derived_cache(:,:,:,:) = 0.0_r8
    end if

    ! Allocate tendency cache
    if (has_tend) then
      allocate(tend_prev(pcols, pver, n_tend, begchunk:endchunk))
      tend_prev(:,:,:,:) = 0.0_r8
      tend_initialized = .false.
    end if

    module_is_initialized = .true.

    if (masterproc) then
      write(iulog,*) 'eam_derived_register: registration complete'
    end if

  end subroutine eam_derived_register

  !============================================================================
  subroutine eam_derived_write(state, pbuf2d)
    !--------------------------------------------------------------------------
    ! Compute and output derived fields and tendencies for the current chunk.
    ! Called each timestep from physpkg.F90.
    !--------------------------------------------------------------------------
    use physics_types,  only: physics_state
    use physics_buffer, only: physics_buffer_desc
    use cam_history,    only: outfld
    use time_manager,   only: get_step_size

    type(physics_state),       intent(in) :: state
    type(physics_buffer_desc), pointer    :: pbuf2d(:,:)

    real(r8) :: tmp_fields(pcols, pver, shr_derived_max_operands)
    real(r8) :: result(pcols, pver)
    real(r8) :: curr_field(pcols, pver)
    real(r8) :: tend_field(pcols, pver)
    integer  :: i, n, ncol, lchnk, dt
    character(len=max_name_len) :: fname

    if (.not. has_derived .and. .not. has_tend) return

    ncol  = state%ncol
    lchnk = state%lchnk
    dt    = get_step_size()

    ! --- Step 1: Compute derived fields in definition order (enables chaining) ---
    do i = 1, n_derived
      ! Load field data for each operand
      do n = 1, expressions(i)%n_operands
        if (.not. expressions(i)%operands(n)%is_constant) then
          call get_field_or_derived(state, pbuf2d, &
               expressions(i)%operands(n)%field_name, &
               tmp_fields(:,:,n), ncol, lchnk)
        end if
      end do

      ! Evaluate expression
      call shr_derived_eval(expressions(i), tmp_fields, ncol, pver, pcols, result)

      ! Cache result for chaining and tendencies
      derived_cache(1:ncol, :, i, lchnk) = result(1:ncol, :)

      ! Output to history
      call outfld(trim(expressions(i)%output_name), result, pcols, lchnk)
    end do

    ! --- Step 2: Compute tendencies ---
    if (has_tend) then
      do i = 1, n_tend
        ! Load current field value
        call get_field_or_derived(state, pbuf2d, tend_names(i), &
             curr_field, ncol, lchnk)

        if (tend_initialized) then
          ! Compute tendency: (curr - prev) / dt
          if (dt > 0) then
            tend_field(1:ncol, :) = (curr_field(1:ncol, :) - &
                 tend_prev(1:ncol, :, i, lchnk)) / real(dt, r8)
          else
            tend_field(1:ncol, :) = 0.0_r8
          end if
        else
          ! First timestep: output zero
          tend_field(1:ncol, :) = 0.0_r8
        end if

        ! Output tendency
        fname = 'd' // trim(tend_names(i)) // '_dt'
        call outfld(trim(fname), tend_field, pcols, lchnk)

        ! Store current value for next timestep
        tend_prev(1:ncol, :, i, lchnk) = curr_field(1:ncol, :)
      end do

      ! Mark as initialized after all chunks have been processed at least once.
      ! Since this is called per-chunk, we set the flag after the first call.
      ! All chunks will output zero on the first timestep, then real tendencies after.
      if (.not. tend_initialized) tend_initialized = .true.
    end if

  end subroutine eam_derived_write

  !============================================================================
  ! Private helper routines
  !============================================================================

  subroutine get_field_or_derived(state, pbuf2d, fname, field_out, ncol, lchnk)
    !--------------------------------------------------------------------------
    ! Look up a field by name. Checks derived field cache first (for chaining),
    ! then falls back to state variables, constituents, and physics buffer.
    !--------------------------------------------------------------------------
    use physics_types,  only: physics_state
    use physics_buffer, only: physics_buffer_desc

    type(physics_state),       intent(in)  :: state
    type(physics_buffer_desc), pointer     :: pbuf2d(:,:)
    character(len=*),          intent(in)  :: fname
    real(r8),                  intent(out) :: field_out(pcols, pver)
    integer,                   intent(in)  :: ncol
    integer,                   intent(in)  :: lchnk

    integer :: i

    ! Check derived field cache (earlier definitions only)
    do i = 1, n_derived
      if (trim(fname) == trim(expressions(i)%output_name)) then
        field_out(1:ncol, :) = derived_cache(1:ncol, :, i, lchnk)
        return
      end if
    end do

    ! Fall back to state/constituent/pbuf lookup
    call get_field(state, pbuf2d, fname, field_out, ncol)

  end subroutine get_field_or_derived

  !============================================================================
  subroutine get_field(state, pbuf2d, fname, field_out, ncol)
    !--------------------------------------------------------------------------
    ! Look up a field name and extract the corresponding 2D slice.
    ! Supports state variables, constituents, and physics buffer fields.
    !--------------------------------------------------------------------------
    use physics_types,  only: physics_state
    use physics_buffer, only: physics_buffer_desc, pbuf_get_index, pbuf_get_field, &
                              pbuf_get_chunk
    use constituents,   only: cnst_get_ind

    type(physics_state),       intent(in)  :: state
    type(physics_buffer_desc), pointer     :: pbuf2d(:,:)
    character(len=*),          intent(in)  :: fname
    real(r8),                  intent(out) :: field_out(pcols, pver)
    integer,                   intent(in)  :: ncol

    integer :: idx, pbuf_idx, errcode
    character(len=max_name_len) :: uname
    real(r8), pointer :: pbuf_fld(:,:)
    type(physics_buffer_desc), pointer :: pbuf_chunk(:)

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
    end select

    ! Try constituent lookup
    call cnst_get_ind(trim(uname), idx, abrtf=.false.)
    if (idx > 0) then
      field_out(1:ncol, :) = state%q(1:ncol, :, idx)
      return
    end if

    ! Try physics buffer lookup
    errcode = -1
    pbuf_idx = pbuf_get_index(trim(uname), errcode)
    if (errcode == 0 .and. pbuf_idx > 0) then
      pbuf_chunk => pbuf_get_chunk(pbuf2d, state%lchnk)
      call pbuf_get_field(pbuf_chunk, pbuf_idx, pbuf_fld)
      field_out(1:ncol, :) = pbuf_fld(1:ncol, :)
      return
    end if

    call endrun('eam_derived: get_field: unknown field "'//trim(uname)// &
         '". Must be a state variable (T,U,V,OMEGA,Z3,Q), constituent, or physics buffer field.')

  end subroutine get_field

  !============================================================================
  subroutine validate_field_name(fname, def_index)
    !--------------------------------------------------------------------------
    ! Validate that a field name is a known state variable, constituent,
    ! physics buffer field, or a previously defined derived field.
    !--------------------------------------------------------------------------
    use constituents,   only: cnst_get_ind
    use physics_buffer, only: pbuf_get_index

    character(len=*), intent(in) :: fname
    integer,          intent(in) :: def_index   ! index of current definition (for chaining check)

    integer :: k, idx, errcode
    logical :: found
    character(len=max_name_len) :: uname

    uname = adjustl(fname)
    found = .false.

    ! Check known state variables
    do k = 1, n_known_state
      if (trim(uname) == trim(known_state_vars(k))) then
        found = .true.
        exit
      end if
    end do

    ! Check previously defined derived fields (for chaining)
    if (.not. found) then
      do k = 1, def_index - 1
        if (trim(uname) == trim(expressions(k)%output_name)) then
          found = .true.
          exit
        end if
      end do
    end if

    ! Check constituents
    if (.not. found) then
      call cnst_get_ind(trim(uname), idx, abrtf=.false.)
      found = (idx > 0)
    end if

    ! Check physics buffer
    if (.not. found) then
      errcode = -1
      idx = pbuf_get_index(trim(uname), errcode)
      found = (errcode == 0 .and. idx > 0)
    end if

    if (.not. found) then
      call endrun('eam_derived: validate_field_name: unknown field "'//trim(uname)// &
           '" in derived definition "'//trim(expressions(def_index)%output_name)// &
           '". Must be a state variable, constituent, physics buffer field, '// &
           'or a previously defined derived field.')
    end if

  end subroutine validate_field_name

  !============================================================================
  subroutine validate_tend_field(fname)
    !--------------------------------------------------------------------------
    ! Validate that a tendency field name is a known field or derived field.
    !--------------------------------------------------------------------------
    use constituents,   only: cnst_get_ind
    use physics_buffer, only: pbuf_get_index

    character(len=*), intent(in) :: fname

    integer :: k, idx, errcode
    logical :: found
    character(len=max_name_len) :: uname

    uname = adjustl(fname)
    found = .false.

    ! Check known state variables
    do k = 1, n_known_state
      if (trim(uname) == trim(known_state_vars(k))) then
        found = .true.
        exit
      end if
    end do

    ! Check derived fields
    if (.not. found) then
      do k = 1, n_derived
        if (trim(uname) == trim(expressions(k)%output_name)) then
          found = .true.
          exit
        end if
      end do
    end if

    ! Check constituents
    if (.not. found) then
      call cnst_get_ind(trim(uname), idx, abrtf=.false.)
      found = (idx > 0)
    end if

    ! Check physics buffer
    if (.not. found) then
      errcode = -1
      idx = pbuf_get_index(trim(uname), errcode)
      found = (errcode == 0 .and. idx > 0)
    end if

    if (.not. found) then
      call endrun('eam_derived: validate_tend_field: unknown field "'//trim(uname)// &
           '". Must be a state variable, constituent, physics buffer field, or derived field.')
    end if

  end subroutine validate_tend_field

end module eam_derived
