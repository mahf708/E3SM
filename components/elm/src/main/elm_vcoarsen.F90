module elm_vcoarsen
  !-------------------------------------------------------------------------------------------
  !
  ! ELM vertical coarsening wrapper.
  !
  ! Thin integration layer between ELM's soil state / history system and
  ! the shared shr_vcoarsen_mod routines. All vertical coarsening math is
  ! delegated to the shared module.
  !
  ! Supports three modes:
  !   1. Overlap-weighted averaging onto coarser soil depth layers (dz-weighted)
  !   2. Soil level selection by index (e.g., TSOI_at_L3)
  !   3. Soil level selection by nearest depth value (e.g., TSOI_at_D0.5 for 0.5 m)
  !
  ! Configuration via namelist (elm_vcoarsen_nl):
  !   elm_vcoarsen_zbounds         - depth boundaries (m), surface to bottom
  !   elm_vcoarsen_avg_flds        - fields to average onto coarsened layers
  !   elm_vcoarsen_select_levs     - level indices for selection
  !   elm_vcoarsen_select_lev_flds - fields for level selection
  !   elm_vcoarsen_select_depths   - depth values (m) for nearest-level selection
  !   elm_vcoarsen_select_depth_flds - fields for depth selection
  !
  ! Use 'all' as the sole entry in any field list to apply to all known soil fields.
  !
  ! Usage:
  !   call elm_vcoarsen_readnl(nlfile)
  !   call elm_vcoarsen_register(bounds)
  !   call elm_vcoarsen_update(bounds, col_es, col_ws)
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,  only: r8 => shr_kind_r8
  use elm_varctl,    only: iulog
  use abortutils,    only: endrun
  use spmdMod,       only: masterproc, mpicom, MPI_REAL8, MPI_INTEGER, MPI_CHARACTER
  use elm_varpar,    only: nlevsoi
  use elm_varcon,    only: spval, zisoi, zsoi

  implicit none
  private
  save

  public :: elm_vcoarsen_readnl
  public :: elm_vcoarsen_register
  public :: elm_vcoarsen_update

  ! Parameters
  integer, parameter :: max_zbounds      = 51
  integer, parameter :: max_flds         = 100
  integer, parameter :: max_select_vals  = 50
  integer, parameter :: max_name_len     = 64
  integer, parameter :: max_fname_len    = 80
  integer, parameter :: max_out_flds     = 500  ! max total output fields

  ! Known soil field names (all column x level fields from column_energy_state
  ! and column_water_state that span soil levels)
  integer, parameter :: n_known_soil = 11
  character(len=max_name_len), parameter :: known_soil_flds(n_known_soil) = &
       (/ 'TSOI        ', &  ! soil/snow temperature (K)
          'H2OSOI_LIQ  ', &  ! liquid water (kg/m2)
          'H2OSOI_ICE  ', &  ! ice lens (kg/m2)
          'H2OSOI_VOL  ', &  ! volumetric soil water (m3/m3)
          'H2OSOI_LIQVOL', & ! volumetric liquid water (m3/m3)
          'H2OSOI_ICEVOL', & ! volumetric ice content (m3/m3)
          'SMP_L       ', &  ! liquid phase soil matric potential
          'SOILP       ', &  ! soil pressure
          'AIR_VOL     ', &  ! air filled porosity
          'EXCESS_ICE  ', &  ! excess ground ice
          'FRAC_ICEOLD ' /)  ! fraction ice relative to total water

  ! Namelist variables
  real(r8) :: elm_vcoarsen_zbounds(max_zbounds)
  character(len=max_name_len) :: elm_vcoarsen_avg_flds(max_flds)
  integer  :: elm_vcoarsen_select_levs(max_select_vals)
  character(len=max_name_len) :: elm_vcoarsen_select_lev_flds(max_flds)
  real(r8) :: elm_vcoarsen_select_depths(max_select_vals)
  character(len=max_name_len) :: elm_vcoarsen_select_depth_flds(max_flds)

  ! Parsed counts
  integer :: n_avg_levs        = 0
  integer :: n_avg_flds        = 0
  integer :: n_sel_levs        = 0
  integer :: n_sel_lev_flds    = 0
  integer :: n_sel_depths      = 0
  integer :: n_sel_depth_flds  = 0

  ! Persistent output arrays for history system (pointers required by hist_addfld1d)
  integer :: n_out_flds = 0
  real(r8), allocatable, target :: out_data(:,:)  ! (begc:endc, n_out_flds)

  ! Flags
  logical :: has_avg       = .false.
  logical :: has_sel_lev   = .false.
  logical :: has_sel_depth = .false.

contains

  !============================================================================
  subroutine elm_vcoarsen_readnl(nlfile)
    use fileutils,   only: getavu, relavu
    use shr_nl_mod,  only: shr_nl_find_group_name

    character(len=*), intent(in) :: nlfile

    integer :: unitn, ierr, i

    namelist /elm_vcoarsen_nl/ elm_vcoarsen_zbounds, elm_vcoarsen_avg_flds, &
         elm_vcoarsen_select_levs, elm_vcoarsen_select_lev_flds, &
         elm_vcoarsen_select_depths, elm_vcoarsen_select_depth_flds

    ! Initialize defaults
    elm_vcoarsen_zbounds(:)          = -1.0_r8
    elm_vcoarsen_avg_flds(:)         = ''
    elm_vcoarsen_select_levs(:)      = -1
    elm_vcoarsen_select_lev_flds(:)  = ''
    elm_vcoarsen_select_depths(:)    = -1.0_r8
    elm_vcoarsen_select_depth_flds(:) = ''

    if (masterproc) then
      unitn = getavu()
      open(unitn, file=trim(nlfile), status='old', iostat=ierr)
      if (ierr /= 0) then
        call relavu(unitn)
        return
      end if
      call shr_nl_find_group_name(unitn, 'elm_vcoarsen_nl', status=ierr)
      if (ierr == 0) then
        read(unitn, elm_vcoarsen_nl, iostat=ierr)
        if (ierr /= 0) then
          call endrun('elm_vcoarsen_readnl: ERROR reading namelist elm_vcoarsen_nl')
        end if
      end if
      close(unitn)
      call relavu(unitn)
    end if

    ! Broadcast
    call mpi_bcast(elm_vcoarsen_zbounds, max_zbounds, MPI_REAL8, 0, mpicom, ierr)
    call mpi_bcast(elm_vcoarsen_avg_flds, max_name_len*max_flds, MPI_CHARACTER, 0, mpicom, ierr)
    call mpi_bcast(elm_vcoarsen_select_levs, max_select_vals, MPI_INTEGER, 0, mpicom, ierr)
    call mpi_bcast(elm_vcoarsen_select_lev_flds, max_name_len*max_flds, MPI_CHARACTER, 0, mpicom, ierr)
    call mpi_bcast(elm_vcoarsen_select_depths, max_select_vals, MPI_REAL8, 0, mpicom, ierr)
    call mpi_bcast(elm_vcoarsen_select_depth_flds, max_name_len*max_flds, MPI_CHARACTER, 0, mpicom, ierr)

    ! Parse depth bounds
    n_avg_levs = 0
    do i = 2, max_zbounds
      if (elm_vcoarsen_zbounds(i) < 0.0_r8) exit
      n_avg_levs = n_avg_levs + 1
    end do
    has_avg = (n_avg_levs > 0)

    if (has_avg) then
      if (elm_vcoarsen_zbounds(1) < 0.0_r8) then
        call endrun('elm_vcoarsen_readnl: elm_vcoarsen_zbounds(1) must be non-negative')
      end if
      do i = 1, n_avg_levs
        if (elm_vcoarsen_zbounds(i+1) <= elm_vcoarsen_zbounds(i)) then
          call endrun('elm_vcoarsen_readnl: elm_vcoarsen_zbounds must be strictly increasing')
        end if
      end do
    end if

    call count_and_expand_flds(elm_vcoarsen_avg_flds, n_avg_flds)

    ! Parse level selection indices
    n_sel_levs = 0
    do i = 1, max_select_vals
      if (elm_vcoarsen_select_levs(i) < 0) exit
      n_sel_levs = n_sel_levs + 1
    end do
    has_sel_lev = (n_sel_levs > 0)
    call count_and_expand_flds(elm_vcoarsen_select_lev_flds, n_sel_lev_flds)

    ! Parse depth selection values
    n_sel_depths = 0
    do i = 1, max_select_vals
      if (elm_vcoarsen_select_depths(i) < 0.0_r8) exit
      n_sel_depths = n_sel_depths + 1
    end do
    has_sel_depth = (n_sel_depths > 0)
    call count_and_expand_flds(elm_vcoarsen_select_depth_flds, n_sel_depth_flds)

    ! Count total output fields
    n_out_flds = n_avg_flds * n_avg_levs + n_sel_lev_flds * n_sel_levs + &
         n_sel_depth_flds * n_sel_depths

    if (masterproc) then
      if (has_avg) then
        write(iulog,*) 'elm_vcoarsen_readnl: averaging enabled, ', n_avg_levs, &
             ' layers, ', n_avg_flds, ' fields'
      end if
      if (has_sel_lev) then
        write(iulog,*) 'elm_vcoarsen_readnl: level selection enabled, ', n_sel_levs, &
             ' levels, ', n_sel_lev_flds, ' fields'
      end if
      if (has_sel_depth) then
        write(iulog,*) 'elm_vcoarsen_readnl: depth selection enabled, ', n_sel_depths, &
             ' depths, ', n_sel_depth_flds, ' fields'
      end if
    end if

  end subroutine elm_vcoarsen_readnl

  !============================================================================
  subroutine elm_vcoarsen_register(bounds)
    use decompMod,   only: bounds_type
    use histFileMod, only: hist_addfld1d

    type(bounds_type), intent(in) :: bounds

    integer :: i, k, idx, begc, endc
    character(len=max_fname_len) :: fname
    character(len=256) :: lname

    if (n_out_flds == 0) return

    begc = bounds%begc
    endc = bounds%endc

    ! Allocate persistent output storage
    allocate(out_data(begc:endc, n_out_flds))
    out_data(:,:) = spval

    ! Register all output fields, each pointing to a column in out_data
    idx = 0

    ! Averaged fields
    if (has_avg) then
      do i = 1, n_avg_flds
        call validate_soil_field(elm_vcoarsen_avg_flds(i))
        do k = 1, n_avg_levs
          idx = idx + 1
          call make_avg_name(elm_vcoarsen_avg_flds(i), k, fname)
          write(lname, '(A,A,I0,A,F0.3,A,F0.3,A)') &
               trim(elm_vcoarsen_avg_flds(i)), ' vcoarsen layer ', k, &
               ' (', elm_vcoarsen_zbounds(k), '-', elm_vcoarsen_zbounds(k+1), ' m)'
          call hist_addfld1d( &
               fname=trim(fname), units='varies', avgflag='A', &
               long_name=trim(lname), type1d_out='column', &
               ptr_col=out_data(:, idx), default='inactive')
        end do
      end do
    end if

    ! Level-selected fields
    if (has_sel_lev) then
      do i = 1, n_sel_lev_flds
        call validate_soil_field(elm_vcoarsen_select_lev_flds(i))
        do k = 1, n_sel_levs
          idx = idx + 1
          call make_sel_lev_name(elm_vcoarsen_select_lev_flds(i), &
               elm_vcoarsen_select_levs(k), fname)
          write(lname, '(A,A,I0)') trim(elm_vcoarsen_select_lev_flds(i)), &
               ' at soil level ', elm_vcoarsen_select_levs(k)
          call hist_addfld1d( &
               fname=trim(fname), units='varies', avgflag='A', &
               long_name=trim(lname), type1d_out='column', &
               ptr_col=out_data(:, idx), default='inactive')
        end do
      end do
    end if

    ! Depth-selected fields
    if (has_sel_depth) then
      do i = 1, n_sel_depth_flds
        call validate_soil_field(elm_vcoarsen_select_depth_flds(i))
        do k = 1, n_sel_depths
          idx = idx + 1
          call make_sel_depth_name(elm_vcoarsen_select_depth_flds(i), &
               elm_vcoarsen_select_depths(k), fname)
          write(lname, '(A,A,F0.3,A)') trim(elm_vcoarsen_select_depth_flds(i)), &
               ' at nearest soil level to ', elm_vcoarsen_select_depths(k), ' m'
          call hist_addfld1d( &
               fname=trim(fname), units='varies', avgflag='A', &
               long_name=trim(lname), type1d_out='column', &
               ptr_col=out_data(:, idx), default='inactive')
        end do
      end do
    end if

    if (masterproc) then
      write(iulog,*) 'elm_vcoarsen_register: registered ', n_out_flds, ' output fields'
    end if

  end subroutine elm_vcoarsen_register

  !============================================================================
  subroutine elm_vcoarsen_update(bounds, col_es, col_ws)
    use decompMod,        only: bounds_type
    use ColumnDataType,   only: column_energy_state, column_water_state
    use shr_vcoarsen_mod, only: shr_vcoarsen_avg_cols, shr_vcoarsen_select_index, &
                                shr_vcoarsen_select_nearest

    type(bounds_type),         intent(in) :: bounds
    type(column_energy_state), intent(in) :: col_es
    type(column_water_state),  intent(in) :: col_ws

    integer :: begc, endc, ncol, nlev, idx
    real(r8), allocatable :: src_field(:,:)    ! (ncol, nlev)
    real(r8), allocatable :: coarsened(:,:)    ! (ncol, n_avg_levs)
    real(r8), allocatable :: selected(:)       ! (ncol)
    real(r8), allocatable :: coord_iface(:,:)  ! (ncol, nlev+1)
    real(r8), allocatable :: coord_mid(:,:)    ! (ncol, nlev)
    integer,  allocatable :: nlev_max(:)       ! (ncol)
    integer :: i, k, c

    if (n_out_flds == 0) return

    begc = bounds%begc
    endc = bounds%endc
    ncol = endc - begc + 1
    nlev = nlevsoi

    allocate(src_field(ncol, nlev))
    allocate(coord_iface(ncol, nlev+1))
    allocate(coord_mid(ncol, nlev))
    allocate(nlev_max(ncol))

    ! ELM soil interfaces are the same for all columns
    do c = 1, ncol
      coord_iface(c, 1) = zisoi(0)
      do k = 1, nlev
        coord_iface(c, k+1) = zisoi(k)
        coord_mid(c, k) = zsoi(k)
      end do
    end do
    nlev_max(:) = nlev

    idx = 0

    ! --- Overlap-weighted averaging ---
    if (has_avg) then
      allocate(coarsened(ncol, n_avg_levs))

      do i = 1, n_avg_flds
        call get_soil_field(begc, endc, elm_vcoarsen_avg_flds(i), &
             col_es, col_ws, src_field, ncol, nlev)

        call shr_vcoarsen_avg_cols(src_field, coord_iface, ncol, nlev, &
             elm_vcoarsen_zbounds(1:n_avg_levs+1), n_avg_levs, spval, coarsened)

        do k = 1, n_avg_levs
          idx = idx + 1
          do c = 1, ncol
            out_data(begc + c - 1, idx) = coarsened(c, k)
          end do
        end do
      end do

      deallocate(coarsened)
    end if

    ! --- Level index selection ---
    if (has_sel_lev) then
      allocate(selected(ncol))

      do i = 1, n_sel_lev_flds
        call get_soil_field(begc, endc, elm_vcoarsen_select_lev_flds(i), &
             col_es, col_ws, src_field, ncol, nlev)

        do k = 1, n_sel_levs
          idx = idx + 1
          call shr_vcoarsen_select_index(src_field, ncol, nlev, &
               elm_vcoarsen_select_levs(k), nlev_max, spval, selected)
          do c = 1, ncol
            out_data(begc + c - 1, idx) = selected(c)
          end do
        end do
      end do

      deallocate(selected)
    end if

    ! --- Nearest depth selection ---
    if (has_sel_depth) then
      allocate(selected(ncol))

      do i = 1, n_sel_depth_flds
        call get_soil_field(begc, endc, elm_vcoarsen_select_depth_flds(i), &
             col_es, col_ws, src_field, ncol, nlev)

        do k = 1, n_sel_depths
          idx = idx + 1
          call shr_vcoarsen_select_nearest(src_field, coord_mid, ncol, nlev, &
               elm_vcoarsen_select_depths(k), nlev_max, spval, selected)
          do c = 1, ncol
            out_data(begc + c - 1, idx) = selected(c)
          end do
        end do
      end do

      deallocate(selected)
    end if

    deallocate(src_field, coord_iface, coord_mid, nlev_max)

  end subroutine elm_vcoarsen_update

  !============================================================================
  ! Private helpers
  !============================================================================

  subroutine get_soil_field(begc, endc, fname, col_es, col_ws, field_out, ncol, nlev)
    use ColumnDataType, only: column_energy_state, column_water_state

    integer,                    intent(in)  :: begc, endc
    character(len=*),           intent(in)  :: fname
    type(column_energy_state),  intent(in)  :: col_es
    type(column_water_state),   intent(in)  :: col_ws
    real(r8),                   intent(out) :: field_out(ncol, nlev)
    integer,                    intent(in)  :: ncol, nlev

    integer :: c, k
    character(len=max_name_len) :: uname
    real(r8), pointer :: src(:,:)

    uname = adjustl(fname)
    nullify(src)

    ! Map field name to data pointer
    select case (trim(uname))
    case ('TSOI')
      src => col_es%t_soisno
    case ('H2OSOI_LIQ')
      src => col_ws%h2osoi_liq
    case ('H2OSOI_ICE')
      src => col_ws%h2osoi_ice
    case ('H2OSOI_VOL')
      src => col_ws%h2osoi_vol
    case ('H2OSOI_LIQVOL')
      src => col_ws%h2osoi_liqvol
    case ('H2OSOI_ICEVOL')
      src => col_ws%h2osoi_icevol
    case ('SMP_L')
      src => col_ws%smp_l
    case ('SOILP')
      src => col_ws%soilp
    case ('AIR_VOL')
      src => col_ws%air_vol
    case ('EXCESS_ICE')
      src => col_ws%excess_ice
    case ('FRAC_ICEOLD')
      src => col_ws%frac_iceold
    case default
      call endrun('elm_vcoarsen: get_soil_field: unknown field: '//trim(uname))
    end select

    ! Copy from (begc:endc, 1:nlev) to (1:ncol, 1:nlev)
    do k = 1, nlev
      do c = 1, ncol
        field_out(c, k) = src(begc + c - 1, k)
      end do
    end do

  end subroutine get_soil_field

  !============================================================================
  subroutine validate_soil_field(fname)
    character(len=*), intent(in) :: fname
    integer :: k
    character(len=max_name_len) :: uname

    uname = adjustl(fname)
    do k = 1, n_known_soil
      if (trim(uname) == trim(known_soil_flds(k))) return
    end do
    call endrun('elm_vcoarsen: unknown soil field "'//trim(uname)// &
         '". Must be TSOI, H2OSOI, or SOILICE.')

  end subroutine validate_soil_field

  !============================================================================
  subroutine count_and_expand_flds(fld_list, n_flds)
    character(len=max_name_len), intent(inout) :: fld_list(max_flds)
    integer, intent(out) :: n_flds

    integer :: i

    n_flds = 0
    if (trim(adjustl(fld_list(1))) == 'all') then
      do i = 1, n_known_soil
        fld_list(i) = known_soil_flds(i)
      end do
      fld_list(n_known_soil + 1:) = ''
      n_flds = n_known_soil
      return
    end if

    do i = 1, max_flds
      if (len_trim(fld_list(i)) == 0) exit
      n_flds = n_flds + 1
    end do

  end subroutine count_and_expand_flds

  !============================================================================
  subroutine make_avg_name(base_name, layer_idx, out_name)
    character(len=*), intent(in)  :: base_name
    integer,          intent(in)  :: layer_idx
    character(len=*), intent(out) :: out_name
    character(len=4) :: idx_str

    write(idx_str, '(I0)') layer_idx
    out_name = trim(base_name) // '_' // trim(idx_str)
  end subroutine make_avg_name

  !============================================================================
  subroutine make_sel_lev_name(base_name, lev_idx, out_name)
    character(len=*), intent(in)  :: base_name
    integer,          intent(in)  :: lev_idx
    character(len=*), intent(out) :: out_name
    character(len=4) :: idx_str

    write(idx_str, '(I0)') lev_idx
    out_name = trim(base_name) // '_at_L' // trim(idx_str)
  end subroutine make_sel_lev_name

  !============================================================================
  subroutine make_sel_depth_name(base_name, depth_m, out_name)
    character(len=*), intent(in)  :: base_name
    real(r8),         intent(in)  :: depth_m
    character(len=*), intent(out) :: out_name
    character(len=16) :: dstr
    integer :: int_depth

    int_depth = nint(depth_m)
    if (abs(depth_m - real(int_depth, r8)) < 0.001_r8) then
      write(dstr, '(I0)') int_depth
    else
      write(dstr, '(F0.3)') depth_m
    end if
    out_name = trim(base_name) // '_at_D' // trim(dstr)
  end subroutine make_sel_depth_name

end module elm_vcoarsen
