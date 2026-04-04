module elm_fme_vcoarsen
  !-------------------------------------------------------------------------------------------
  !
  ! ELM wrapper for FME (Full Model Emulation) vertical coarsening.
  !
  ! Provides overlap-weighted depth averaging of multi-level soil fields
  ! onto a coarser set of depth layers using the shared shr_vcoarsen_mod.
  !
  ! Configuration via namelist (elm_fme_vcoarsen_nl):
  !   elm_vcoarsen_zbounds - depth boundaries (m) defining coarsened layers
  !   elm_vcoarsen_flds    - field names to vertically coarsen (e.g., 'TSOI','H2OSOI')
  !
  ! Usage:
  !   call elm_fme_vcoarsen_readnl(nlfile)      ! during namelist reading (controlMod)
  !   call elm_fme_vcoarsen_register(bounds)    ! during init, before hist_htapes_build
  !   call elm_fme_vcoarsen_update(bounds)      ! during driver timestep, before hist_update_hbuf
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,     only: r8 => shr_kind_r8
  use shr_vcoarsen_mod, only: shr_vcoarsen_avg
  use elm_varctl,       only: iulog
  use elm_varpar,       only: nlevsoi, nlevgrnd
  use elm_varcon,       only: spval, zisoi
  use abortutils,       only: endrun
  use spmdMod,          only: masterproc, mpicom, MPI_REAL8, MPI_INTEGER, MPI_CHARACTER
  use decompMod,        only: bounds_type

  implicit none
  private
  save

  public :: elm_fme_vcoarsen_readnl
  public :: elm_fme_vcoarsen_register
  public :: elm_fme_vcoarsen_update

  ! Parameters
  integer, parameter :: max_zbounds = 51
  integer, parameter :: max_flds    = 100
  integer, parameter :: max_name_len = 64

  ! Namelist variables
  real(r8) :: elm_vcoarsen_zbounds(max_zbounds)
  character(len=max_name_len) :: elm_vcoarsen_flds(max_flds)

  ! Parsed state
  integer :: n_vcoarsen_levs = 0
  integer :: n_vcoarsen_flds = 0
  character(len=max_name_len), allocatable :: vcoarsen_fld_names(:)

  ! Persistent output arrays for history field pointers (column-level)
  ! Shape: (begc:endc, n_vcoarsen_levs, n_vcoarsen_flds)
  ! Flattened to 2D for hist_addfld1d: (begc:endc) per output field
  real(r8), allocatable, target :: vcoarsen_data(:,:)  ! (begc:endc, n_levs*n_flds)

  logical :: module_is_initialized = .false.
  logical :: has_vcoarsen = .false.

contains

  !============================================================================
  subroutine elm_fme_vcoarsen_readnl(nlfile)
    use shr_nl_mod, only: shr_nl_find_group_name
    use fileutils,  only: getavu, relavu

    character(len=*), intent(in) :: nlfile

    integer :: unitn, ierr, i
    logical :: lexist

    namelist /elm_fme_vcoarsen_nl/ elm_vcoarsen_zbounds, elm_vcoarsen_flds

    elm_vcoarsen_zbounds(:) = -1.0_r8
    elm_vcoarsen_flds(:) = ''

    if (masterproc) then
      inquire(file=trim(nlfile), exist=lexist)
      if (lexist) then
        unitn = getavu()
        open(unitn, file=trim(nlfile), status='old')
        call shr_nl_find_group_name(unitn, 'elm_fme_vcoarsen_nl', status=ierr)
        if (ierr == 0) then
          read(unitn, elm_fme_vcoarsen_nl, iostat=ierr)
          if (ierr /= 0) then
            call endrun('elm_fme_vcoarsen_readnl: ERROR reading namelist')
          end if
        end if
        close(unitn)
        call relavu(unitn)
      end if
    end if

    call mpi_bcast(elm_vcoarsen_zbounds, max_zbounds, MPI_REAL8, 0, mpicom, ierr)
    call mpi_bcast(elm_vcoarsen_flds, max_name_len*max_flds, MPI_CHARACTER, 0, mpicom, ierr)

    ! Count valid bounds (n boundaries = n_levels + 1)
    n_vcoarsen_levs = 0
    do i = 2, max_zbounds
      if (elm_vcoarsen_zbounds(i) < 0.0_r8) exit
      n_vcoarsen_levs = n_vcoarsen_levs + 1
    end do

    ! Count and store valid field names
    n_vcoarsen_flds = 0
    do i = 1, max_flds
      if (len_trim(elm_vcoarsen_flds(i)) > 0) then
        n_vcoarsen_flds = n_vcoarsen_flds + 1
      end if
    end do

    if (n_vcoarsen_flds > 0) then
      allocate(vcoarsen_fld_names(n_vcoarsen_flds))
      n_vcoarsen_flds = 0
      do i = 1, max_flds
        if (len_trim(elm_vcoarsen_flds(i)) > 0) then
          n_vcoarsen_flds = n_vcoarsen_flds + 1
          vcoarsen_fld_names(n_vcoarsen_flds) = elm_vcoarsen_flds(i)
        end if
      end do
    end if

    has_vcoarsen = (n_vcoarsen_levs > 0 .and. n_vcoarsen_flds > 0)

    if (masterproc .and. has_vcoarsen) then
      write(iulog, *) 'elm_fme_vcoarsen: ', n_vcoarsen_levs, ' coarsened levels, ', &
           n_vcoarsen_flds, ' fields'
      do i = 1, n_vcoarsen_levs + 1
        write(iulog, *) '  depth bound ', i, ': ', elm_vcoarsen_zbounds(i), ' m'
      end do
      do i = 1, n_vcoarsen_flds
        write(iulog, *) '  field: ', trim(vcoarsen_fld_names(i))
      end do
    end if

    module_is_initialized = .true.

  end subroutine elm_fme_vcoarsen_readnl

  !============================================================================
  subroutine elm_fme_vcoarsen_register(bounds)
    !--------------------------------------------------------------------------
    ! Register vertically coarsened output fields with ELM's history system.
    ! Creates one output field per (source_field, coarsened_level) pair.
    ! Must be called BEFORE hist_htapes_build().
    !--------------------------------------------------------------------------
    use histFileMod, only: hist_addfld1d

    type(bounds_type), intent(in) :: bounds

    integer :: i, k, idx, begc, endc, n_total
    character(len=128) :: out_name, out_long

    if (.not. has_vcoarsen) return

    begc = bounds%begc
    endc = bounds%endc
    n_total = n_vcoarsen_levs * n_vcoarsen_flds

    ! Allocate persistent storage for coarsened field values
    allocate(vcoarsen_data(begc:endc, n_total))
    vcoarsen_data(:,:) = spval

    ! Register each coarsened level as a separate 1D column-level field
    idx = 0
    do i = 1, n_vcoarsen_flds
      do k = 1, n_vcoarsen_levs
        idx = idx + 1
        write(out_name, '(A,A,I0)') trim(vcoarsen_fld_names(i)), '_VC', k
        write(out_long, '(A,A,I0,A,F0.1,A,F0.1,A)') &
             'FME vcoarsen: ', trim(vcoarsen_fld_names(i)), k, &
             ' (', elm_vcoarsen_zbounds(k), '-', elm_vcoarsen_zbounds(k+1), ' m)'

        if (masterproc) then
          write(iulog, *) '  registering: ', trim(out_name)
        end if

        call hist_addfld1d( &
             fname=trim(out_name), &
             units='coarsened', &
             avgflag='A', &
             long_name=trim(out_long), &
             ptr_col=vcoarsen_data(:,idx), &
             default='inactive')
      end do
    end do

  end subroutine elm_fme_vcoarsen_register

  !============================================================================
  subroutine elm_fme_vcoarsen_update(bounds)
    !--------------------------------------------------------------------------
    ! Compute vertically coarsened fields each timestep.
    ! Uses shr_vcoarsen_avg with ELM's zisoi (soil interface depths)
    ! as the coordinate interfaces.
    !--------------------------------------------------------------------------
    use ColumnDataType, only: col_es, col_ws

    type(bounds_type), intent(in) :: bounds

    integer :: c, ifld, k, idx, begc, endc
    real(r8) :: coord_iface(0:nlevgrnd)
    real(r8) :: field_col(nlevgrnd)
    real(r8) :: coarsened_out(n_vcoarsen_levs)
    real(r8), pointer :: src_2d(:,:)

    if (.not. has_vcoarsen) return

    begc = bounds%begc
    endc = bounds%endc

    ! Build depth interface array from ELM soil interfaces (constant across columns)
    ! zisoi(0:nlevgrnd) gives the interface depths in meters
    do k = 0, nlevgrnd
      coord_iface(k) = zisoi(k)
    end do

    do ifld = 1, n_vcoarsen_flds
      ! Map field name to ELM data pointer
      nullify(src_2d)
      select case (trim(vcoarsen_fld_names(ifld)))
      case ('TSOI')
        src_2d => col_es%t_soisno(:,1:nlevgrnd)
      case ('H2OSOI')
        src_2d => col_ws%h2osoi_vol(:,1:nlevgrnd)
      case ('SOILICE')
        src_2d => col_ws%h2osoi_ice(:,1:nlevgrnd)
      case default
        if (masterproc) then
          write(iulog,*) 'elm_fme_vcoarsen_update: unknown field ', &
               trim(vcoarsen_fld_names(ifld)), ', filling with spval'
        end if
        idx = (ifld - 1) * n_vcoarsen_levs
        vcoarsen_data(begc:endc, idx+1:idx+n_vcoarsen_levs) = spval
        cycle
      end select

      ! Compute vertical coarsening for each column
      do c = begc, endc
        field_col(1:nlevgrnd) = src_2d(c, 1:nlevgrnd)

        call shr_vcoarsen_avg(field_col, coord_iface, nlevgrnd, &
             elm_vcoarsen_zbounds, n_vcoarsen_levs, spval, coarsened_out)

        idx = (ifld - 1) * n_vcoarsen_levs
        do k = 1, n_vcoarsen_levs
          vcoarsen_data(c, idx + k) = coarsened_out(k)
        end do
      end do
    end do

  end subroutine elm_fme_vcoarsen_update

end module elm_fme_vcoarsen
