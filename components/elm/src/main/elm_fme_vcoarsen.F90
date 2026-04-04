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
  !   elm_vcoarsen_flds    - field names to vertically coarsen
  !
  ! Usage from controlMod / elm_driver:
  !   call elm_fme_vcoarsen_readnl(nlfile)    ! during namelist reading
  !   call elm_fme_vcoarsen_register()        ! during init
  !   call elm_fme_vcoarsen_update()          ! during driver timestep
  !
  !-------------------------------------------------------------------------------------------

  use shr_kind_mod,     only: r8 => shr_kind_r8
  use shr_vcoarsen_mod, only: shr_vcoarsen_avg
  use elm_varctl,       only: iulog
  use elm_varpar,       only: nlevsoi
  use elm_varcon,       only: zisoi
  use abortutils,       only: endrun
  use spmdMod,          only: masterproc, mpicom, MPI_REAL8, MPI_INTEGER, MPI_CHARACTER

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

    ! Count valid bounds
    n_vcoarsen_levs = 0
    do i = 2, max_zbounds
      if (elm_vcoarsen_zbounds(i) < 0.0_r8) exit
      n_vcoarsen_levs = n_vcoarsen_levs + 1
    end do

    ! Count valid field names
    n_vcoarsen_flds = 0
    do i = 1, max_flds
      if (len_trim(elm_vcoarsen_flds(i)) > 0) then
        n_vcoarsen_flds = n_vcoarsen_flds + 1
      end if
    end do

    has_vcoarsen = (n_vcoarsen_levs > 0 .and. n_vcoarsen_flds > 0)

    if (masterproc .and. has_vcoarsen) then
      write(iulog, *) 'elm_fme_vcoarsen: ', n_vcoarsen_levs, ' coarsened levels, ', &
           n_vcoarsen_flds, ' fields'
    end if

    module_is_initialized = .true.

  end subroutine elm_fme_vcoarsen_readnl

  !============================================================================
  subroutine elm_fme_vcoarsen_register()
    ! Placeholder for registering coarsened output fields with histFileMod.
    ! Full integration deferred until ELM component wrapper is connected.

    if (.not. has_vcoarsen) return

    if (masterproc) then
      write(iulog, *) 'elm_fme_vcoarsen: registration placeholder (', &
           n_vcoarsen_flds, ' fields x ', n_vcoarsen_levs, ' levels)'
    end if

  end subroutine elm_fme_vcoarsen_register

  !============================================================================
  subroutine elm_fme_vcoarsen_update()
    ! Placeholder for runtime vertical coarsening evaluation.
    ! Will use shr_vcoarsen_avg with zisoi as coordinate interfaces.

    if (.not. has_vcoarsen) return

  end subroutine elm_fme_vcoarsen_update

end module elm_fme_vcoarsen
