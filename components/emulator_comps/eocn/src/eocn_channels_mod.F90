module eocn_channels_mod

  !-----------------------------------------------------------------------------
  ! Channel table for the ocean emulators EOCN knows how to drive.
  !
  ! A traced ACE-family ocean model is a black box consuming a
  ! [1, n_in, ny, nx] tensor and producing a [1, n_out, ny, nx] tensor.  EOCN
  ! needs to know which physical field sits in which channel so it can fill the
  ! input tensor from the coupler plus its own previous output, pull the fields
  ! the coupler wants back out, and name the channels in a restart file.
  !
  ! Derived from the checkpoint's own in_names/out_names, which are written to
  ! the *_metadata.yaml emitted next to the traced .pt by
  ! tools/trace_eocn_model.py.  Keep in sync with that yaml.
  !-----------------------------------------------------------------------------

  use shr_kind_mod, only: R8=>SHR_KIND_R8, CL=>SHR_KIND_CL
  use shr_sys_mod,  only: shr_sys_abort

  implicit none
  private
  save

  integer, parameter, public :: eocn_nlen = 48   ! channel name length
  integer, parameter, public :: eocn_llen = 128  ! channel long_name length
  integer, parameter, public :: eocn_ulen = 16   ! channel units length

  integer, public :: n_input_channels   = 0
  integer, public :: n_output_channels  = 0
  integer, public :: n_forcing_channels = 0

  character(len=eocn_nlen), allocatable, public :: in_names(:)
  character(len=eocn_nlen), allocatable, public :: out_names(:)
  character(len=eocn_nlen), allocatable, public :: forcing_names(:)

  ! For every input channel, the output channel it is carried forward from.
  ! Zero means EOCN has to supply it (land mask, or a coupler flux).
  integer, allocatable, public :: in_from_out(:)

  ! Named channel indices (1-based; 0 means this emulator does not have it).
  integer, public :: ix_in_landfrac = 0
  integer, public :: ix_in_ssfrac   = 0   ! sea surface fraction (static)
  integer, public :: ix_in_taux     = 0
  integer, public :: ix_in_tauy     = 0
  integer, public :: ix_in_prec     = 0   ! total surface precipitation rate
  integer, public :: ix_in_snow     = 0   ! frozen precipitation rate
  integer, public :: ix_in_flus     = 0
  integer, public :: ix_in_fsus     = 0
  integer, public :: ix_in_flds     = 0
  integer, public :: ix_in_fsds     = 0
  integer, public :: ix_in_lhflx    = 0
  integer, public :: ix_in_shflx    = 0

  integer, public :: ix_out_sst     = 0
  integer, public :: ix_out_ssh     = 0
  integer, public :: ix_out_sss     = 0   ! surface layer salinity
  integer, public :: ix_out_uvel    = 0   ! surface layer zonal velocity
  integer, public :: ix_out_vvel    = 0   ! surface layer meridional velocity
  integer, public :: ix_out_sifrac  = 0   ! emulator sea ice fraction
  integer, public :: ix_out_icevol  = 0   ! emulator sea ice volume

  ! Emulator timestep in seconds.  Samudra steps 5 days at a time.
  integer, public :: eocn_model_dt = 5 * 24 * 60 * 60

  character(len=CL), public :: eocn_emulator_name = 'unset'

  public :: eocn_channels_init
  public :: eocn_channels_final
  public :: eocn_channel_index
  public :: eocn_channel_metadata

contains

  !===============================================================================
  subroutine eocn_channels_init(emulator, logunit)

    character(len=*), intent(in) :: emulator
    integer,          intent(in) :: logunit

    integer :: k
    character(len=*), parameter :: subname = '(eocn_channels_init) '

    select case (trim(emulator))
    case ('SamudrACE-E3SMv3')
      call set_table_samudrace()
    case default
      call shr_sys_abort(trim(subname)//' ERROR: unknown eocn_emulator "'// &
           trim(emulator)//'"')
    end select

    eocn_emulator_name = trim(emulator)

    allocate(in_from_out(n_input_channels))
    in_from_out(:) = 0
    do k = 1, n_input_channels
      in_from_out(k) = eocn_channel_index(out_names, in_names(k))
    end do

    ix_in_landfrac = eocn_channel_index(in_names, 'LANDFRAC')
    ix_in_ssfrac   = eocn_channel_index(in_names, 'sea_surface_fraction')
    ix_in_taux     = eocn_channel_index(in_names, 'TAUX')
    ix_in_tauy     = eocn_channel_index(in_names, 'TAUY')
    ix_in_prec     = eocn_channel_index(in_names, 'surface_precipitation_rate')
    ix_in_snow     = eocn_channel_index(in_names, 'frozen_precipitation_rate')
    ix_in_flus     = eocn_channel_index(in_names, 'FLUS')
    ix_in_fsus     = eocn_channel_index(in_names, 'FSUS')
    ix_in_flds     = eocn_channel_index(in_names, 'FLDS')
    ix_in_fsds     = eocn_channel_index(in_names, 'FSDS')
    ix_in_lhflx    = eocn_channel_index(in_names, 'LHFLX')
    ix_in_shflx    = eocn_channel_index(in_names, 'SHFLX')

    ix_out_sst     = eocn_channel_index(out_names, 'sst')
    ix_out_ssh     = eocn_channel_index(out_names, 'ssh')
    ix_out_sss     = eocn_channel_index(out_names, 'salinityCoarsened_0')
    ix_out_uvel    = eocn_channel_index(out_names, 'velocityZonalCoarsened_0')
    ix_out_vvel    = eocn_channel_index(out_names, 'velocityMeridionalCoarsened_0')
    ix_out_sifrac  = eocn_channel_index(out_names, 'ocean_sea_ice_fraction')
    ix_out_icevol  = eocn_channel_index(out_names, 'iceVolumeTotal')

    ! Every field the coupler is handed has to come from somewhere.  A silent
    ! zero here would export a 0 K sea surface to the atmosphere.
    if (min(ix_out_sst, ix_out_ssh, ix_out_sss, ix_out_uvel, ix_out_vvel) <= 0) &
         call shr_sys_abort(trim(subname)//' ERROR: emulator "'//trim(emulator)// &
         '" does not provide the surface state the coupler needs')

    write(logunit,'(a,i0,a,i0,a,i0,a)') &
         '(eocn_channels_init) '//trim(emulator)//': ', n_input_channels, &
         ' input, ', n_output_channels, ' output, ', n_forcing_channels, &
         ' next-step forcing channels'
    write(logunit,'(a,i0,a)') &
         '(eocn_channels_init) emulator timestep ', eocn_model_dt, ' s'

  end subroutine eocn_channels_init

  !===============================================================================
  subroutine set_table_samudrace()

    ! SamudrACE-E3SMv3 ocean: Samudra on the 180x360 Gaussian grid, 19 depth
    ! levels, 5 day step.  Two static forcings, ten atmospheric flux forcings,
    ! then the 80 prognostic channels the model both consumes and produces.

    integer :: k, n

    n_input_channels   = 92
    n_output_channels  = 80
    n_forcing_channels = 10

    allocate(in_names(n_input_channels))
    allocate(out_names(n_output_channels))
    allocate(forcing_names(n_forcing_channels))

    in_names(1)  = 'LANDFRAC'
    in_names(2)  = 'sea_surface_fraction'
    in_names(3)  = 'TAUX'
    in_names(4)  = 'TAUY'
    in_names(5)  = 'surface_precipitation_rate'
    in_names(6)  = 'frozen_precipitation_rate'
    in_names(7)  = 'FLUS'
    in_names(8)  = 'FSUS'
    in_names(9)  = 'FLDS'
    in_names(10) = 'FSDS'
    in_names(11) = 'LHFLX'
    in_names(12) = 'SHFLX'

    out_names(1) = 'sst'
    out_names(2) = 'ssh'
    n = 2
    do k = 0, 18
      n = n + 1
      write(out_names(n),'(a,i0)') 'salinityCoarsened_', k
    end do
    do k = 0, 18
      n = n + 1
      write(out_names(n),'(a,i0)') 'temperatureCoarsened_', k
    end do
    do k = 0, 18
      n = n + 1
      write(out_names(n),'(a,i0)') 'velocityZonalCoarsened_', k
    end do
    do k = 0, 18
      n = n + 1
      write(out_names(n),'(a,i0)') 'velocityMeridionalCoarsened_', k
    end do
    n = n + 1
    out_names(n) = 'ocean_sea_ice_fraction'
    n = n + 1
    out_names(n) = 'iceVolumeTotal'

    ! the state block of the input tensor is the output block, in order
    do k = 1, n_output_channels
      in_names(12 + k) = out_names(k)
    end do

    ! next-step forcing: the ten atmospheric fluxes, again
    do k = 1, n_forcing_channels
      forcing_names(k) = in_names(2 + k)
    end do

    eocn_model_dt = 5 * 24 * 60 * 60

  end subroutine set_table_samudrace

  !===============================================================================
  integer function eocn_channel_index(names, want)

    character(len=*), intent(in) :: names(:)
    character(len=*), intent(in) :: want

    integer :: k

    eocn_channel_index = 0
    do k = 1, size(names)
      if (trim(names(k)) == trim(want)) then
        eocn_channel_index = k
        return
      end if
    end do

  end function eocn_channel_index

  !===============================================================================
  subroutine eocn_channel_metadata(name, long_name, units)

    ! Enough metadata to make the restart file self-describing.  Matched on the
    ! prefix so the 19 levels of each 3D field share one entry.

    character(len=*), intent(in)  :: name
    character(len=*), intent(out) :: long_name
    character(len=*), intent(out) :: units

    if (index(name, 'salinityCoarsened_') == 1) then
      long_name = 'coarsened potential salinity'
      units     = 'g/kg'
    else if (index(name, 'temperatureCoarsened_') == 1) then
      long_name = 'coarsened potential temperature'
      units     = 'degC'
    else if (index(name, 'velocityZonalCoarsened_') == 1) then
      long_name = 'coarsened zonal velocity'
      units     = 'm/s'
    else if (index(name, 'velocityMeridionalCoarsened_') == 1) then
      long_name = 'coarsened meridional velocity'
      units     = 'm/s'
    else
      select case (trim(name))
      case ('sst')
        long_name = 'sea surface temperature'
        units     = 'K'
      case ('ssh')
        long_name = 'sea surface height'
        units     = 'm'
      case ('ocean_sea_ice_fraction')
        long_name = 'sea ice fraction of the ocean surface'
        units     = '1'
      case ('iceVolumeTotal')
        long_name = 'sea ice volume per unit area'
        units     = 'm'
      case ('LANDFRAC')
        long_name = 'land fraction'
        units     = '1'
      case ('sea_surface_fraction')
        long_name = 'sea surface fraction'
        units     = '1'
      case default
        long_name = trim(name)
        units     = 'unknown'
      end select
    end if

  end subroutine eocn_channel_metadata

  !===============================================================================
  subroutine eocn_channels_final()

    if (allocated(in_names))      deallocate(in_names)
    if (allocated(out_names))     deallocate(out_names)
    if (allocated(forcing_names)) deallocate(forcing_names)
    if (allocated(in_from_out))   deallocate(in_from_out)

  end subroutine eocn_channels_final

end module eocn_channels_mod
