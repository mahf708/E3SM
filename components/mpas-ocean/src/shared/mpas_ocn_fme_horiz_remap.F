! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_fme_horiz_remap
!
!> \brief MPAS ocean FME horizontal remapping wrapper
!> \author FME Training Data Support
!> \date   April 2026
!> \details
!>  Wrapper around shr_horiz_remap_t for MPAS-Ocean. Handles MPAS cell
!>  decomposition mapping, send buffer packing, CRS SpMV remapping, and
!>  direct PIO output of remapped fields on a lat-lon grid.
!>
!>  The PIO output bypasses the MPAS stream manager because streams
!>  only support the native unstructured grid (nCells). Remapped output
!>  uses (nlon, nlat) dimensions directly.
!>
!>  Usage:
!>    call ocn_fme_horiz_remap_init(domain, mapfile, err)
!>    call ocn_fme_horiz_remap_field(domain, field2d, numlev, fld_out, err)
!>    call ocn_fme_horiz_remap_open_file(domain, filename, err)
!>    call ocn_fme_horiz_remap_def_var(varname, units, long_name, numlev, err)
!>    call ocn_fme_horiz_remap_write_var(varname, fld_out, numlev, err)
!>    call ocn_fme_horiz_remap_write_time(time_str, err)
!>    call ocn_fme_horiz_remap_close_file(err)
!>    logical = ocn_fme_horiz_remap_active()
!>
!-----------------------------------------------------------------------

module ocn_fme_horiz_remap

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_kind_types
   use mpas_timekeeping
   use mpas_stream_manager, only: MPAS_stream_mgr_get_property
   use mpas_log, only: mpas_log_write

   use shr_kind_mod, only: r8 => shr_kind_r8
   use shr_horiz_remap_mod, only: shr_horiz_remap_t
   use pio, only: file_desc_t, var_desc_t, io_desc_t, iosystem_desc_t, &
                  pio_openfile, pio_closefile, pio_createfile, &
                  pio_def_dim, pio_def_var, pio_enddef, &
                  pio_put_att, pio_put_var, pio_write_darray, &
                  pio_initdecomp, pio_freedecomp, pio_setframe, &
                  PIO_IOTYPE_PNETCDF, PIO_IOTYPE_NETCDF4P, &
                  PIO_CLOBBER, PIO_REAL, PIO_DOUBLE, PIO_CHAR, &
                  PIO_OFFSET_KIND, PIO_GLOBAL, PIO_NOERR, &
                  PIO_UNLIMITED

   implicit none
   private
   save

   public :: ocn_fme_horiz_remap_init
   public :: ocn_fme_horiz_remap_field
   public :: ocn_fme_horiz_remap_active
   public :: ocn_fme_horiz_remap_get_dims
   public :: ocn_fme_remap_file_t
   public :: ocn_fme_remap_build_filename

   type(shr_horiz_remap_t) :: remap_data

   ! Send-side cell mapping
   integer, allocatable :: send_local_cell(:)
   real(r8), allocatable :: ws_send_buf(:)
   logical :: is_initialized = .false.

   ! PIO decomposition cache (shared across all file instances)
   type(iosystem_desc_t), pointer :: pio_sys => null()
   type(io_desc_t) :: iodesc_2d
   logical :: iodesc_2d_valid = .false.

   ! Per-file output state: each AM creates its own instance
   integer, parameter :: max_pio_vars = 100

   type :: ocn_fme_remap_file_t
      type(file_desc_t) :: pio_file
      logical :: file_is_open = .false.
      integer :: dim_lon_id = 0, dim_lat_id = 0, dim_time_id = 0
      integer :: time_record = 0
      integer :: n_pio_vars = 0
      character(len=64)  :: pio_var_names(max_pio_vars)
      type(var_desc_t)   :: pio_var_descs(max_pio_vars)
      type(var_desc_t)   :: pio_lon_var, pio_lat_var
      ! Stored metadata for replaying var definitions after rotation
      character(len=64)  :: stored_units(max_pio_vars)
      character(len=128) :: stored_long_names(max_pio_vars)
      integer :: n_registered_vars = 0
      ! Current open filename and stream name for rotation
      character(len=512) :: current_filename = ''
      character(len=128) :: stream_name = ''
   contains
      procedure :: open_file    => remap_file_open
      procedure :: def_var      => remap_file_def_var
      procedure :: register_var => remap_file_register_var
      procedure :: write_var    => remap_file_write_var
      procedure :: write_time   => remap_file_write_time
      procedure :: close_file   => remap_file_close
      procedure :: check_rotate => remap_file_check_rotate
   end type ocn_fme_remap_file_t

!***********************************************************************

contains

!***********************************************************************

   subroutine ocn_fme_horiz_remap_init(domain, mapfile, err)!{{{
      use mpi, only: MPI_INTEGER, MPI_MAX, MPI_IN_PLACE

      type(domain_type), intent(inout) :: domain
      character(len=*), intent(in) :: mapfile
      integer, intent(out) :: err

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: meshPool
      integer, pointer :: nCellsSolve
      integer, dimension(:), pointer :: indexToCellID
      integer :: myrank, nprocs, comm, ierr
      integer :: iCell, gcol
      integer, allocatable :: gcol_to_rank(:)
      integer, allocatable :: send_gcol_list(:)
      integer :: i

      err = 0

      if (is_initialized) return

      if (len_trim(mapfile) == 0 .or. trim(mapfile) == 'none') then
         is_initialized = .false.
         return
      end if

      myrank = domain % dminfo % my_proc_id
      nprocs = domain % dminfo % nprocs
      comm   = domain % dminfo % comm

      ! Phase 1: Read map file
      call remap_data%read_mapfile(trim(mapfile), comm, myrank, nprocs, &
           domain % ioContext % pio_iosystem, ierr)
      if (ierr /= 0) then
         call mpas_log_write('ocn_fme_horiz_remap_init: failed to read map file: ' &
              // trim(mapfile), messageType=MPAS_LOG_ERR)
         err = ierr
         return
      end if

      ! Build gcol_to_rank mapping from MPAS cell decomposition
      allocate(gcol_to_rank(remap_data%n_a))
      gcol_to_rank(:) = -1

      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)

         do iCell = 1, nCellsSolve
            gcol = indexToCellID(iCell)
            if (gcol >= 1 .and. gcol <= remap_data%n_a) then
               gcol_to_rank(gcol) = myrank
            end if
         end do

         block => block % next
      end do

      ! Gather full gcol_to_rank across all ranks: each rank only knows which
      ! cells it owns locally; after MPI_MAX every entry holds the owning rank.
      call mpi_allreduce(MPI_IN_PLACE, gcol_to_rank, remap_data%n_a, &
           MPI_INTEGER, MPI_MAX, comm, ierr)

      ! Phase 2: Build MPI communication pattern
      call remap_data%build_comm(gcol_to_rank, comm, myrank, nprocs, &
           send_gcol_list, ierr)
      deallocate(gcol_to_rank)

      if (ierr /= 0) then
         call mpas_log_write('ocn_fme_horiz_remap_init: failed to build comm pattern', &
              messageType=MPAS_LOG_ERR)
         err = ierr
         return
      end if

      ! Convert send_gcol_list to local cell indices
      allocate(send_local_cell(remap_data%n_send_total))
      send_local_cell(:) = 0
      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)

         do i = 1, remap_data%n_send_total
            if (send_local_cell(i) > 0) cycle  ! already found
            do iCell = 1, nCellsSolve
               if (indexToCellID(iCell) == send_gcol_list(i)) then
                  send_local_cell(i) = iCell
                  exit
               end if
            end do
         end do

         block => block % next
      end do

      if (allocated(send_gcol_list)) deallocate(send_gcol_list)

      ! Store PIO system reference for later file creation
      pio_sys => domain % ioContext % pio_iosystem

      ! Build 2D PIO decomposition for lat-lon output
      call build_iodesc_2d()

      is_initialized = .true.

      if (myrank == 0) then
         call mpas_log_write('ocn_fme_horiz_remap_init: initialized with ' &
              // 'n_a=$i, n_b=$i, nlat=$i, nlon=$i', &
              intArgs=(/remap_data%n_a, remap_data%n_b, &
                        remap_data%nlat, remap_data%nlon/))
      end if

   end subroutine ocn_fme_horiz_remap_init!}}}

!***********************************************************************

   subroutine build_iodesc_2d()!{{{
      integer(PIO_OFFSET_KIND), allocatable :: idof(:)
      integer :: i, global_row, ilon, ilat, nlat, nlon

      nlat = remap_data%nlat
      nlon = remap_data%nlon

      allocate(idof(remap_data%n_b_local))
      do i = 1, remap_data%n_b_local
         global_row = remap_data%row_start + i - 1
         ilon = mod(global_row - 1, nlon) + 1
         ilat = (global_row - 1) / nlon + 1
         idof(i) = int(ilon, PIO_OFFSET_KIND) + &
              int(nlon, PIO_OFFSET_KIND) * int(ilat - 1, PIO_OFFSET_KIND)
      end do

      call pio_initdecomp(pio_sys, PIO_REAL, (/nlon, nlat/), idof, iodesc_2d)
      deallocate(idof)
      iodesc_2d_valid = .true.

   end subroutine build_iodesc_2d!}}}

!***********************************************************************

   subroutine ocn_fme_horiz_remap_field(domain, field2d, numlev, fld_out, err, &
        mask)!{{{
   !> Remap a field from the MPAS unstructured grid to the lat-lon target grid.
   !> field2d(numlev, *) is on the native MPAS grid indexed by cell.
   !> If mask(nCells) is provided, cells where mask(iCell)==0 are set to zero
   !> before remapping (use maxLevelCell for ocean fields to exclude land cells).

      type(domain_type), intent(inout) :: domain
      integer, intent(in) :: numlev
      real(kind=RKIND), intent(in) :: field2d(numlev, *)
      real(r8), intent(out) :: fld_out(remap_data%n_b_local, numlev)
      integer, intent(out) :: err
      integer, intent(in), optional :: mask(*)

      integer :: i, k, needed, comm, nprocs, icell

      err = 0

      if (.not. is_initialized) then
         err = 1
         return
      end if

      nprocs = domain % dminfo % nprocs
      comm   = domain % dminfo % comm

      needed = remap_data%n_send_total * numlev
      if (needed > 0) then
         if (.not. allocated(ws_send_buf) .or. size(ws_send_buf) < needed) then
            if (allocated(ws_send_buf)) deallocate(ws_send_buf)
            allocate(ws_send_buf(needed))
         end if

         do i = 1, remap_data%n_send_total
            icell = send_local_cell(i)
            do k = 1, numlev
               if (present(mask)) then
                  ! Zero out land/below-bottom: mask(iCell) is maxLevelCell
                  ! for ocean, or >0 for valid cells in other contexts.
                  if (mask(icell) < k) then
                     ws_send_buf((i-1)*numlev + k) = 0.0_r8
                     cycle
                  end if
               end if
               ws_send_buf((i-1)*numlev + k) = &
                    real(field2d(k, icell), r8)
            end do
         end do
      end if

      call remap_data%apply(ws_send_buf, numlev, fld_out, comm, nprocs, err)

   end subroutine ocn_fme_horiz_remap_field!}}}

!***********************************************************************

   !-----------------------------------------------------------------
   ! Type-bound methods for per-file PIO output
   !-----------------------------------------------------------------

   subroutine remap_file_open(self, filename, err)!{{{
      class(ocn_fme_remap_file_t), intent(inout) :: self
      character(len=*), intent(in) :: filename
      integer, intent(out) :: err

      integer :: nlat, nlon, ierr

      err = 0
      if (.not. is_initialized) then; err = 1; return; end if

      nlat = remap_data%nlat
      nlon = remap_data%nlon

      ierr = pio_createfile(pio_sys, self%pio_file, PIO_IOTYPE_PNETCDF, &
           trim(filename), PIO_CLOBBER)
      if (ierr /= PIO_NOERR) then
         call mpas_log_write('ocn_fme_remap_file: failed to create ' // trim(filename), &
              messageType=MPAS_LOG_ERR)
         err = ierr; return
      end if

      ierr = pio_def_dim(self%pio_file, 'lon', nlon, self%dim_lon_id)
      ierr = pio_def_dim(self%pio_file, 'lat', nlat, self%dim_lat_id)
      ierr = pio_def_dim(self%pio_file, 'Time', PIO_UNLIMITED, self%dim_time_id)

      ierr = pio_def_var(self%pio_file, 'lon', PIO_DOUBLE, &
           (/self%dim_lon_id/), self%pio_lon_var)
      ierr = pio_put_att(self%pio_file, self%pio_lon_var, 'units', 'degrees_east')
      ierr = pio_def_var(self%pio_file, 'lat', PIO_DOUBLE, &
           (/self%dim_lat_id/), self%pio_lat_var)
      ierr = pio_put_att(self%pio_file, self%pio_lat_var, 'units', 'degrees_north')

      ierr = pio_put_att(self%pio_file, PIO_GLOBAL, 'source', 'E3SM FME online remapping')
      ierr = pio_put_att(self%pio_file, PIO_GLOBAL, 'Conventions', 'CF-1.8')

      self%n_pio_vars = 0
      self%time_record = 0
      self%file_is_open = .true.

   end subroutine remap_file_open!}}}

   subroutine remap_file_def_var(self, varname, units, long_name, numlev, err)!{{{
      class(ocn_fme_remap_file_t), intent(inout) :: self
      character(len=*), intent(in) :: varname, units, long_name
      integer, intent(in) :: numlev
      integer, intent(out) :: err
      integer :: ierr
      real(kind=4) :: fill_val

      err = 0
      if (.not. self%file_is_open) then; err = 1; return; end if

      self%n_pio_vars = self%n_pio_vars + 1
      if (self%n_pio_vars > max_pio_vars) then
         call mpas_log_write('ocn_fme_remap_file: exceeded max_pio_vars=$i for var: ' &
              // trim(varname), messageType=MPAS_LOG_ERR, &
              intArgs=(/max_pio_vars/))
         self%n_pio_vars = self%n_pio_vars - 1
         err = 2; return
      end if

      self%pio_var_names(self%n_pio_vars) = varname
      ierr = pio_def_var(self%pio_file, trim(varname), PIO_REAL, &
           (/self%dim_lon_id, self%dim_lat_id, self%dim_time_id/), &
           self%pio_var_descs(self%n_pio_vars))
      ierr = pio_put_att(self%pio_file, self%pio_var_descs(self%n_pio_vars), &
           'units', trim(units))
      ierr = pio_put_att(self%pio_file, self%pio_var_descs(self%n_pio_vars), &
           'long_name', trim(long_name))
      fill_val = 1.0E34
      ierr = pio_put_att(self%pio_file, self%pio_var_descs(self%n_pio_vars), &
           '_FillValue', fill_val)

   end subroutine remap_file_def_var!}}}

   subroutine remap_file_write_var(self, varname, fld_out, numlev, err)!{{{
      class(ocn_fme_remap_file_t), intent(inout) :: self
      character(len=*), intent(in) :: varname
      real(r8), intent(in) :: fld_out(:,:)
      integer, intent(in) :: numlev
      integer, intent(out) :: err
      integer :: iv, ierr
      real(r8) :: fld_1d(remap_data%n_b_local)

      err = 0
      if (.not. self%file_is_open) then; err = 1; return; end if

      do iv = 1, self%n_pio_vars
         if (trim(self%pio_var_names(iv)) == trim(varname)) then
            call pio_setframe(self%pio_file, self%pio_var_descs(iv), &
                 int(self%time_record, PIO_OFFSET_KIND))
            fld_1d(1:remap_data%n_b_local) = real(fld_out(1:remap_data%n_b_local, 1), r8)
            call pio_write_darray(self%pio_file, self%pio_var_descs(iv), &
                 iodesc_2d, fld_1d, ierr)
            return
         end if
      end do
      err = 3

   end subroutine remap_file_write_var!}}}

   subroutine remap_file_write_time(self, time_str, err)!{{{
      class(ocn_fme_remap_file_t), intent(inout) :: self
      character(len=*), intent(in) :: time_str
      integer, intent(out) :: err
      integer :: ierr

      err = 0
      if (self%time_record == 0) then
         ierr = pio_enddef(self%pio_file)
         ierr = pio_put_var(self%pio_file, self%pio_lon_var, &
              remap_data%lon(1:remap_data%nlon))
         ierr = pio_put_var(self%pio_file, self%pio_lat_var, &
              remap_data%lat(1:remap_data%nlat))
      end if
      self%time_record = self%time_record + 1

   end subroutine remap_file_write_time!}}}

   subroutine remap_file_close(self, err)!{{{
      class(ocn_fme_remap_file_t), intent(inout) :: self
      integer, intent(out) :: err
      err = 0
      if (self%file_is_open) then
         call pio_closefile(self%pio_file)
         self%file_is_open = .false.
         self%n_pio_vars = 0
         self%time_record = 0
         self%current_filename = ''
      end if
   end subroutine remap_file_close!}}}

   subroutine remap_file_register_var(self, varname, units, long_name, err)!{{{
   !> Register a variable for output. Stores metadata so that def_var can
   !> be replayed automatically when a file is opened or rotated.
   !> Call this once per variable during setup (before any writes).
      class(ocn_fme_remap_file_t), intent(inout) :: self
      character(len=*), intent(in) :: varname, units, long_name
      integer, intent(out) :: err

      err = 0
      self%n_registered_vars = self%n_registered_vars + 1
      if (self%n_registered_vars > max_pio_vars) then
         call mpas_log_write('ocn_fme_remap_file: exceeded max_pio_vars=$i for var: ' &
              // trim(varname), messageType=MPAS_LOG_ERR, &
              intArgs=(/max_pio_vars/))
         self%n_registered_vars = self%n_registered_vars - 1
         err = 2; return
      end if
      self%pio_var_names(self%n_registered_vars) = varname
      self%stored_units(self%n_registered_vars) = units
      self%stored_long_names(self%n_registered_vars) = long_name
   end subroutine remap_file_register_var!}}}

   subroutine remap_file_check_rotate(self, domain, err)!{{{
   !> Check whether the file needs rotation based on the current model date.
   !> Builds a filename from the stream template + current date. If different
   !> from the currently open file, closes the old file and opens a new one,
   !> replaying all registered variable definitions.
      class(ocn_fme_remap_file_t), intent(inout) :: self
      type(domain_type), intent(inout) :: domain
      integer, intent(out) :: err

      character(len=512) :: new_filename
      integer :: ierr, iv

      err = 0
      if (self%n_registered_vars == 0) return
      if (len_trim(self%stream_name) == 0) return

      call ocn_fme_remap_build_filename(domain, trim(self%stream_name), &
           new_filename, ierr)
      if (ierr /= 0) then; err = ierr; return; end if

      ! First call or date boundary crossed: open (or rotate) file
      if (.not. self%file_is_open .or. &
           trim(new_filename) /= trim(self%current_filename)) then

         if (self%file_is_open) then
            call self%close_file(ierr)
         end if

         call self%open_file(trim(new_filename), ierr)
         if (ierr /= 0) then; err = ierr; return; end if
         self%current_filename = new_filename

         ! Replay all registered variable definitions
         do iv = 1, self%n_registered_vars
            call self%def_var(trim(self%pio_var_names(iv)), &
                 trim(self%stored_units(iv)), &
                 trim(self%stored_long_names(iv)), 1, ierr)
         end do
      end if

   end subroutine remap_file_check_rotate!}}}

!***********************************************************************

   subroutine ocn_fme_horiz_remap_get_dims(nlat, nlon, n_b_local)!{{{
      integer, intent(out) :: nlat, nlon, n_b_local
      nlat = remap_data%nlat
      nlon = remap_data%nlon
      n_b_local = remap_data%n_b_local
   end subroutine ocn_fme_horiz_remap_get_dims!}}}

!***********************************************************************

   logical function ocn_fme_horiz_remap_active()!{{{
      ocn_fme_horiz_remap_active = is_initialized
   end function ocn_fme_horiz_remap_active!}}}

!***********************************************************************

   subroutine ocn_fme_remap_build_filename(domain, streamName, filename, err)!{{{
   !> Build a date-stamped remapped output filename by querying the stream
   !> manager for the native stream's filename_template and substituting
   !> the current date. The native stream's filename_template is set by
   !> buildnml and contains the casename, instance string, and $Y-$M-$D
   !> date placeholders (e.g., "mycase.mpaso.hist.am.fmeDerived.$Y-$M-$D.nc").
   !> We replace the ".nc" extension with ".remapped.nc" and substitute dates.

      type(domain_type), intent(inout) :: domain
      character(len=*), intent(in) :: streamName
      character(len=*), intent(out) :: filename
      integer, intent(out) :: err

      character(len=StrKIND) :: tmpl
      type(MPAS_Time_Type) :: currTime
      integer :: yr, mo, dy, ierr, idx
      character(len=4) :: yr_str
      character(len=2) :: mo_str, dy_str

      err = 0
      filename = ''

      ! Query the stream manager for the native stream's filename template
      call MPAS_stream_mgr_get_property(domain % streamManager, &
           trim(streamName), MPAS_STREAM_PROPERTY_FILENAME, tmpl, ierr=ierr)
      if (ierr /= 0) then
         call mpas_log_write('ocn_fme_remap_build_filename: could not query stream ' &
              // trim(streamName) // ', using fallback name', messageType=MPAS_LOG_WARN)
         ! Fallback: use stream name directly with date
         tmpl = trim(streamName) // '.$Y-$M-$D.nc'
      end if

      ! Insert ".remapped" before ".nc" extension
      idx = index(tmpl, '.nc', back=.true.)
      if (idx > 0) then
         tmpl = tmpl(1:idx-1) // '.remapped.nc'
      else
         tmpl = trim(tmpl) // '.remapped.nc'
      end if

      ! Substitute date placeholders from current model time
      currTime = mpas_get_clock_time(domain % clock, MPAS_NOW, ierr)
      call mpas_get_time(currTime, YYYY=yr, MM=mo, DD=dy, ierr=ierr)

      write(yr_str, '(I4.4)') yr
      write(mo_str, '(I2.2)') mo
      write(dy_str, '(I2.2)') dy

      ! Replace $Y, $M, $D tokens
      filename = tmpl
      call replace_token(filename, '$Y', yr_str)
      call replace_token(filename, '$M', mo_str)
      call replace_token(filename, '$D', dy_str)

   end subroutine ocn_fme_remap_build_filename!}}}

!***********************************************************************

   subroutine replace_token(str, token, replacement)!{{{
      character(len=*), intent(inout) :: str
      character(len=*), intent(in) :: token, replacement
      integer :: idx
      character(len=len(str)) :: tmp

      idx = index(str, token)
      if (idx > 0) then
         tmp = str(1:idx-1) // trim(replacement) // str(idx+len(token):)
         str = tmp
      end if
   end subroutine replace_token!}}}

end module ocn_fme_horiz_remap

! vim: foldmethod=marker
