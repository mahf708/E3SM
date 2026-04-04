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
   use mpas_log, only: mpas_log_write

   use shr_kind_mod, only: r8 => shr_kind_r8
   use shr_horiz_remap_mod, only: shr_horiz_remap_t
   use pio, only: file_desc_t, var_desc_t, io_desc_t, iosystem_desc_t, &
                  pio_openfile, pio_closefile, pio_createfile, &
                  pio_def_dim, pio_def_var, pio_enddef, &
                  pio_put_att, pio_put_var, pio_write_darray, &
                  pio_initdecomp, pio_freedecomp, &
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
   public :: ocn_fme_horiz_remap_open_file
   public :: ocn_fme_horiz_remap_def_var
   public :: ocn_fme_horiz_remap_write_var
   public :: ocn_fme_horiz_remap_write_time
   public :: ocn_fme_horiz_remap_close_file
   public :: ocn_fme_horiz_remap_get_dims

   type(shr_horiz_remap_t) :: remap_data

   ! Send-side cell mapping
   integer, allocatable :: send_local_cell(:)
   real(r8), allocatable :: ws_send_buf(:)
   logical :: is_initialized = .false.

   ! PIO output file state
   type(file_desc_t) :: pio_file
   type(iosystem_desc_t), pointer :: pio_sys => null()
   logical :: file_is_open = .false.
   integer :: dim_lon_id, dim_lat_id, dim_time_id
   integer :: time_record = 0

   ! PIO decomposition cache
   type(io_desc_t) :: iodesc_2d
   logical :: iodesc_2d_valid = .false.

   ! Variable registry for the current file (up to 100 vars)
   integer, parameter :: max_pio_vars = 100
   integer :: n_pio_vars = 0
   character(len=64) :: pio_var_names(max_pio_vars)
   type(var_desc_t) :: pio_var_descs(max_pio_vars)
   integer :: pio_var_nlevs(max_pio_vars)

   ! Time and coordinate variables
   type(var_desc_t) :: pio_time_var
   type(var_desc_t) :: pio_lon_var, pio_lat_var

!***********************************************************************

contains

!***********************************************************************

   subroutine ocn_fme_horiz_remap_init(domain, mapfile, err)!{{{

      type(domain_type), intent(inout) :: domain
      character(len=*), intent(in) :: mapfile
      integer, intent(out) :: err

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: meshPool
      integer, pointer :: nCellsSolve
      integer, dimension(:), pointer :: indexToCellID

      use mpi, only: MPI_INTEGER, MPI_MAX, MPI_IN_PLACE

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

   subroutine ocn_fme_horiz_remap_field(domain, field2d, numlev, fld_out, err)!{{{

      type(domain_type), intent(inout) :: domain
      integer, intent(in) :: numlev
      real(kind=RKIND), intent(in) :: field2d(numlev, *)
      real(r8), intent(out) :: fld_out(remap_data%n_b_local, numlev)
      integer, intent(out) :: err

      integer :: i, k, needed, comm, nprocs

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
            do k = 1, numlev
               ws_send_buf((i-1)*numlev + k) = &
                    real(field2d(k, send_local_cell(i)), r8)
            end do
         end do
      end if

      call remap_data%apply(ws_send_buf, numlev, fld_out, comm, nprocs, err)

   end subroutine ocn_fme_horiz_remap_field!}}}

!***********************************************************************

   subroutine ocn_fme_horiz_remap_open_file(domain, filename, err)!{{{
      !-----------------------------------------------------------------
      ! Create a new NetCDF file for remapped lat-lon output.
      ! Defines lon, lat, Time dimensions and coordinate variables.
      !-----------------------------------------------------------------

      type(domain_type), intent(inout) :: domain
      character(len=*), intent(in) :: filename
      integer, intent(out) :: err

      integer :: nlat, nlon, ierr

      err = 0
      if (.not. is_initialized) then
         err = 1; return
      end if

      nlat = remap_data%nlat
      nlon = remap_data%nlon

      ! Create file
      ierr = pio_createfile(pio_sys, pio_file, PIO_IOTYPE_PNETCDF, &
           trim(filename), PIO_CLOBBER)
      if (ierr /= PIO_NOERR) then
         call mpas_log_write('ocn_fme_horiz_remap: failed to create file ' &
              // trim(filename), messageType=MPAS_LOG_ERR)
         err = ierr; return
      end if

      ! Define dimensions
      ierr = pio_def_dim(pio_file, 'lon', nlon, dim_lon_id)
      ierr = pio_def_dim(pio_file, 'lat', nlat, dim_lat_id)
      ierr = pio_def_dim(pio_file, 'Time', PIO_UNLIMITED, dim_time_id)

      ! Define coordinate variables (descriptors saved at module scope for write_time)
      ierr = pio_def_var(pio_file, 'lon', PIO_DOUBLE, (/dim_lon_id/), pio_lon_var)
      ierr = pio_put_att(pio_file, pio_lon_var, 'units', 'degrees_east')
      ierr = pio_put_att(pio_file, pio_lon_var, 'long_name', 'longitude')

      ierr = pio_def_var(pio_file, 'lat', PIO_DOUBLE, (/dim_lat_id/), pio_lat_var)
      ierr = pio_put_att(pio_file, pio_lat_var, 'units', 'degrees_north')
      ierr = pio_put_att(pio_file, pio_lat_var, 'long_name', 'latitude')

      ! Define time variable
      ierr = pio_def_var(pio_file, 'xtime', PIO_CHAR, (/dim_time_id/), pio_time_var)

      ! Global attributes
      ierr = pio_put_att(pio_file, PIO_GLOBAL, 'source', 'E3SM FME online remapping')
      ierr = pio_put_att(pio_file, PIO_GLOBAL, 'Conventions', 'CF-1.6')

      ! Reset variable registry
      n_pio_vars = 0
      time_record = 0

      ! Don't end define mode yet — caller will add variables via def_var
      ! End define mode will happen on first write

      file_is_open = .true.

      ! Write coordinate data (must be after enddef, handled in first write)

   end subroutine ocn_fme_horiz_remap_open_file!}}}

!***********************************************************************

   subroutine ocn_fme_horiz_remap_def_var(varname, units, long_name, numlev, err)!{{{
      !-----------------------------------------------------------------
      ! Define a variable in the remapped output file.
      ! Must be called after open_file and before any write_var.
      ! For 2D fields: numlev=1 -> dims (lon, lat, Time)
      !-----------------------------------------------------------------
      character(len=*), intent(in) :: varname, units, long_name
      integer, intent(in) :: numlev
      integer, intent(out) :: err

      integer :: ierr

      err = 0
      if (.not. file_is_open) then
         err = 1; return
      end if

      n_pio_vars = n_pio_vars + 1
      if (n_pio_vars > max_pio_vars) then
         call mpas_log_write('ocn_fme_horiz_remap: too many variables defined', &
              messageType=MPAS_LOG_ERR)
         err = 2; return
      end if

      pio_var_names(n_pio_vars) = varname
      pio_var_nlevs(n_pio_vars) = numlev

      ! Define as (lon, lat, Time) for 2D
      ierr = pio_def_var(pio_file, trim(varname), PIO_REAL, &
           (/dim_lon_id, dim_lat_id, dim_time_id/), &
           pio_var_descs(n_pio_vars))
      ierr = pio_put_att(pio_file, pio_var_descs(n_pio_vars), 'units', trim(units))
      ierr = pio_put_att(pio_file, pio_var_descs(n_pio_vars), 'long_name', trim(long_name))

   end subroutine ocn_fme_horiz_remap_def_var!}}}

!***********************************************************************

   subroutine ocn_fme_horiz_remap_write_var(varname, fld_out, numlev, err)!{{{
      !-----------------------------------------------------------------
      ! Write a remapped field to the output file.
      ! fld_out should be the result of ocn_fme_horiz_remap_field.
      !-----------------------------------------------------------------
      character(len=*), intent(in) :: varname
      real(r8), intent(in) :: fld_out(:,:)
      integer, intent(in) :: numlev
      integer, intent(out) :: err

      integer :: iv, ierr
      real(r8) :: fld_1d(remap_data%n_b_local)

      err = 0
      if (.not. file_is_open) then
         err = 1; return
      end if

      ! Find variable index
      do iv = 1, n_pio_vars
         if (trim(pio_var_names(iv)) == trim(varname)) then
            ! Write 2D field using PIO decomposition
            fld_1d(1:remap_data%n_b_local) = real(fld_out(1:remap_data%n_b_local, 1), r8)
            call pio_write_darray(pio_file, pio_var_descs(iv), iodesc_2d, fld_1d, ierr)
            return
         end if
      end do

      ! Variable not found
      call mpas_log_write('ocn_fme_horiz_remap_write_var: variable not found: ' &
           // trim(varname))
      err = 3

   end subroutine ocn_fme_horiz_remap_write_var!}}}

!***********************************************************************

   subroutine ocn_fme_horiz_remap_write_time(time_str, err)!{{{
      !-----------------------------------------------------------------
      ! Advance the time record and write coordinate data on first call.
      ! Must be called once per output timestep before write_var calls.
      !-----------------------------------------------------------------
      character(len=*), intent(in) :: time_str
      integer, intent(out) :: err

      integer :: ierr

      err = 0

      if (time_record == 0) then
         ! First write: end define mode and write coordinates
         ierr = pio_enddef(pio_file)

         ! Write lat/lon coordinate variables using module-level var descriptors
         ierr = pio_put_var(pio_file, pio_lon_var, remap_data%lon(1:remap_data%nlon))
         ierr = pio_put_var(pio_file, pio_lat_var, remap_data%lat(1:remap_data%nlat))
      end if

      time_record = time_record + 1

   end subroutine ocn_fme_horiz_remap_write_time!}}}

!***********************************************************************

   subroutine ocn_fme_horiz_remap_close_file(err)!{{{
      integer, intent(out) :: err

      err = 0
      if (file_is_open) then
         call pio_closefile(pio_file)
         file_is_open = .false.
         n_pio_vars = 0
         time_record = 0
      end if

   end subroutine ocn_fme_horiz_remap_close_file!}}}

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

end module ocn_fme_horiz_remap

! vim: foldmethod=marker
