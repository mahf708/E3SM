! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.io/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_fme_horiz_remap
!
!> \brief MPAS sea ice FME horizontal remapping wrapper
!> \author FME Training Data Support
!> \date   April 2026
!> \details
!>  Wrapper around shr_horiz_remap_t for MPAS-Sea-Ice with PIO output.
!>  Same architecture as ocn_fme_horiz_remap.
!>
!-----------------------------------------------------------------------

module seaice_fme_horiz_remap

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_dmpar
   use mpas_kind_types
   use mpas_log, only: mpas_log_write

   use shr_kind_mod, only: r8 => shr_kind_r8
   use shr_horiz_remap_mod, only: shr_horiz_remap_t
   use pio, only: file_desc_t, var_desc_t, io_desc_t, iosystem_desc_t, &
                  pio_createfile, pio_closefile, &
                  pio_def_dim, pio_def_var, pio_enddef, &
                  pio_put_att, pio_put_var, pio_write_darray, &
                  pio_initdecomp, pio_setframe, &
                  PIO_IOTYPE_PNETCDF, PIO_CLOBBER, PIO_REAL, PIO_DOUBLE, &
                  PIO_CHAR, PIO_OFFSET_KIND, PIO_GLOBAL, PIO_NOERR, &
                  PIO_UNLIMITED

   implicit none
   private
   save

   public :: seaice_fme_horiz_remap_init
   public :: seaice_fme_horiz_remap_field
   public :: seaice_fme_horiz_remap_active
   public :: seaice_fme_horiz_remap_open_file
   public :: seaice_fme_horiz_remap_def_var
   public :: seaice_fme_horiz_remap_write_var
   public :: seaice_fme_horiz_remap_write_time
   public :: seaice_fme_horiz_remap_close_file
   public :: seaice_fme_horiz_remap_get_dims

   type(shr_horiz_remap_t) :: remap_data
   integer, allocatable :: send_local_cell(:)
   real(r8), allocatable :: ws_send_buf(:)
   logical :: is_initialized = .false.

   ! PIO output
   type(file_desc_t) :: pio_file
   type(iosystem_desc_t), pointer :: pio_sys => null()
   logical :: file_is_open = .false.
   integer :: dim_lon_id, dim_lat_id, dim_time_id
   integer :: time_record = 0
   type(io_desc_t) :: iodesc_2d
   logical :: iodesc_2d_valid = .false.

   integer, parameter :: max_pio_vars = 100
   integer :: n_pio_vars = 0
   character(len=64) :: pio_var_names(max_pio_vars)
   type(var_desc_t) :: pio_var_descs(max_pio_vars)
   type(var_desc_t) :: pio_lon_var, pio_lat_var

contains

   subroutine seaice_fme_horiz_remap_init(domain, mapfile, err)!{{{
      use mpi, only: MPI_INTEGER, MPI_MAX, MPI_IN_PLACE

      type(domain_type), intent(inout) :: domain
      character(len=*), intent(in) :: mapfile
      integer, intent(out) :: err

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: meshPool
      integer, pointer :: nCellsSolve
      integer, dimension(:), pointer :: indexToCellID
      integer :: myrank, nprocs, comm, ierr, iCell, gcol, i
      integer, allocatable :: gcol_to_rank(:), send_gcol_list(:)
      integer(PIO_OFFSET_KIND), allocatable :: idof(:)

      err = 0
      if (is_initialized) return
      if (len_trim(mapfile) == 0 .or. trim(mapfile) == 'none') then
         is_initialized = .false.; return
      end if

      myrank = domain % dminfo % my_proc_id
      nprocs = domain % dminfo % nprocs
      comm   = domain % dminfo % comm

      call remap_data%read_mapfile(trim(mapfile), comm, myrank, nprocs, &
           domain % ioContext % pio_iosystem, ierr)
      if (ierr /= 0) then
         err = ierr; return
      end if

      allocate(gcol_to_rank(remap_data%n_a))
      gcol_to_rank(:) = -1
      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)
         do iCell = 1, nCellsSolve
            gcol = indexToCellID(iCell)
            if (gcol >= 1 .and. gcol <= remap_data%n_a) gcol_to_rank(gcol) = myrank
         end do
         block => block % next
      end do

      ! Gather full gcol_to_rank across all ranks: each rank only knows which
      ! cells it owns locally; after MPI_MAX every entry holds the owning rank.
      call mpi_allreduce(MPI_IN_PLACE, gcol_to_rank, remap_data%n_a, &
           MPI_INTEGER, MPI_MAX, comm, ierr)

      call remap_data%build_comm(gcol_to_rank, comm, myrank, nprocs, &
           send_gcol_list, ierr)
      deallocate(gcol_to_rank)
      if (ierr /= 0) then
         err = ierr; return
      end if

      allocate(send_local_cell(remap_data%n_send_total))
      send_local_cell(:) = 0
      block => domain % blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_dimension(block % dimensions, 'nCellsSolve', nCellsSolve)
         call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)
         do i = 1, remap_data%n_send_total
            if (send_local_cell(i) > 0) cycle
            do iCell = 1, nCellsSolve
               if (indexToCellID(iCell) == send_gcol_list(i)) then
                  send_local_cell(i) = iCell; exit
               end if
            end do
         end do
         block => block % next
      end do
      if (allocated(send_gcol_list)) deallocate(send_gcol_list)

      pio_sys => domain % ioContext % pio_iosystem

      ! Build 2D PIO decomposition
      allocate(idof(remap_data%n_b_local))
      do i = 1, remap_data%n_b_local
         gcol = remap_data%row_start + i - 1
         idof(i) = int(mod(gcol-1, remap_data%nlon) + 1, PIO_OFFSET_KIND) + &
              int(remap_data%nlon, PIO_OFFSET_KIND) * &
              int((gcol-1) / remap_data%nlon, PIO_OFFSET_KIND)
      end do
      call pio_initdecomp(pio_sys, PIO_REAL, &
           (/remap_data%nlon, remap_data%nlat/), idof, iodesc_2d)
      deallocate(idof)
      iodesc_2d_valid = .true.

      is_initialized = .true.
   end subroutine seaice_fme_horiz_remap_init!}}}

   subroutine seaice_fme_horiz_remap_field(domain, field1d, nCells, fld_out, err)!{{{
      type(domain_type), intent(inout) :: domain
      integer, intent(in) :: nCells
      real(kind=RKIND), intent(in) :: field1d(nCells)
      real(r8), intent(out) :: fld_out(remap_data%n_b_local)
      integer, intent(out) :: err
      real(r8) :: fld_out_2d(remap_data%n_b_local, 1)
      integer :: i

      err = 0
      if (.not. is_initialized) then; err = 1; return; end if

      if (remap_data%n_send_total > 0) then
         if (.not. allocated(ws_send_buf) .or. size(ws_send_buf) < remap_data%n_send_total) then
            if (allocated(ws_send_buf)) deallocate(ws_send_buf)
            allocate(ws_send_buf(remap_data%n_send_total))
         end if
         do i = 1, remap_data%n_send_total
            ws_send_buf(i) = real(field1d(send_local_cell(i)), r8)
         end do
      end if

      call remap_data%apply(ws_send_buf, 1, fld_out_2d, &
           domain % dminfo % comm, domain % dminfo % nprocs, err)
      fld_out(1:remap_data%n_b_local) = fld_out_2d(1:remap_data%n_b_local, 1)
   end subroutine seaice_fme_horiz_remap_field!}}}

   subroutine seaice_fme_horiz_remap_open_file(domain, filename, err)!{{{
      type(domain_type), intent(inout) :: domain
      character(len=*), intent(in) :: filename
      integer, intent(out) :: err
      integer :: ierr

      err = 0
      ierr = pio_createfile(pio_sys, pio_file, PIO_IOTYPE_PNETCDF, trim(filename), PIO_CLOBBER)
      if (ierr /= PIO_NOERR) then; err = ierr; return; end if

      ierr = pio_def_dim(pio_file, 'lon', remap_data%nlon, dim_lon_id)
      ierr = pio_def_dim(pio_file, 'lat', remap_data%nlat, dim_lat_id)
      ierr = pio_def_dim(pio_file, 'Time', PIO_UNLIMITED, dim_time_id)

      ! Define coordinate variables; descriptors saved at module scope for write_time
      ierr = pio_def_var(pio_file, 'lon', PIO_DOUBLE, (/dim_lon_id/), pio_lon_var)
      ierr = pio_put_att(pio_file, pio_lon_var, 'units', 'degrees_east')
      ierr = pio_put_att(pio_file, pio_lon_var, 'long_name', 'longitude')
      ierr = pio_def_var(pio_file, 'lat', PIO_DOUBLE, (/dim_lat_id/), pio_lat_var)
      ierr = pio_put_att(pio_file, pio_lat_var, 'units', 'degrees_north')
      ierr = pio_put_att(pio_file, pio_lat_var, 'long_name', 'latitude')

      n_pio_vars = 0
      time_record = 0
      file_is_open = .true.
   end subroutine seaice_fme_horiz_remap_open_file!}}}

   subroutine seaice_fme_horiz_remap_def_var(varname, units, long_name, numlev, err)!{{{
      character(len=*), intent(in) :: varname, units, long_name
      integer, intent(in) :: numlev
      integer, intent(out) :: err
      integer :: ierr

      err = 0
      n_pio_vars = n_pio_vars + 1
      pio_var_names(n_pio_vars) = varname
      ierr = pio_def_var(pio_file, trim(varname), PIO_REAL, &
           (/dim_lon_id, dim_lat_id, dim_time_id/), pio_var_descs(n_pio_vars))
      ierr = pio_put_att(pio_file, pio_var_descs(n_pio_vars), 'units', trim(units))
      ierr = pio_put_att(pio_file, pio_var_descs(n_pio_vars), 'long_name', trim(long_name))
   end subroutine seaice_fme_horiz_remap_def_var!}}}

   subroutine seaice_fme_horiz_remap_write_var(varname, fld_out, numlev, err)!{{{
      character(len=*), intent(in) :: varname
      real(r8), intent(in) :: fld_out(:)
      integer, intent(in) :: numlev
      integer, intent(out) :: err
      integer :: iv, ierr

      err = 0
      do iv = 1, n_pio_vars
         if (trim(pio_var_names(iv)) == trim(varname)) then
            call pio_setframe(pio_file, pio_var_descs(iv), &
                 int(time_record, PIO_OFFSET_KIND))
            call pio_write_darray(pio_file, pio_var_descs(iv), iodesc_2d, fld_out, ierr)
            return
         end if
      end do
      err = 3
   end subroutine seaice_fme_horiz_remap_write_var!}}}

   subroutine seaice_fme_horiz_remap_write_time(time_str, err)!{{{
      character(len=*), intent(in) :: time_str
      integer, intent(out) :: err
      integer :: ierr

      err = 0
      if (time_record == 0) then
         ierr = pio_enddef(pio_file)
         ! Write lat/lon coordinate variables using module-level var descriptors
         ierr = pio_put_var(pio_file, pio_lon_var, remap_data%lon(1:remap_data%nlon))
         ierr = pio_put_var(pio_file, pio_lat_var, remap_data%lat(1:remap_data%nlat))
      end if
      time_record = time_record + 1
   end subroutine seaice_fme_horiz_remap_write_time!}}}

   subroutine seaice_fme_horiz_remap_close_file(err)!{{{
      integer, intent(out) :: err
      err = 0
      if (file_is_open) then
         call pio_closefile(pio_file)
         file_is_open = .false.
      end if
   end subroutine seaice_fme_horiz_remap_close_file!}}}

   subroutine seaice_fme_horiz_remap_get_dims(nlat, nlon, n_b_local)!{{{
      integer, intent(out) :: nlat, nlon, n_b_local
      nlat = remap_data%nlat; nlon = remap_data%nlon; n_b_local = remap_data%n_b_local
   end subroutine seaice_fme_horiz_remap_get_dims!}}}

   logical function seaice_fme_horiz_remap_active()!{{{
      seaice_fme_horiz_remap_active = is_initialized
   end function seaice_fme_horiz_remap_active!}}}

end module seaice_fme_horiz_remap
