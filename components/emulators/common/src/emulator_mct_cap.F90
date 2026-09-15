module emulator_mct_cap
   !------------------------------------------------------------------------
   ! Shared MCT cap logic for emulated components; compiled only in the
   ! integrated build, which has the driver's modules.
   !------------------------------------------------------------------------

   use esmf
   use mct_mod
   use seq_cdata_mod,    only: seq_cdata, seq_cdata_setptrs
   use seq_infodata_mod, only: seq_infodata_type, seq_infodata_getdata
   use seq_timemgr_mod,  only: seq_timemgr_EClockGetData, &
                               seq_timemgr_RestartAlarmIsOn
   use seq_comm_mct,     only: seq_comm_suffix
   use shr_kind_mod,     only: IN=>SHR_KIND_IN, R8=>SHR_KIND_R8, &
                               CL=>SHR_KIND_CL
   use shr_file_mod,     only: shr_file_getunit, shr_file_freeunit, &
                               shr_file_setIO, &
                               shr_file_getLogUnit, shr_file_setLogUnit
   use shr_sys_mod,      only: shr_sys_flush, shr_sys_abort
   use iso_c_binding
   use emulator_f_api,   only: emulator_create_cfg, emulator_coupling_desc, &
                               create_config
   use emulator_f2c_api

   implicit none
   private

   public :: emulator_cap, emulator_cap_init, emulator_cap_run, &
             emulator_cap_final, emulator_creator

   ! A component library's emulator_create_<kind>.  Each cap passes its own,
   ! so this module refers to none of them: it lives in emulator_common,
   ! which links after the component libraries, and a case with only an
   ! emulated atmosphere has no ocean creator to resolve.
   abstract interface
      function emulator_creator(cfg) result(handle)
         import :: c_ptr, emulator_create_cfg
         type(emulator_create_cfg), intent(in) :: cfg
         type(c_ptr) :: handle
      end function emulator_creator
   end interface

   integer, parameter :: master_task = 0

   type :: emulator_cap
      type(c_ptr)       :: handle = c_null_ptr
      character(len=3)  :: kind = ''
      integer           :: comp_id = -1
      integer           :: mpicom = -1
      integer           :: my_task = 0
      integer           :: logunit = 6
      character(len=CL) :: case_name = ''
      character(len=16) :: inst_suffix = ''
   end type emulator_cap

contains

   !==========================================================================
   subroutine emulator_cap_init(cap, kind, create, EClock, cdata, x2c, c2x, &
                                import_fields, export_fields)
      ! Create the component, hand the driver its gsMap and domain, bind the
      ! attribute vectors, and initialize.  On return c2x holds the
      ! component's initial exports.

      type(emulator_cap),    intent(inout) :: cap
      character(len=3),      intent(in)    :: kind   ! atm, ocn or ice
      procedure(emulator_creator)          :: create
      type(ESMF_Clock),      intent(inout) :: EClock
      type(seq_cdata),       intent(inout) :: cdata
      type(mct_aVect),       intent(inout) :: x2c, c2x
      character(len=*),      intent(in)    :: import_fields, export_fields

      type(seq_infodata_type), pointer :: infodata
      type(mct_gsMap),         pointer :: gsMap
      type(mct_gGrid),         pointer :: dom
      integer :: ierr, lsize, shrlogunit, cur_ymd, cur_tod
      character(CL) :: run_type
      character(len=16) :: inst_suffix
      character(kind=c_char, len=256), target :: input_file_c, log_file_c
      character(len=256) :: log_file_f
      character(len=CL) :: restart_file
      character(kind=c_char), allocatable, target :: import_c(:), export_c(:)
      integer(c_int) :: run_type_c
      type(emulator_create_cfg)    :: cfg
      type(emulator_coupling_desc) :: cpl

      cap%kind = kind
      call seq_cdata_setptrs(cdata, id=cap%comp_id, mpicom=cap%mpicom, &
         gsMap=gsMap, dom=dom, infodata=infodata)
      call seq_infodata_getData(infodata, start_type=run_type, &
         case_name=cap%case_name)
      inst_suffix = seq_comm_suffix(cap%comp_id)
      cap%inst_suffix = inst_suffix
      call MPI_Comm_rank(cap%mpicom, cap%my_task, ierr)

      if (cap%my_task == master_task) then
         cap%logunit = shr_file_getunit()
         call shr_file_setIO(kind//'_modelio.nml'//trim(inst_suffix), &
                             cap%logunit)
      else
         cap%logunit = 6
      endif
      call shr_file_getLogUnit(shrlogunit)
      call shr_file_setLogUnit(cap%logunit)

      if (cap%my_task == master_task) then
         write(cap%logunit,*) '(emulator'//kind//') '//kind// &
            '_init_mct starting'
         call shr_sys_flush(cap%logunit)
      endif

      select case (trim(run_type))
      case ('continue')
         run_type_c = 1
      case ('branch')
         run_type_c = 2
      case default
         run_type_c = 0
      end select

      call seq_timemgr_EClockGetData(EClock, curr_ymd=cur_ymd, &
         curr_tod=cur_tod)

      input_file_c = kind//'_in'//trim(inst_suffix)//C_NULL_CHAR
      log_file_c = C_NULL_CHAR
      if (cap%my_task == master_task) then
         inquire(unit=cap%logunit, name=log_file_f)
         log_file_c = trim(log_file_f)//C_NULL_CHAR
      endif
      cfg = create_config(f_comm=cap%mpicom, comp_id=cap%comp_id, &
            run_type=run_type_c, start_ymd=cur_ymd, start_tod=cur_tod, &
            input_file=input_file_c, log_file=log_file_c)

      cap%handle = create(cfg)
      if (.not. c_associated(cap%handle)) then
         call shr_sys_abort('(emulator_cap_init) no emulated '//kind)
      endif

      call set_gsmap(cap, gsMap)
      lsize = mct_gsMap_lsize(gsMap, cap%mpicom)
      call set_domain(cap, lsize, gsMap, dom)

      call mct_aVect_init(x2c, rList=import_fields, lsize=lsize)
      call mct_aVect_init(c2x, rList=export_fields, lsize=lsize)
      call mct_aVect_zero(x2c)
      call mct_aVect_zero(c2x)

      call to_c_string(import_fields, import_c)
      call to_c_string(export_fields, export_c)
      call emulator_init_coupling_indices(cap%handle, c_loc(import_c), &
         c_loc(export_c))

      ! x2c%rAttr is (nflds, lsize): point-major in C.
      cpl%import_data = c_loc(x2c%rAttr(1,1))
      cpl%export_data = c_loc(c2x%rAttr(1,1))
      cpl%num_imports = mct_aVect_nRattr(x2c)
      cpl%num_exports = mct_aVect_nRattr(c2x)
      cpl%field_size  = lsize
      call emulator_setup_coupling(cap%handle, cpl)

      ! A continued or branched run restores from the file its rpointer
      ! names, which the component reads on its whole grid at init.
      if (run_type_c /= 0) then
         call read_rpointer(cap, restart_file)
         if (cap%my_task == master_task) then
            write(cap%logunit,*) '(emulator'//kind//') restarting from ', &
               trim(restart_file)
            call shr_sys_flush(cap%logunit)
         endif
         call emulator_set_restart_file(cap%handle, &
            trim(restart_file)//C_NULL_CHAR)
      endif

      call emulator_init(cap%handle)

      if (cap%my_task == master_task) then
         write(cap%logunit,*) '(emulator'//kind//') '//kind// &
            '_init_mct complete: nx, ny, global cells =', &
            emulator_get_nx(cap%handle), emulator_get_ny(cap%handle), &
            emulator_get_num_global_cols(cap%handle)
         call shr_sys_flush(cap%logunit)
      endif
      call shr_file_setLogUnit(shrlogunit)

   end subroutine emulator_cap_init

   !==========================================================================
   subroutine emulator_cap_run(cap, EClock)
      type(emulator_cap), intent(inout) :: cap
      type(ESMF_Clock),   intent(inout) :: EClock

      integer :: dt, ymd, tod, shrlogunit

      call seq_timemgr_EClockGetData(EClock, dtime=dt, curr_ymd=ymd, &
         curr_tod=tod)
      call shr_file_getLogUnit(shrlogunit)
      call shr_file_setLogUnit(cap%logunit)
      ! The driver's time, not a count: run can be called twice at one time.
      call emulator_run_at(cap%handle, int(dt,c_int), int(ymd,c_int), &
                           int(tod,c_int))
      if (seq_timemgr_RestartAlarmIsOn(EClock)) then
         call write_restart(cap, EClock)
      endif
      call shr_file_setLogUnit(shrlogunit)
   end subroutine emulator_cap_run

   !==========================================================================
   subroutine write_restart(cap, EClock)
      ! $CASE.emulator<kind>$NINST.r.YYYY-MM-DD-SSSSS.nc, as the component's
      ! config_archive.xml expects, and its name in rpointer.<kind>$NINST.
      type(emulator_cap), intent(in)    :: cap
      type(ESMF_Clock),   intent(inout) :: EClock

      integer :: yr, mon, day, tod, unit
      character(len=CL) :: fname

      call seq_timemgr_EClockGetData(EClock, curr_yr=yr, curr_mon=mon, &
         curr_day=day, curr_tod=tod)
      write(fname, '(a,".emulator",a,a,".r.",i4.4,"-",i2.2,"-",i2.2,"-",i5.5,".nc")') &
         trim(cap%case_name), cap%kind, trim(cap%inst_suffix), yr, mon, day, tod
      call emulator_write_restart(cap%handle, trim(fname)//C_NULL_CHAR)
      if (cap%my_task == master_task) then
         unit = shr_file_getunit()
         open(unit, file='rpointer.'//cap%kind//trim(cap%inst_suffix), &
              form='formatted', status='replace')
         write(unit, '(a)') trim(fname)
         close(unit)
         call shr_file_freeUnit(unit)
         write(cap%logunit,*) '(emulator'//cap%kind//') wrote ', trim(fname)
         call shr_sys_flush(cap%logunit)
      endif
   end subroutine write_restart

   !==========================================================================
   subroutine read_rpointer(cap, restart_file)
      type(emulator_cap), intent(in)  :: cap
      character(len=*),   intent(out) :: restart_file

      integer :: unit, ios
      character(len=CL) :: rpointer

      rpointer = 'rpointer.'//cap%kind//trim(cap%inst_suffix)
      unit = shr_file_getunit()
      open(unit, file=trim(rpointer), form='formatted', status='old', &
           action='read', iostat=ios)
      if (ios /= 0) then
         call shr_sys_abort('(emulator_cap) cannot open '//trim(rpointer)// &
            ' for a continue or branch run')
      endif
      read(unit, '(a)', iostat=ios) restart_file
      close(unit)
      call shr_file_freeUnit(unit)
      if (ios /= 0 .or. len_trim(restart_file) == 0) then
         call shr_sys_abort('(emulator_cap) '//trim(rpointer)// &
            ' names no restart file')
      endif
      restart_file = adjustl(restart_file)
   end subroutine read_rpointer

   !==========================================================================
   subroutine emulator_cap_final(cap)
      type(emulator_cap), intent(inout) :: cap

      call emulator_finalize(cap%handle)
      if (cap%my_task == master_task) then
         write(cap%logunit,*) '(emulator'//cap%kind//') '//cap%kind// &
            '_final_mct complete'
         call shr_sys_flush(cap%logunit)
      endif
   end subroutine emulator_cap_final

   !==========================================================================
   subroutine set_gsmap(cap, gsMap)
      type(emulator_cap), intent(in)  :: cap
      type(mct_gsMap),    intent(out) :: gsMap

      integer(c_int) :: nlocal, nglobal
      integer(c_int), allocatable, target :: gids(:)

      nlocal  = emulator_get_num_local_cols(cap%handle)
      nglobal = emulator_get_num_global_cols(cap%handle)
      allocate(gids(nlocal))
      call emulator_get_local_col_gids(cap%handle, gids)
      call mct_gsMap_init(gsMap, gids, cap%mpicom, cap%comp_id, nlocal, &
                          nglobal)
      deallocate(gids)
   end subroutine set_gsmap

   !==========================================================================
   subroutine set_domain(cap, lsize, gsMap, dom)
      type(emulator_cap), intent(in)    :: cap
      integer,            intent(in)    :: lsize
      type(mct_gsMap),    intent(in)    :: gsMap
      type(mct_gGrid),    intent(inout) :: dom

      integer,  pointer :: idata(:)
      real(R8), pointer :: data1(:), data2(:)

      allocate(data1(lsize), data2(lsize))
      call mct_gGrid_init(GGrid=dom, CoordChars='lat:lon:hgt', &
         OtherChars='area:aream:mask:frac', lsize=lsize)

      nullify(idata)
      call mct_gsMap_orderedPoints(gsMap, cap%my_task, idata)
      call mct_gGrid_importIAttr(dom, 'GlobGridNum', idata, lsize)

      data1(:) = -9999.0_R8
      call mct_gGrid_importRAttr(dom, 'hgt', data1, lsize)

      call emulator_get_cols_latlon(cap%handle, data1, data2)
      call mct_gGrid_importRAttr(dom, 'lat', data1, lsize)
      call mct_gGrid_importRAttr(dom, 'lon', data2, lsize)

      call emulator_get_cols_area(cap%handle, data1)
      call mct_gGrid_importRAttr(dom, 'area',  data1, lsize)
      call mct_gGrid_importRAttr(dom, 'aream', data1, lsize)

      ! Mask and frac from the component: 1 for the atmosphere, the ocean's
      ! binary mask for the ocean and the sea ice that shares its domain.
      call emulator_get_cols_mask_frac(cap%handle, data1, data2)
      call mct_gGrid_importRAttr(dom, 'mask', data1, lsize)
      call mct_gGrid_importRAttr(dom, 'frac', data2, lsize)

      deallocate(data1, data2)
      if (associated(idata)) deallocate(idata)
   end subroutine set_domain

   !==========================================================================
   subroutine to_c_string(s, buf)
      ! A NUL-terminated copy of trim(s), however long.
      character(len=*), intent(in) :: s
      character(kind=c_char), allocatable, intent(out) :: buf(:)
      integer :: i, n
      n = len_trim(s)
      allocate(buf(n + 1))
      do i = 1, n
         buf(i) = s(i:i)
      end do
      buf(n + 1) = C_NULL_CHAR
   end subroutine to_c_string

end module emulator_mct_cap
