module ocn_comp_mct

  ! !USES:

  use esmf
  use netcdf
  use pio
  use mct_mod
  use perf_mod
  use seq_cdata_mod   , only: seq_cdata, seq_cdata_setptrs
  use seq_infodata_mod, only: seq_infodata_type, seq_infodata_putdata, seq_infodata_getdata
  use seq_comm_mct    , only: seq_comm_inst, seq_comm_name, seq_comm_suffix
  use shr_kind_mod    , only: IN=>SHR_KIND_IN, R8=>SHR_KIND_R8, CS=>SHR_KIND_CS, CL=>SHR_KIND_CL
  use shr_const_mod   , only: SHR_CONST_RHOFW
  use shr_file_mod    , only: shr_file_getunit, shr_file_getlogunit, shr_file_getloglevel
  use shr_file_mod    , only: shr_file_setlogunit, shr_file_setloglevel, shr_file_setio
  use shr_file_mod    , only: shr_file_freeunit
  use shr_mpi_mod     , only: shr_mpi_bcast
  use shr_emul_flux_mod, only: shr_emul_flux_get, shr_emul_flux_avail,       &
                               shr_emul_flux_nchan,                          &
                               shr_emul_flux_taux,  shr_emul_flux_tauy,      &
                               shr_emul_flux_prec,  shr_emul_flux_snow,      &
                               shr_emul_flux_flus,  shr_emul_flux_fsus,      &
                               shr_emul_flux_flds,  shr_emul_flux_fsds,      &
                               shr_emul_flux_lhflx, shr_emul_flux_shflx
  use shr_sys_mod     , only: shr_sys_flush, shr_sys_abort
  use seq_flds_mod    , only: seq_flds_o2x_fields, seq_flds_x2o_fields
  use seq_flds_mod    , only: seq_flds_dom_coord, seq_flds_dom_other

  use ocn_cpl_indices
  use eocnMod
  use eocn_comp_mod
  use eocn_channels_mod
  use eocnSpmdMod
  use eocnIO
  use shr_emul_ice_mod, only: shr_emul_ice_put_grid

  implicit none
  private

  public :: ocn_init_mct
  public :: ocn_run_mct
  public :: ocn_final_mct

  integer                :: inst_index
  integer(IN),parameter  :: master_task=0

  ! global cell = all cells in the mesh
  real(r8), allocatable :: latc_g(:)
  real(r8), allocatable :: lonc_g(:)
  real(r8), allocatable :: areac_g(:)

  integer, allocatable :: start(:)
  integer, allocatable :: length(:)
  integer, allocatable :: pe_loc(:)

  ! Ocean albedo used to split the coupler's net surface shortwave back into
  ! the downward and upward channels the emulator was trained on.  The coupler
  ! sends only the net (Foxx_swnet), and the emulator wants FSDS and FSUS
  ! separately.  Splitting with a constant keeps their *difference* -- the term
  ! that actually drives the ocean heat budget -- exactly right, and misstates
  ! only how the pair is apportioned.  0.06 is E3SM's diffuse ocean albedo.
  real(r8), parameter :: ocn_albedo = 0.06_r8

CONTAINS

  !===============================================================================
  subroutine ocn_init_mct( EClock, cdata, x2o, o2x, NLFilename )

    implicit none

    type(ESMF_Clock)            , intent(inout) :: EClock
    type(seq_cdata)             , intent(inout) :: cdata
    type(mct_aVect)             , intent(inout) :: x2o, o2x
    character(len=*), optional  , intent(in)    :: NLFilename

    type(seq_infodata_type), pointer :: infodata
    type(mct_gsMap)        , pointer :: gsMap
    type(mct_gGrid)        , pointer :: ggrid
    logical           :: ocn_present
    logical           :: ocn_prognostic
    integer(IN)       :: shrlogunit
    integer(IN)       :: shrloglev
    logical           :: read_restart
    logical           :: exists
    logical           :: first_time = .true.
    integer           :: mpicom_loc

    character(*), parameter :: F00 = "('(ocn_comp_init) ',8a)"
    character(*), parameter :: subName = "(ocn_init_mct) "

    call seq_cdata_setptrs(cdata, &
         id=compid, &
         mpicom=mpicom_loc, &
         gsMap=gsmap, &
         dom=ggrid, &
         infodata=infodata)

    call seq_infodata_getData(infodata, read_restart=read_restart)

    if (first_time) then

       call ocn_cpl_indices_set()
       call eocnSpmdInit(mpicom_loc)

       inst_name   = seq_comm_name(compid)
       inst_index  = seq_comm_inst(compid)
       inst_suffix = seq_comm_suffix(compid)

       call ncd_pio_init(inst_name)

       call shr_file_getLogUnit (shrlogunit)
       if (masterproc) then
          inquire(file='ocn_modelio.nml'//trim(inst_suffix),exist=exists)
          if (exists) then
             logunit_ocn = shr_file_getUnit()
             call shr_file_setIO('ocn_modelio.nml'//trim(inst_suffix),logunit_ocn)
          end if
          write(logunit_ocn,*) "eocn model initialization"
       else
          logunit_ocn = shrlogunit
       endif

       call shr_file_getLogLevel(shrloglev)
       call shr_file_setLogUnit (logunit_ocn)

       call t_startf('eocn_readnml')
       ocn_present    = .true.
       ocn_prognostic = .true.
       call eocn_read_namelist()

       call seq_infodata_PutData(infodata, &
            ocn_present=ocn_present, &
            ocn_prognostic=ocn_prognostic, &
            ocnrof_prognostic=.false.)
       call t_stopf('eocn_readnml')

       call ocn_SetGSMap_mct( mpicom_ocn, compid, gsMap)

       ! Allocate and read the emulator state before the domain is built: the
       ! ocean fraction the coupler needs in the domain is one of the
       ! emulator's own input channels, and nothing else on disk carries the
       ! mask the checkpoint was trained with.
       call eocn_comp_alloc(read_restart)

       call ocn_domain_mct( gsMap, ggrid )

       call mct_aVect_init(x2o, rList=seq_flds_x2o_fields, lsize=lsize)
       call mct_aVect_zero(x2o)

       call mct_aVect_init(o2x, rList=seq_flds_o2x_fields, lsize=lsize)
       call mct_aVect_zero(o2x)

       call eocn_comp_init(Eclock, x2o, o2x, &
            seq_flds_x2o_fields, seq_flds_o2x_fields, &
            gsmap, ggrid, read_restart)

       call seq_infodata_PutData(infodata, &
            ocn_nx=lsize_x, &
            ocn_ny=lsize_y)

       call ocn_export_mct(o2x)

       if (masterproc) write(logunit_ocn,F00) 'eocn_comp_init done'
       call shr_sys_flush(logunit_ocn)

       call shr_file_setLogUnit (shrlogunit)
       call shr_file_setLogLevel(shrloglev)

       first_time = .false.

    else

       call shr_file_getLogUnit (shrlogunit)
       call shr_file_getLogLevel(shrloglev)
       call shr_file_setLogUnit (logunit_ocn)

       if (.not. read_restart) then
          call ocn_import_mct(x2o)
          call t_startf('EOCN_run')
          call eocn_comp_run(EClock=EClock, x2o=x2o, o2x=o2x, gsmap=gsmap, ggrid=ggrid)
          call t_stopf('EOCN_run')
          call ocn_export_mct(o2x)
       end if

       call shr_file_setLogUnit (shrlogunit)
       call shr_file_setLogLevel(shrloglev)

    end if

    call shr_sys_flush(logunit_ocn)

  end subroutine ocn_init_mct

  !===============================================================================
  subroutine ocn_run_mct( EClock, cdata,  x2o, o2x)

    implicit none

    type(ESMF_Clock)            ,intent(inout) :: EClock
    type(seq_cdata)             ,intent(inout) :: cdata
    type(mct_aVect)             ,intent(inout) :: x2o
    type(mct_aVect)             ,intent(inout) :: o2x

    type(seq_infodata_type), pointer :: infodata
    type(mct_gsMap)        , pointer :: gsMap
    type(mct_gGrid)        , pointer :: ggrid
    character(*), parameter :: subName = "(ocn_run_mct) "

    call seq_cdata_setptrs(cdata, gsMap=gsmap, dom=ggrid, infodata=infodata)
    call seq_infodata_GetData(infodata, case_name=case_name)

    call t_startf ('lc_eocn_import')
    call ocn_import_mct( x2o )
    call t_stopf ('lc_eocn_import')

    call eocn_comp_run(EClock=EClock, x2o=x2o, o2x=o2x, gsmap=gsmap, ggrid=ggrid)

    call t_startf ('lc_eocn_export')
    call ocn_export_mct( o2x )
    call t_stopf ('lc_eocn_export')

  end subroutine ocn_run_mct

  !===============================================================================
  subroutine ocn_final_mct(EClock, cdata, x2o, o2x)

    implicit none
    type(ESMF_Clock)            ,intent(inout) :: EClock
    type(seq_cdata)             ,intent(inout) :: cdata
    type(mct_aVect)             ,intent(inout) :: x2o
    type(mct_aVect)             ,intent(inout) :: o2x

    call eocn_comp_final()

  end subroutine ocn_final_mct

  !===============================================================================
  subroutine ocn_SetGSMap_mct( mpicom_ocn, compid, gsMap)

    implicit none
    integer        , intent(in)    :: mpicom_ocn
    integer        , intent(in)    :: compid
    type(mct_gsMap), intent(inout) :: gsMap

    integer :: n

    call ocn_read_eocn()

    allocate(start(npes), length(npes), pe_loc(npes))
    start = 0
    length = 0
    pe_loc = 0

    do n = 1,npes
       length(n)  = gsize/npes
       if (n <= mod(gsize,npes)) length(n) = length(n) + 1
       if (n == 1) then
           start(n) = 1
       else
           start(n) = start(n-1) + length(n-1)
       endif
       pe_loc(n) = n-1
    enddo

    call mct_gsMap_init( gsMap, compid, npes, gsize, start, length, pe_loc)

  end subroutine ocn_SetGSMap_mct

  !===============================================================================
  subroutine ocn_domain_mct( gsMap, dom_ocn )

    ! Build the MCT domain.  Unlike EATM, the fraction is not 1 everywhere:
    ! this grid covers the whole globe and only part of it is ocean.
    !
    ! The fraction is the emulator's ocean mask, and it is binary rather than
    ! the (continuous) sea surface fraction the checkpoint also carries.  That
    ! is the coupler's convention, not a simplification: when the atmosphere
    ! and ocean grids differ, seq_domain_mct.F90:301 derives the ocean fraction
    ! on the atmosphere grid by mapping this domain's *mask*, and then requires
    ! it to equal one minus the land model's fraction.  A continuous frac
    ! beside a binary mask fails that check on every coastal cell.

    implicit none
    type(mct_gsMap), intent(in)    :: gsMap
    type(mct_gGrid), intent(inout) :: dom_ocn

    integer :: n, i, j
    integer , pointer :: idata(:)
    real(r8), pointer :: data(:)
    real(r8) :: frac

    call mct_gGrid_init( GGrid=dom_ocn, CoordChars=trim(seq_flds_dom_coord), &
      OtherChars=trim(seq_flds_dom_other), lsize=lsize )

    allocate(data(lsize))

    call mct_gsMap_orderedPoints(gsMap, iam, idata)
    call mct_gGrid_importIAttr(dom_ocn,'GlobGridNum',idata,lsize)

    data(:) = -9999.0_R8
    call mct_gGrid_importRAttr(dom_ocn,"lat"  ,data,lsize)
    call mct_gGrid_importRAttr(dom_ocn,"lon"  ,data,lsize)
    call mct_gGrid_importRAttr(dom_ocn,"area" ,data,lsize)
    call mct_gGrid_importRAttr(dom_ocn,"aream",data,lsize)
    data(:) = 0.0_R8
    call mct_gGrid_importRAttr(dom_ocn,"mask" ,data,lsize)

    data(:) = lonc_g(:)
    call mct_gGrid_importRattr(dom_ocn,"lon",data,lsize)
    data(:) = latc_g(:)
    call mct_gGrid_importRattr(dom_ocn,"lat",data,lsize)
    data(:) = areac_g(:)
    call mct_gGrid_importRattr(dom_ocn,"area",data,lsize)

    n = 0
    do j = 1, lsize_y
       do i = 1, lsize_x
          n = n + 1
          data(n) = ocn_mask(i,j)
       end do
    end do
    call mct_gGrid_importRattr(dom_ocn,"frac",data,lsize)

    do n = 1, lsize
       if (data(n) > 0.0_R8) then
          data(n) = 1.0_R8
       else
          data(n) = 0.0_R8
       end if
    end do
    call mct_gGrid_importRattr(dom_ocn,"mask",data,lsize)

    ! Publish the mesh for the ice half of the emulated ocean+ice entity.
    ! EICE has no grid of its own -- it reports a fraction of *this* domain --
    ! so taking the mesh from here is not a shortcut but the only way the two
    ! cannot disagree.  The MCT driver initialises ocn before ice, so this is
    ! always in place before anyone reads it.
    n = 0
    do j = 1, lsize_y
       do i = 1, lsize_x
          n = n + 1
          data(n) = ocn_mask(i,j)
       end do
    end do
    call shr_emul_ice_put_grid(lonc_g, latc_g, areac_g, data, gsize)

    deallocate(start,length,pe_loc)
    deallocate(data)
    deallocate(idata)

  end subroutine ocn_domain_mct

  !===============================================================================
  subroutine ocn_import_mct( x2o_o )

    ! Accumulate the coupler's surface exchange over the emulator interval.
    !
    ! Samudra's ten flux channels are means over its 5 day step, while the
    ! coupler recomputes and re-imports its fluxes every coupling step.
    ! Sampling once at the emulator boundary would compare a 5 day mean channel
    ! against a single 30 minute sample; summing here and averaging at the
    ! boundary makes both interval means over the same interval.
    !
    ! Signs and units are converted where the emulator's convention is applied,
    ! in samudra_import_forcing.  What is stored here is the coupler's own
    ! convention: fluxes positive into the ocean, precipitation in kg/m2/s.

    implicit none
    type(mct_aVect), intent(inout) :: x2o_o

    integer  :: i, j, n
    real(r8) :: swnet, fsds, w, ws, cpl_ifrac

    n = 0
    do j = 1, lsize_y
       do i = 1, lsize_x
          n = n + 1

          ! Undo the coupler's open-water weighting.
          !
          ! With a sea ice component present the coupler hands the ocean only
          ! the open-water share of the turbulent and longwave fluxes.  The
          ! generic merge in prep_ocn_mod would combine an atmosphere term and
          ! an ice term, but it skips every i2x field whose name begins Faii --
          ! the atmosphere/ice fluxes -- so for these channels no ice term is
          ! ever found and what the ocean receives is afrac*<atm/ocn flux> and
          ! nothing else.  The coupler prints the merge it performed at
          ! startup, and it reads
          !     x2o%Foxx_lat  = + afrac*xao%Faox_lat
          !     x2o%Foxx_sen  = + afrac*xao%Faox_sen
          !     x2o%Foxx_lwup = + afrac*xao%Faox_lwup
          !     x2o%Foxx_evap = + afrac*xao%Faox_evap
          !     x2o%Faxa_lwdn = afrac*a2x%Faxa_lwdn
          !     x2o%Faxa_rain = afrac*(a2x%Faxa_rainc + a2x%Faxa_rainl)*...
          ! with afrac and ifrac renormalised by their sum.  On this grid the
          ! ice and ocean share a mesh, so that normalised afrac is exactly
          ! 1 - so_ifrac.
          !
          ! Dividing by afrac therefore recovers the whole-cell atmosphere/
          ! ocean flux and cannot amplify an ice contribution, because there
          ! is none to amplify.  What it does not recover is the flux over the
          ! ice-covered part of the cell, which is a different surface: under
          ! heavy ice the true whole-cell flux is dominated by the atm/ice
          ! term the ocean never sees, and this substitutes the open-water
          ! flux for it.  The floor below caps that extrapolation at 100x,
          ! which binds on about 14% of ocean cells in an ice-covered winter
          ! and leaves them with roughly half the open-water flux.  Both the
          ! substitution and the floor are gone on the reference forcing path;
          ! see eocn_forcing_source.
          !
          ! Samudra was not trained on the open-water share.  In SamudrACE the
          ! ocean receives the atmosphere's whole-cell surface fluxes and
          ! accounts for the insulating effect of its own ice internally --
          ! ocean_sea_ice_fraction is one of its prognostic outputs, not a
          ! boundary condition it is handed.  Letting the coupler scale the
          ! forcing down by (1 - sea ice fraction) would apply that insulation
          ! a second time, in exactly the cells where it is largest.
          !
          ! So divide it back out and give the emulator the whole-cell flux it
          ! expects.  Set eocn_flux_ifrac_unweight = .false. to keep the
          ! coupler's open-water values instead, which is the physically
          ! standard choice for a model that does *not* carry its own ice.
          ! Undo the coupler's open-water weighting using the fraction the
          ! coupler actually weighted with -- Si_ifrac, which arrives in x2o --
          ! rather than Samudra's own prediction.  Two things follow.  With a
          ! stub ice component Si_ifrac is zero, so w is one and nothing is
          ! un-weighted: the amplification that used to make E2000-EATM-EOCN
          ! diverge cannot happen.  And where an ice component is present the
          ! two agree to about 4e-3 rather than exactly, so this also removes a
          ! small coupling-step lag between what the merge did and what is
          ! divided back out.
          if (eocn_flux_ifrac_unweight .and. index_x2o_Si_ifrac > 0) then
             cpl_ifrac = min(max(x2o_o%rAttr(index_x2o_Si_ifrac,n), 0.0_r8), 1.0_r8)
             w = 1.0_r8 / max(1.0_r8 - cpl_ifrac, 0.01_r8)
          else
             w = 1.0_r8
          end if
          ! TAUX/TAUY are the exception.  prep_ocn builds them as
          !   ifrac*i2x%Fioi_taux + afrac*xao%Faox_taux
          ! -- a genuine weighted mean, because EICE reports the under-ice
          ! stress equal to the atm/ice stress precisely so the merge
          ! reconstructs the whole-cell stress Samudra was trained on
          ! (eice/src/ice_comp_mct.F90, "The stress is passed through").
          ! Every other channel here has no ice term and arrives scaled by
          ! afrac, which is what w undoes.  Applying w to a field that is
          ! already whole-cell multiplies it by 1/(1-ifrac), up to the 100x
          ! the floor allows, in exactly the cells where the ice is thickest.
          if (eocn_unweight_stress) then
             ws = w
          else
             ws = 1.0_r8
          end if
          if (index_x2o_Foxx_taux > 0) &
               acc_taux(i,j) = acc_taux(i,j) + ws*x2o_o%rAttr(index_x2o_Foxx_taux,n)
          if (index_x2o_Foxx_tauy > 0) &
               acc_tauy(i,j) = acc_tauy(i,j) + ws*x2o_o%rAttr(index_x2o_Foxx_tauy,n)
          if (index_x2o_Foxx_lat > 0) &
               acc_lhflx(i,j) = acc_lhflx(i,j) + w*x2o_o%rAttr(index_x2o_Foxx_lat,n)
          if (index_x2o_Foxx_sen > 0) &
               acc_shflx(i,j) = acc_shflx(i,j) + w*x2o_o%rAttr(index_x2o_Foxx_sen,n)
          if (index_x2o_Foxx_lwup > 0) &
               acc_flus(i,j) = acc_flus(i,j) + w*x2o_o%rAttr(index_x2o_Foxx_lwup,n)
          if (index_x2o_Faxa_lwdn > 0) &
               acc_flds(i,j) = acc_flds(i,j) + w*x2o_o%rAttr(index_x2o_Faxa_lwdn,n)
          if (index_x2o_Faxa_rain > 0) &
               acc_prec(i,j) = acc_prec(i,j) + w*x2o_o%rAttr(index_x2o_Faxa_rain,n)
          if (index_x2o_Faxa_snow > 0) then
             acc_prec(i,j) = acc_prec(i,j) + w*x2o_o%rAttr(index_x2o_Faxa_snow,n)
             acc_snow(i,j) = acc_snow(i,j) + w*x2o_o%rAttr(index_x2o_Faxa_snow,n)
          end if
          ! Shortwave is the one channel here that the merge does give an ice
          ! term:
          !   Foxx_swnet = afracr*(open-water absorbed) + ifrac*i2x%Fioi_swpen
          ! (prep_ocn_mod.F90:1111).  Unweighting that by 1/(1-ifrac) recovers
          ! the open-water part and amplifies the penetrating part, which is
          ! why this path is only correct while the ice component reports no
          ! penetrating shortwave.  EICE sets Fioi_swpen to zero deliberately
          ! (eice/src/ice_comp_mct.F90) and the stub ice has none, and those
          ! are the only ice components any EOCN compset pairs with; a real
          ! ice component would need Fioi_swpen carried into x2o so it could be
          ! subtracted before the division.  The reference forcing path
          ! (eocn_forcing_source = 'atm_raw') takes FSDS and FSUS straight from
          ! the atmosphere emulator and sidesteps all of this, including the
          ! assumed ocean albedo below.
          if (index_x2o_Foxx_swnet > 0) then
             swnet = w*x2o_o%rAttr(index_x2o_Foxx_swnet,n)
             fsds  = swnet / (1.0_r8 - ocn_albedo)
             acc_fsds(i,j) = acc_fsds(i,j) + fsds
             acc_fsus(i,j) = acc_fsus(i,j) + (fsds - swnet)
          end if
       end do
    end do

    acc_n = acc_n + 1

    call ocn_import_atm_raw()

  end subroutine ocn_import_mct

  !===============================================================================
  subroutine ocn_import_atm_raw()

    ! Accumulate the atmosphere emulator's own generated forcing channels, when
    ! one is sharing this executable and publishing them.
    !
    ! This is the path reference SamudrACE uses and the coupler cannot express:
    ! the ten channels arrive exactly as the network produced them, so there is
    ! no bulk-flux recomputation, no open-water weighting to undo, no assumed
    ! ocean albedo to split the shortwave with, and no sign or unit change.
    ! They are accumulated over the same window as the coupler's fields and on
    ! the same steps, so the two are directly comparable at the boundary
    ! whichever one eocn_forcing_source selects.

    implicit none

    integer  :: i, j, n
    real(r8), allocatable, save :: f(:,:)
    logical,  save :: warned = .false.

    if (.not. shr_emul_flux_avail(lsize_x, lsize_y)) then
       if (.not. warned .and. trim(eocn_forcing_source) == 'atm_raw') then
          write(logunit_ocn,'(a)') '(ocn_import_atm_raw) WARNING: '// &
               "eocn_forcing_source = 'atm_raw' but no emulator atmosphere is "// &
               'publishing forcing channels on this grid; using the coupler fields'
          warned = .true.
       end if
       return
    end if

    if (.not. allocated(f)) allocate(f(shr_emul_flux_nchan, lsize_x*lsize_y))
    call shr_emul_flux_get(f, lsize_x, lsize_y)

    n = 0
    do j = 1, lsize_y
       do i = 1, lsize_x
          n = n + 1
          raw_taux(i,j)  = raw_taux(i,j)  + f(shr_emul_flux_taux, n)
          raw_tauy(i,j)  = raw_tauy(i,j)  + f(shr_emul_flux_tauy, n)
          raw_prec(i,j)  = raw_prec(i,j)  + f(shr_emul_flux_prec, n)
          raw_snow(i,j)  = raw_snow(i,j)  + f(shr_emul_flux_snow, n)
          raw_flus(i,j)  = raw_flus(i,j)  + f(shr_emul_flux_flus, n)
          raw_fsus(i,j)  = raw_fsus(i,j)  + f(shr_emul_flux_fsus, n)
          raw_flds(i,j)  = raw_flds(i,j)  + f(shr_emul_flux_flds, n)
          raw_fsds(i,j)  = raw_fsds(i,j)  + f(shr_emul_flux_fsds, n)
          raw_lhflx(i,j) = raw_lhflx(i,j) + f(shr_emul_flux_lhflx,n)
          raw_shflx(i,j) = raw_shflx(i,j) + f(shr_emul_flux_shflx,n)
       end do
    end do

    raw_n = raw_n + 1

  end subroutine ocn_import_atm_raw

  !===============================================================================
  subroutine ocn_export_mct( o2x_o )

    implicit none
    type(mct_aVect), intent(inout) :: o2x_o

    integer :: i, j, n

    n = 0
    do j = 1, lsize_y
       do i = 1, lsize_x
          n = n + 1
          if (index_o2x_So_t    > 0) o2x_o%rAttr(index_o2x_So_t,   n) = so_t(i,j)
          if (index_o2x_So_s    > 0) o2x_o%rAttr(index_o2x_So_s,   n) = so_s(i,j)
          if (index_o2x_So_u    > 0) o2x_o%rAttr(index_o2x_So_u,   n) = so_u(i,j)
          if (index_o2x_So_v    > 0) o2x_o%rAttr(index_o2x_So_v,   n) = so_v(i,j)
          if (index_o2x_So_dhdx > 0) o2x_o%rAttr(index_o2x_So_dhdx,n) = so_dhdx(i,j)
          if (index_o2x_So_dhdy > 0) o2x_o%rAttr(index_o2x_So_dhdy,n) = so_dhdy(i,j)
          if (index_o2x_So_ssh  > 0) o2x_o%rAttr(index_o2x_So_ssh, n) = so_ssh(i,j)
          ! The emulator carries its own sea ice internally and does not report
          ! a freeze/melt potential, so nothing is offered to a sea ice model.
          if (index_o2x_Fioo_q  > 0) o2x_o%rAttr(index_o2x_Fioo_q, n) = 0.0_R8
       end do
    end do

    call eocn_scan_avect(o2x_o, 'o2x')

  end subroutine ocn_export_mct

  !===============================================================================
  subroutine eocn_scan_avect( av, label )

    ! Report any coupler field leaving the emulator that is not a finite,
    ! plausibly physical number.  A NaN or a 1e30 here is invisible until it
    ! surfaces a dozen routines later as a negative absolute temperature in
    ! the microphysics, which is a bad place to start looking.

    implicit none
    type(mct_aVect) , intent(in) :: av
    character(len=*), intent(in) :: label

    integer :: k, n, nbad
    real(r8) :: v, vmin, vmax
    character(len=64) :: fname

    do k = 1, mct_aVect_nRAttr(av)
       nbad = 0
       vmin =  1.0e30_r8
       vmax = -1.0e30_r8
       do n = 1, mct_aVect_lsize(av)
          v = av%rAttr(k,n)
          if (.not. (v == v) .or. abs(v) > 1.0e20_r8) then
             nbad = nbad + 1
          else
             vmin = min(vmin, v)
             vmax = max(vmax, v)
          end if
       end do
       if (nbad > 0) then
          fname = mct_aVect_getRList2c(k, av)
          write(logunit_ocn,'(a,i6,a)') '(eocn_scan_avect) '//trim(label)//' '// &
               trim(fname)//': ', nbad, ' non-finite or out of range values'
          call shr_sys_flush(logunit_ocn)
       end if
    end do

  end subroutine eocn_scan_avect

  !===============================================================================
  subroutine eocn_read_namelist()

    implicit none

    integer :: nu_nml, nml_error
    character(len=*), parameter :: subname = '(eocn_read_namelist) '

    namelist /eocn_inparm/ do_eocn, filename_eocn, eocn_emulator, &
         eocn_model_file, eocn_ic_file, eocn_model_device, eocn_rng_seed, &
         eocn_interp_state, &
         eocn_flux_ifrac_unweight, eocn_unweight_stress, eocn_precip_units, &
         eocn_forcing_source

    do_eocn           = .true.
    filename_eocn     = ' '
    eocn_emulator     = 'SamudrACE-E3SMv3'
    eocn_model_file   = ' '
    eocn_ic_file      = ' '
    eocn_model_device = 'gpu'
    eocn_rng_seed     = 0
    eocn_interp_state = .true.
    eocn_flux_ifrac_unweight = .true.
    eocn_unweight_stress     = .false.
    eocn_precip_units        = 'kg/m2/s'
    eocn_forcing_source      = 'atm_raw'   ! must match namelist_defaults_eocn.xml

    if (masterproc) then
       nu_nml = shr_file_getUnit()
       open( nu_nml, file='eocn_in'//trim(inst_suffix), status='old', &
            iostat=nml_error )
       if (nml_error /= 0) call shr_sys_abort(trim(subname)// &
            ' ERROR: cannot open eocn_in'//trim(inst_suffix))
       read(nu_nml, nml=eocn_inparm, iostat=nml_error)
       if (nml_error /= 0) call shr_sys_abort(trim(subname)// &
            ' ERROR: reading namelist eocn_inparm')
       close(nu_nml)
       call shr_file_freeUnit(nu_nml)

       write(logunit_ocn,*) ' '
       write(logunit_ocn,*) 'eocn_inparm:'
       write(logunit_ocn,*) '   do_eocn           = ', do_eocn
       write(logunit_ocn,*) '   filename_eocn     = ', trim(filename_eocn)
       write(logunit_ocn,*) '   eocn_emulator     = ', trim(eocn_emulator)
       write(logunit_ocn,*) '   eocn_model_file   = ', trim(eocn_model_file)
       write(logunit_ocn,*) '   eocn_ic_file      = ', trim(eocn_ic_file)
       write(logunit_ocn,*) '   eocn_model_device = ', trim(eocn_model_device)
       write(logunit_ocn,*) '   eocn_rng_seed     = ', eocn_rng_seed
       write(logunit_ocn,*) '   eocn_interp_state = ', eocn_interp_state
       write(logunit_ocn,*) '   eocn_flux_ifrac_unweight = ', eocn_flux_ifrac_unweight
       write(logunit_ocn,*) '   eocn_unweight_stress     = ', eocn_unweight_stress
       write(logunit_ocn,*) '   eocn_precip_units        = ', trim(eocn_precip_units)
       write(logunit_ocn,*) '   eocn_forcing_source      = ', trim(eocn_forcing_source)
    end if

    call shr_mpi_bcast(do_eocn,           mpicom_ocn)
    call shr_mpi_bcast(filename_eocn,     mpicom_ocn)
    call shr_mpi_bcast(eocn_emulator,     mpicom_ocn)
    call shr_mpi_bcast(eocn_model_file,   mpicom_ocn)
    call shr_mpi_bcast(eocn_ic_file,      mpicom_ocn)
    call shr_mpi_bcast(eocn_model_device, mpicom_ocn)
    call shr_mpi_bcast(eocn_rng_seed,     mpicom_ocn)
    call shr_mpi_bcast(eocn_interp_state, mpicom_ocn)
    call shr_mpi_bcast(eocn_flux_ifrac_unweight, mpicom_ocn)
    call shr_mpi_bcast(eocn_unweight_stress,     mpicom_ocn)
    call shr_mpi_bcast(eocn_precip_units,        mpicom_ocn)
    call shr_mpi_bcast(eocn_forcing_source,      mpicom_ocn)

    ! Validate here, not where the value is first used.  samudra_import_forcing
    ! only runs on an emulator-step boundary, so a typo there would survive
    ! initialisation, the grid setup and a ~380 MB TorchScript load and only
    ! abort five simulated days in.
    select case (trim(eocn_forcing_source))
    case ('atm_raw', 'coupler')
      ! ok
    case default
      call shr_sys_abort(trim(subname)//' ERROR: eocn_forcing_source must be '// &
           '"atm_raw" or "coupler", got "'//trim(eocn_forcing_source)//'"')
    end select

  end subroutine eocn_read_namelist

  !===============================================================================
  subroutine ocn_read_eocn()

    implicit none

    logical :: found
    integer, dimension(:) :: grid_dims(2)
    character(len=*),parameter :: subname = '(ocn_read_eocn) '
    type(file_desc_t) :: ncid

    if (masterproc) then
       write(logunit_ocn,*) 'Read in eocn file name: ',trim(filename_eocn)
       call shr_sys_flush(logunit_ocn)
    endif

    call ncd_pio_openfile(ncid, trim(filename_eocn), 0)
    call ncd_inqfdims(ncid, gsize)

    call ncd_io(varname='grid_dims', data=grid_dims, flag='read', ncid=ncid, readvar=found)
    if ( .not. found ) call shr_sys_abort( trim(subname)//' ERROR: reading EOCN grid_dims')

    lsize   = gsize
    lsize_x = grid_dims(1)
    lsize_y = grid_dims(2)

    if (masterproc) then
       write(logunit_ocn,*) 'Values for lon/lat: ', lsize_x, lsize_y
       call shr_sys_flush(logunit_ocn)
    endif

    allocate(lonc_g(gsize))
    allocate(latc_g(gsize))
    allocate(areac_g(gsize))

    call ncd_io(ncid=ncid, varname='grid_center_lon', flag='read', data=lonc_g, dim1name='grid_size', readvar=found)
    if ( .not. found ) call shr_sys_abort( trim(subname)//' ERROR: read eocn longitudes')

    call ncd_io(ncid=ncid, varname='grid_center_lat', flag='read', data=latc_g, dim1name='grid_size', readvar=found)
    if ( .not. found ) call shr_sys_abort( trim(subname)//' ERROR: read eocn latitudes')

    call ncd_io(ncid=ncid, varname='grid_area', flag='read', data=areac_g, dim1name='grid_size', readvar=found)
    if ( .not. found ) call shr_sys_abort( trim(subname)//' ERROR: read eocn area')

    call ncd_pio_closefile(ncid)

  end subroutine ocn_read_eocn

end module ocn_comp_mct
