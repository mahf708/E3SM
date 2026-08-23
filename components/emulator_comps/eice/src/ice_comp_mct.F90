module ice_comp_mct

  !-----------------------------------------------------------------------------
  ! EICE -- the sea ice half of the emulated ocean.
  !
  ! Samudra predicts its own sea ice.  ocean_sea_ice_fraction is one of its
  ! eighty output channels, and in SamudrACE that fraction is what the coupler
  ! hands the atmosphere:
  !
  !     ICEFRAC = ocean_sea_ice_fraction * (1 - LANDFRAC)
  !
  ! (fme/core/ocean_data.py:194).  E3SM writes the same identity as
  ! lfrac + ifrac + ofrac = 1, but it fills ifrac from a sea ice *component*.
  ! With a stub ice there is none, so the coupler tells the atmosphere the
  ! polar ocean is open water at the freezing point all year -- and measurement
  ! showed that costs about 70 W/m2 in the northern polar band.
  !
  ! This component exists to put that number where the coupler can see it.  It
  ! reports Si_ifrac = ocean_sea_ice_fraction, and seq_frac_mct.F90:651 does
  ! the rest:
  !
  !     fractions_i(ifrac) = Si_ifrac * dom_i%frac
  !     fractions_i(ofrac) = dom_i%frac - fractions_i(ifrac)
  !
  ! which is the SamudrACE identity exactly, with dom_i%frac playing the part
  ! of (1 - LANDFRAC).  No conversion, because Si_ifrac is already defined as a
  ! proportion of the ice domain's fraction.
  !
  ! Everything else here -- surface temperature, albedos, the atm/ice turbulent
  ! fluxes -- is dice in prescribed mode, which is what an AMIP run already
  ! uses for its ice.  Samudra gives a fraction and an ice volume, not a
  ! surface energy balance, so a slab is the honest ceiling.  The fabricated
  ! seasonal surface temperature (dice_comp_mod.F90:670) is the weakest part
  ! and is the first thing to improve if the polar atmosphere still looks
  ! wrong.
  !
  ! It carries no state of its own: the fraction arrives from the ocean every
  ! step and the fluxes are diagnosed from it, so there is no restart file and
  ! nothing to keep bit-for-bit across one.
  !
  ! The MCT driver runs ice before ocn within a coupling step, so the fraction
  ! read here is the one EOCN published at the end of the previous step.  EOCN
  ! republishes every step from its blended state (samudra_comp_mod.F90:252),
  ! so this is a one coupling step lag -- thirty minutes on a field whose
  ! underlying emulator step is five days.
  !
  ! The grid is not read from disk.  It is published by the ocean at init
  ! (shr_emul_ice_put_grid) so that the two halves of the entity cannot
  ! disagree about the mesh, and so that a mismatched decomposition is a hard
  ! error rather than a silent mis-index.
  !-----------------------------------------------------------------------------

  use esmf
  use mct_mod
  use perf_mod
  use seq_cdata_mod    , only: seq_cdata, seq_cdata_setptrs
  use seq_infodata_mod , only: seq_infodata_type, seq_infodata_PutData, seq_infodata_GetData
  use seq_comm_mct     , only: seq_comm_inst, seq_comm_name, seq_comm_suffix
  use seq_timemgr_mod  , only: seq_timemgr_EClockGetData
  use seq_flds_mod     , only: seq_flds_i2x_fields, seq_flds_x2i_fields
  use seq_flds_mod     , only: seq_flds_dom_coord, seq_flds_dom_other
  use shr_kind_mod     , only: IN=>SHR_KIND_IN, R8=>SHR_KIND_R8, CL=>SHR_KIND_CL
  use shr_const_mod    , only: shr_const_tkfrz, shr_const_pi, shr_const_tkfrzsw
  use shr_file_mod     , only: shr_file_getunit, shr_file_getlogunit, shr_file_getloglevel
  use shr_file_mod     , only: shr_file_setlogunit, shr_file_setloglevel, shr_file_setio
  use shr_sys_mod      , only: shr_sys_abort, shr_sys_flush
  use shr_mpi_mod      , only: shr_mpi_commrank, shr_mpi_commsize
  use shr_cal_mod      , only: shr_cal_ymd2julian, shr_cal_date2ymd
  use shr_emul_ice_mod , only: shr_emul_ice_get, shr_emul_ice_avail
  use shr_emul_ice_mod , only: shr_emul_ice_get_grid, shr_emul_ice_grid_size
  use shr_emul_ice_mod , only: shr_emul_ice_grid_gsize
  use eice_flux_atmice_mod, only: eice_flux_atmice

  implicit none
  private
  save

  public :: ice_init_mct
  public :: ice_run_mct
  public :: ice_final_mct

  integer(IN), parameter :: master_task = 0

  integer(IN) :: compid
  integer(IN) :: mpicom_ice
  integer(IN) :: iam, npes
  logical     :: masterproc
  integer(IN) :: logunit_ice
  character(CL) :: inst_name, inst_suffix

  integer(IN) :: lsize, gsize

  ! grid, taken from the ocean
  real(R8),    allocatable :: lonc(:), latc(:), areac(:), fracc(:)
  integer(IN), allocatable :: imask(:)

  ! working space for the emulator's fraction
  real(R8), allocatable :: ifrac_emul(:)

  ! Where the bulk flux scheme may be evaluated.  Not the same as imask: at
  ! init the coupler has not filled x2i yet, so the atmosphere state is all
  ! zeros, and the scheme divides by air density and takes log(zbot).
  integer(IN), allocatable :: fmask(:)

  ! x2i indices
  integer(IN) :: kswvdr, kswndr, kswvdf, kswndf
  integer(IN) :: kz, kua, kva, kptem, kshum, kdens, ktbot, ksalinity

  ! i2x indices
  integer(IN) :: kiFrac, kt, ktref, kqref
  integer(IN) :: kavsdr, kanidr, kavsdf, kanidf
  integer(IN) :: kswnet, ksen, klat, klwup, kevap, ktauxa, ktauya
  integer(IN) :: kmelth, kmeltw, kswpen, ktauxo, ktauyo, ksalt
  integer(IN) :: ksnowh, kithick

  ! Surface albedo constants, straight from dice_comp_mod.F90:66.  A snow
  ! covered ice surface, climatologically averaged -- no melt pond, no
  ! zenith-angle dependence.  These are the numbers an AMIP run uses.
  real(R8), parameter :: snwfrac = 0.286_R8
  real(R8), parameter :: as_nidf = 0.950_R8, as_vsdf = 0.700_R8
  real(R8), parameter :: as_nidr = 0.960_R8, as_vsdr = 0.800_R8
  real(R8), parameter :: ai_nidf = 0.700_R8, ai_vsdf = 0.500_R8
  real(R8), parameter :: ai_nidr = 0.700_R8, ai_vsdr = 0.500_R8
  real(R8), parameter :: ax_nidf = ai_nidf*(1.0_R8-snwfrac) + as_nidf*snwfrac
  real(R8), parameter :: ax_vsdf = ai_vsdf*(1.0_R8-snwfrac) + as_vsdf*snwfrac
  real(R8), parameter :: ax_nidr = ai_nidr*(1.0_R8-snwfrac) + as_nidr*snwfrac
  real(R8), parameter :: ax_vsdr = ai_vsdr*(1.0_R8-snwfrac) + as_vsdr*snwfrac

  real(R8), parameter :: tFrz    = shr_const_tkfrz
  ! Freezing point of sea water.  Deliberately a constant rather than
  ! shr_frz_freezetemp(So_s): that routine needs shr_frz_freezetemp_init,
  ! which only MPAS-Ocean calls, so with an emulator ocean it returns its
  ! uninitialised sentinel and aborts.  This is the same constant EOCN clamps
  ! its own SST to (samudra_comp_mod.F90:69), which keeps the two halves of
  ! the entity agreeing on where the ocean freezes.
  real(R8), parameter :: tFrzSw  = shr_const_tkfrzsw
  real(R8), parameter :: snowh_i = 0.20_R8   ! m, nominal snow depth on ice
  real(R8), parameter :: ithick_i = 2.00_R8  ! m, nominal ice thickness

CONTAINS

  !===============================================================================
  subroutine ice_init_mct( EClock, cdata, x2i, i2x, NLFilename )

    implicit none

    type(ESMF_Clock)           , intent(inout) :: EClock
    type(seq_cdata)            , intent(inout) :: cdata
    type(mct_aVect)            , intent(inout) :: x2i, i2x
    character(len=*), optional , intent(in)    :: NLFilename

    type(seq_infodata_type), pointer :: infodata
    type(mct_gsMap)        , pointer :: gsMap
    type(mct_gGrid)        , pointer :: ggrid

    integer(IN) :: shrlogunit, shrloglev
    logical     :: exists
    character(*), parameter :: subName = "(ice_init_mct) "

    call seq_cdata_setptrs(cdata, id=compid, mpicom=mpicom_ice, &
         gsMap=gsMap, dom=ggrid, infodata=infodata)

    call shr_mpi_commrank(mpicom_ice, iam)
    call shr_mpi_commsize(mpicom_ice, npes)
    masterproc = (iam == master_task)

    inst_name   = seq_comm_name(compid)
    inst_suffix = seq_comm_suffix(compid)

    call shr_file_getLogUnit (shrlogunit)
    call shr_file_getLogLevel(shrloglev)
    if (masterproc) then
       inquire(file='ice_modelio.nml'//trim(inst_suffix), exist=exists)
       if (exists) then
          logunit_ice = shr_file_getUnit()
          call shr_file_setIO('ice_modelio.nml'//trim(inst_suffix), logunit_ice)
       else
          logunit_ice = shrlogunit
       end if
       write(logunit_ice,*) 'eice model initialization'
    else
       logunit_ice = shrlogunit
    end if
    call shr_file_setLogUnit(logunit_ice)

    ! The ocean must have gone first.  The MCT driver initialises components in
    ! the order atm, lnd, rof, ocn, ice, so it always has -- but if EICE is ever
    ! paired with something other than EOCN, say so rather than run on zeros.
    lsize = shr_emul_ice_grid_size()
    gsize = shr_emul_ice_grid_gsize()
    if (lsize <= 0 .or. gsize <= 0) then
       call shr_sys_abort(trim(subName)//' ERROR: no emulator grid published. '// &
            'EICE requires EOCN as the ocean component.')
    end if

    allocate(lonc(lsize), latc(lsize), areac(lsize), fracc(lsize), imask(lsize))
    call shr_emul_ice_get_grid(lonc, latc, areac, fracc)
    where (fracc > 0.0_R8)
       imask = 1
    elsewhere
       imask = 0
    end where

    allocate(ifrac_emul(lsize), fmask(lsize))
    ifrac_emul = 0.0_R8

    call ice_SetGSMap_mct(gsMap)
    call ice_domain_mct(gsMap, ggrid)

    call mct_aVect_init(x2i, rList=seq_flds_x2i_fields, lsize=lsize)
    call mct_aVect_zero(x2i)
    call mct_aVect_init(i2x, rList=seq_flds_i2x_fields, lsize=lsize)
    call mct_aVect_zero(i2x)

    call ice_set_indices(x2i, i2x)

    call seq_infodata_PutData(infodata, &
         ice_present    = .true., &
         ice_prognostic = .true., &
         iceberg_prognostic = .false., &
         ice_nx = 360, ice_ny = 180)

    ! Publish the fraction the ocean already computed at its own init, so the
    ! coupler's first fraction bookkeeping pass sees ice rather than an all
    ! open-water ocean.
    call ice_export_mct(EClock, x2i, i2x)

    if (masterproc) then
       write(logunit_ice,*) trim(subName)//' done, lsize = ', lsize, ' gsize = ', gsize
    end if
    call shr_sys_flush(logunit_ice)

    call shr_file_setLogUnit (shrlogunit)
    call shr_file_setLogLevel(shrloglev)

  end subroutine ice_init_mct

  !===============================================================================
  subroutine ice_run_mct( EClock, cdata, x2i, i2x )

    implicit none

    type(ESMF_Clock), intent(inout) :: EClock
    type(seq_cdata) , intent(inout) :: cdata
    type(mct_aVect) , intent(inout) :: x2i
    type(mct_aVect) , intent(inout) :: i2x

    integer(IN) :: shrlogunit, shrloglev

    call shr_file_getLogUnit (shrlogunit)
    call shr_file_getLogLevel(shrloglev)
    call shr_file_setLogUnit (logunit_ice)

    call t_startf('EICE_run')
    call ice_export_mct(EClock, x2i, i2x)
    call t_stopf('EICE_run')

    call shr_file_setLogUnit (shrlogunit)
    call shr_file_setLogLevel(shrloglev)

  end subroutine ice_run_mct

  !===============================================================================
  subroutine ice_final_mct( EClock, cdata, x2i, i2x )

    implicit none
    type(ESMF_Clock), intent(inout) :: EClock
    type(seq_cdata) , intent(inout) :: cdata
    type(mct_aVect) , intent(inout) :: x2i
    type(mct_aVect) , intent(inout) :: i2x

    if (allocated(lonc))  deallocate(lonc, latc, areac, fracc, imask)
    if (allocated(ifrac_emul)) deallocate(ifrac_emul, fmask)

  end subroutine ice_final_mct

  !===============================================================================
  subroutine ice_SetGSMap_mct( gsMap )

    ! Rebuild the ocean's decomposition rather than choosing one.  EOCN splits
    ! the mesh into contiguous blocks by task (ocn_comp_mct.F90:245); repeating
    ! that formula here gives an identical map whenever the two run on the same
    ! number of tasks, which is what makes the published grid arrays line up
    ! index for index.  If they do not, stop -- a silent mis-index would look
    ! like a plausible but wrong ice field.

    implicit none
    type(mct_gsMap), intent(inout) :: gsMap

    integer(IN) :: n
    integer(IN), allocatable :: start(:), length(:), pe_loc(:)
    character(*), parameter :: subName = "(ice_SetGSMap_mct) "

    allocate(start(npes), length(npes), pe_loc(npes))

    do n = 1, npes
       length(n) = gsize/npes
       if (n <= mod(gsize,npes)) length(n) = length(n) + 1
       if (n == 1) then
          start(n) = 1
       else
          start(n) = start(n-1) + length(n-1)
       end if
       pe_loc(n) = n-1
    end do

    if (length(iam+1) /= lsize) then
       write(logunit_ice,*) trim(subName)//' ERROR: ice decomposition ', &
            length(iam+1), ' does not match the ocean''s ', lsize
       call shr_sys_abort(trim(subName)//' ERROR: EICE and EOCN must run on '// &
            'the same number of tasks (set NTASKS_ICE = NTASKS_OCN)')
    end if

    call mct_gsMap_init(gsMap, compid, npes, gsize, start, length, pe_loc)

    deallocate(start, length, pe_loc)

  end subroutine ice_SetGSMap_mct

  !===============================================================================
  subroutine ice_domain_mct( gsMap, dom_ice )

    ! The ice domain is the ocean domain.  seq_domain_mct compares the two and
    ! requires them to agree; taking them from one source makes that check a
    ! tautology instead of a coincidence.

    implicit none
    type(mct_gsMap), intent(in)    :: gsMap
    type(mct_gGrid), intent(inout) :: dom_ice

    integer(IN)          :: n
    integer(IN), pointer :: idata(:)
    real(R8),    pointer :: data(:)

    call mct_gGrid_init(GGrid=dom_ice, CoordChars=trim(seq_flds_dom_coord), &
         OtherChars=trim(seq_flds_dom_other), lsize=lsize)

    allocate(data(lsize))

    call mct_gsMap_orderedPoints(gsMap, iam, idata)
    call mct_gGrid_importIAttr(dom_ice, 'GlobGridNum', idata, lsize)

    data(:) = -9999.0_R8
    call mct_gGrid_importRAttr(dom_ice, "aream", data, lsize)

    data(:) = lonc(:)
    call mct_gGrid_importRAttr(dom_ice, "lon",  data, lsize)
    data(:) = latc(:)
    call mct_gGrid_importRAttr(dom_ice, "lat",  data, lsize)
    data(:) = areac(:)
    call mct_gGrid_importRAttr(dom_ice, "area", data, lsize)
    data(:) = fracc(:)
    call mct_gGrid_importRAttr(dom_ice, "frac", data, lsize)
    do n = 1, lsize
       data(n) = real(imask(n), R8)
    end do
    call mct_gGrid_importRAttr(dom_ice, "mask", data, lsize)

    deallocate(data)
    deallocate(idata)

  end subroutine ice_domain_mct

  !===============================================================================
  subroutine ice_set_indices( x2i, i2x )

    implicit none
    type(mct_aVect), intent(in) :: x2i, i2x

    kswvdr    = mct_aVect_indexRA(x2i,'Faxa_swvdr', perrWith='quiet')
    kswndr    = mct_aVect_indexRA(x2i,'Faxa_swndr', perrWith='quiet')
    kswvdf    = mct_aVect_indexRA(x2i,'Faxa_swvdf', perrWith='quiet')
    kswndf    = mct_aVect_indexRA(x2i,'Faxa_swndf', perrWith='quiet')
    kz        = mct_aVect_indexRA(x2i,'Sa_z',       perrWith='quiet')
    kua       = mct_aVect_indexRA(x2i,'Sa_u',       perrWith='quiet')
    kva       = mct_aVect_indexRA(x2i,'Sa_v',       perrWith='quiet')
    kptem     = mct_aVect_indexRA(x2i,'Sa_ptem',    perrWith='quiet')
    kshum     = mct_aVect_indexRA(x2i,'Sa_shum',    perrWith='quiet')
    kdens     = mct_aVect_indexRA(x2i,'Sa_dens',    perrWith='quiet')
    ktbot     = mct_aVect_indexRA(x2i,'Sa_tbot',    perrWith='quiet')
    ksalinity = mct_aVect_indexRA(x2i,'So_s',       perrWith='quiet')

    kiFrac = mct_aVect_indexRA(i2x,'Si_ifrac')
    kt     = mct_aVect_indexRA(i2x,'Si_t')
    ktref  = mct_aVect_indexRA(i2x,'Si_tref',  perrWith='quiet')
    kqref  = mct_aVect_indexRA(i2x,'Si_qref',  perrWith='quiet')
    ksnowh = mct_aVect_indexRA(i2x,'Si_snowh', perrWith='quiet')
    kithick= mct_aVect_indexRA(i2x,'Si_ithick',perrWith='quiet')
    kavsdr = mct_aVect_indexRA(i2x,'Si_avsdr', perrWith='quiet')
    kanidr = mct_aVect_indexRA(i2x,'Si_anidr', perrWith='quiet')
    kavsdf = mct_aVect_indexRA(i2x,'Si_avsdf', perrWith='quiet')
    kanidf = mct_aVect_indexRA(i2x,'Si_anidf', perrWith='quiet')
    kswnet = mct_aVect_indexRA(i2x,'Faii_swnet', perrWith='quiet')
    ksen   = mct_aVect_indexRA(i2x,'Faii_sen',   perrWith='quiet')
    klat   = mct_aVect_indexRA(i2x,'Faii_lat',   perrWith='quiet')
    klwup  = mct_aVect_indexRA(i2x,'Faii_lwup',  perrWith='quiet')
    kevap  = mct_aVect_indexRA(i2x,'Faii_evap',  perrWith='quiet')
    ktauxa = mct_aVect_indexRA(i2x,'Faii_taux',  perrWith='quiet')
    ktauya = mct_aVect_indexRA(i2x,'Faii_tauy',  perrWith='quiet')
    kmelth = mct_aVect_indexRA(i2x,'Fioi_melth', perrWith='quiet')
    kmeltw = mct_aVect_indexRA(i2x,'Fioi_meltw', perrWith='quiet')
    kswpen = mct_aVect_indexRA(i2x,'Fioi_swpen', perrWith='quiet')
    ktauxo = mct_aVect_indexRA(i2x,'Fioi_taux',  perrWith='quiet')
    ktauyo = mct_aVect_indexRA(i2x,'Fioi_tauy',  perrWith='quiet')
    ksalt  = mct_aVect_indexRA(i2x,'Fioi_salt',  perrWith='quiet')

  end subroutine ice_set_indices

  !===============================================================================
  subroutine ice_export_mct( EClock, x2i, i2x )

    ! Take the emulator's fraction and dress it up as a sea ice component.

    implicit none
    type(ESMF_Clock), intent(inout) :: EClock
    type(mct_aVect) , intent(inout) :: x2i
    type(mct_aVect) , intent(inout) :: i2x

    integer(IN) :: n, cdate, sec, yy, mm, dd
    real(R8)    :: cosArg, jDay, jDay0, swnet
    character(CL) :: calendar
    character(*), parameter :: subName = "(ice_export_mct) "

    if (.not. shr_emul_ice_avail(lsize)) then
       ! The ocean has not published yet, or published on a different mesh.
       ! Report no ice rather than guess: an all open-water polar ocean is a
       ! known, documented error, while a mis-indexed fraction is not.
       ifrac_emul(:) = 0.0_R8
    else
       call shr_emul_ice_get(ifrac_emul)
    end if

    call seq_timemgr_EClockGetData(EClock, curr_ymd=cdate, curr_tod=sec, &
         calendar=calendar)
    call shr_cal_date2ymd(cdate, yy, mm, dd)
    call shr_cal_ymd2julian(0, mm, dd, sec, jDay,  calendar)
    call shr_cal_ymd2julian(0,  9,  1, 0,   jDay0, calendar)
    cosArg = 2.0_R8*shr_const_pi*(jDay - jDay0)/365.0_R8

    do n = 1, lsize

       if (imask(n) == 0) then
          i2x%rAttr(kiFrac,n) = 0.0_R8
          i2x%rAttr(kt,n)     = tFrzSw
          cycle
       end if

       ! The whole point of the component.  Si_ifrac is a proportion of the ice
       ! domain's fraction, and ocean_sea_ice_fraction is a proportion of the
       ! non-land area -- the same quantity, so this is a copy and not a
       ! conversion.
       i2x%rAttr(kiFrac,n) = min(1.0_R8, max(0.0_R8, ifrac_emul(n)))

       ! Fabricated surface temperature, from dice_comp_mod.F90:670.  Samudra
       ! carries no ice surface energy balance, so there is nothing better to
       ! use; this is what an AMIP run's ice does.
       !
       ! Reported as the ice skin temperature over the ice-covered part of the
       ! cell, and not blended towards freezing as the fraction vanishes, which
       ! is what dice does and what every consumer of Si_t expects.  A cell with
       ! a sliver of ice does not thereby report a 260 K skin: the coupler
       ! weights this by the same fraction on its way to the atmosphere,
       !     x2a%Sx_t = lfrac*Sl_t + ifrac*Si_t + ofrac*So_t
       ! (prep_atm_merge), so blending by ifrac here as well would apply the
       ! weight twice and leave the thermal anomaly scaled by ifrac**2.  The
       ! same value is what eice_flux_atmice_mod evaluates the bulk fluxes over
       ! the ice with, where a temperature pulled towards freezing by the open
       ! water beside it is simply the wrong surface to use.
       if (latc(n) > 0.0_R8) then
          i2x%rAttr(kt,n) = 260.0_R8 + 10.0_R8*cos(cosArg)
       else
          i2x%rAttr(kt,n) = 260.0_R8 - 10.0_R8*cos(cosArg)
       end if

       if (kavsdr > 0) i2x%rAttr(kavsdr,n) = ax_vsdr
       if (kanidr > 0) i2x%rAttr(kanidr,n) = ax_nidr
       if (kavsdf > 0) i2x%rAttr(kavsdf,n) = ax_vsdf
       if (kanidf > 0) i2x%rAttr(kanidf,n) = ax_nidf
       if (ksnowh > 0) i2x%rAttr(ksnowh,n) = snowh_i * i2x%rAttr(kiFrac,n)
       if (kithick> 0) i2x%rAttr(kithick,n)= ithick_i * i2x%rAttr(kiFrac,n)

       if (kswnet > 0 .and. kswvdr > 0) then
          swnet = (1.0_R8 - ax_vsdr)*x2i%rAttr(kswvdr,n) &
                + (1.0_R8 - ax_nidr)*x2i%rAttr(kswndr,n) &
                + (1.0_R8 - ax_vsdf)*x2i%rAttr(kswvdf,n) &
                + (1.0_R8 - ax_nidf)*x2i%rAttr(kswndf,n)
          i2x%rAttr(kswnet,n) = swnet
       end if

    end do

    ! Atm/ice turbulent fluxes, from the same bulk formulae dice uses.  EATM
    ! ignores these -- it diagnoses its own surface fluxes from its own state
    ! -- but EAM does not, so they have to be real numbers rather than zeros
    ! for the F2010-style compset to mean anything.
    !
    ! Evaluate them only where the atmosphere state is physical.  At init the
    ! coupler has not filled x2i, so it is all zeros, and the scheme divides by
    ! air density (eice_flux_atmice_mod.F90:144) and takes log(zbot) (:148).
    ! Both are NaN at zero, and a NaN reaching the merge survives the ifrac
    ! weighting that would have killed an ordinary large number -- which is how
    ! it takes down CLUBB on the first physics step rather than showing up as
    ! an obviously wrong flux.  EATM never reads these fields, so the emulated
    ! pair does not notice; EAM does.
    if (ksen > 0 .and. kz > 0) then
       do n = 1, lsize
          if (imask(n) /= 0 .and. x2i%rAttr(kdens,n) > 0.0_R8 &
                            .and. x2i%rAttr(kz,n)    > 0.0_R8) then
             fmask(n) = 1
          else
             fmask(n) = 0
          end if
       end do

       call eice_flux_atmice( &
            fmask              , x2i%rAttr(kz,:)    , x2i%rAttr(kua,:)   , &
            x2i%rAttr(kva,:)   , x2i%rAttr(kptem,:) , x2i%rAttr(kshum,:) , &
            x2i%rAttr(kdens,:) , x2i%rAttr(ktbot,:) , i2x%rAttr(kt,:)    , &
            i2x%rAttr(ksen,:)  , i2x%rAttr(klat,:)  , i2x%rAttr(klwup,:) , &
            i2x%rAttr(kevap,:) , i2x%rAttr(ktauxa,:), i2x%rAttr(ktauya,:), &
            i2x%rAttr(ktref,:) , i2x%rAttr(kqref,:) , logunit_ice )

       ! The scheme fills skipped cells with spval.  Zero them instead: the
       ! coupler merges these fields rather than testing them, and 1e30 times
       ! a small ifrac is still enormous.
       do n = 1, lsize
          if (fmask(n) == 0) then
             if (ksen   > 0) i2x%rAttr(ksen  ,n) = 0.0_R8
             if (klat   > 0) i2x%rAttr(klat  ,n) = 0.0_R8
             if (klwup  > 0) i2x%rAttr(klwup ,n) = 0.0_R8
             if (kevap  > 0) i2x%rAttr(kevap ,n) = 0.0_R8
             if (ktauxa > 0) i2x%rAttr(ktauxa,n) = 0.0_R8
             if (ktauya > 0) i2x%rAttr(ktauya,n) = 0.0_R8
             if (ktref  > 0) i2x%rAttr(ktref ,n) = i2x%rAttr(kt,n)
             if (kqref  > 0) i2x%rAttr(kqref ,n) = 0.0_R8
          end if
       end do
    end if

    ! Ice/ocean fluxes.
    !
    ! The melt and freeze terms are deliberately zero.  Samudra advances its
    ! own ice -- ocean_sea_ice_fraction and iceVolumeTotal are prognostic
    ! outputs -- so the heat and freshwater that go with a melting ice pack are
    ! already inside its step.  Handing them to it again through Fioi_melth and
    ! Fioi_meltw would count them twice.
    !
    ! The stress is passed through: the coupler will weight it by ifrac and the
    ! open-water stress by afrac, so setting the under-ice stress equal to the
    ! atm/ice stress keeps the total the ocean feels equal to what the
    ! atmosphere applied.  That matters because Samudra was trained on
    ! whole-cell wind stress.
    do n = 1, lsize
       if (kmelth > 0) i2x%rAttr(kmelth,n) = 0.0_R8
       if (kmeltw > 0) i2x%rAttr(kmeltw,n) = 0.0_R8
       if (ksalt  > 0) i2x%rAttr(ksalt ,n) = 0.0_R8
       if (kswpen > 0) i2x%rAttr(kswpen,n) = 0.0_R8
       if (ktauxo > 0 .and. ktauxa > 0) i2x%rAttr(ktauxo,n) = i2x%rAttr(ktauxa,n)
       if (ktauyo > 0 .and. ktauya > 0) i2x%rAttr(ktauyo,n) = i2x%rAttr(ktauya,n)
    end do

    call eice_scan_avect(i2x, 'i2x')

  end subroutine ice_export_mct

  !===============================================================================
  subroutine eice_scan_avect( av, label )

    ! Same check EOCN does on its own export, for the same reason: a NaN
    ! leaving here is invisible until it surfaces as a negative absolute
    ! temperature in the microphysics a dozen routines later.

    implicit none
    type(mct_aVect) , intent(in) :: av
    character(len=*), intent(in) :: label

    integer(IN) :: k, n, nbad
    real(R8)    :: v
    character(len=64) :: fname

    do k = 1, mct_aVect_nRAttr(av)
       nbad = 0
       do n = 1, mct_aVect_lsize(av)
          v = av%rAttr(k,n)
          if (.not. (v == v) .or. abs(v) > 1.0e20_R8) nbad = nbad + 1
       end do
       if (nbad > 0) then
          fname = mct_aVect_getRList2c(k, av)
          write(logunit_ice,'(a,i6,a)') '(eice_scan_avect) '//trim(label)//' '// &
               trim(fname)//': ', nbad, ' non-finite or out of range values'
          call shr_sys_flush(logunit_ice)
       end if
    end do

  end subroutine eice_scan_avect

end module ice_comp_mct
