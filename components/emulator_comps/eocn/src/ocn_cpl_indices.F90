module ocn_cpl_indices

  use mct_mod
  use seq_flds_mod, only: seq_flds_x2o_fields, seq_flds_o2x_fields

  implicit none
  save
  public

  ! coupler -> eocn
  integer :: index_x2o_Si_ifrac   = 0   ! sea ice fraction the coupler merged with
  integer :: index_x2o_Foxx_taux  = 0   ! zonal wind stress on the ocean  (N/m2)
  integer :: index_x2o_Foxx_tauy  = 0   ! meridional wind stress          (N/m2)
  integer :: index_x2o_Foxx_swnet = 0   ! net shortwave into the ocean    (W/m2)
  integer :: index_x2o_Foxx_lat   = 0   ! latent   heat flux, into ocean  (W/m2)
  integer :: index_x2o_Foxx_sen   = 0   ! sensible heat flux, into ocean  (W/m2)
  integer :: index_x2o_Foxx_lwup  = 0   ! upward longwave, into ocean     (W/m2)
  integer :: index_x2o_Faxa_lwdn  = 0   ! downward longwave               (W/m2)
  integer :: index_x2o_Faxa_rain  = 0   ! liquid precipitation         (kg/m2/s)
  integer :: index_x2o_Faxa_snow  = 0   ! frozen precipitation         (kg/m2/s)

  ! eocn -> coupler
  integer :: index_o2x_So_t       = 0
  integer :: index_o2x_So_s       = 0
  integer :: index_o2x_So_u       = 0
  integer :: index_o2x_So_v       = 0
  integer :: index_o2x_So_dhdx    = 0
  integer :: index_o2x_So_dhdy    = 0
  integer :: index_o2x_So_ssh     = 0
  integer :: index_o2x_Fioo_q     = 0

contains

  subroutine ocn_cpl_indices_set()

    type(mct_aVect) :: o2x, x2o
    integer :: lsize = 1

    call mct_aVect_init(x2o, rList=seq_flds_x2o_fields, lsize=lsize)
    call mct_aVect_init(o2x, rList=seq_flds_o2x_fields, lsize=lsize)

    index_x2o_Si_ifrac   = mct_avect_indexra(x2o,'Si_ifrac'  ,perrWith='quiet')
    index_x2o_Foxx_taux  = mct_avect_indexra(x2o,'Foxx_taux' ,perrWith='quiet')
    index_x2o_Foxx_tauy  = mct_avect_indexra(x2o,'Foxx_tauy' ,perrWith='quiet')
    index_x2o_Foxx_swnet = mct_avect_indexra(x2o,'Foxx_swnet',perrWith='quiet')
    index_x2o_Foxx_lat   = mct_avect_indexra(x2o,'Foxx_lat'  ,perrWith='quiet')
    index_x2o_Foxx_sen   = mct_avect_indexra(x2o,'Foxx_sen'  ,perrWith='quiet')
    index_x2o_Foxx_lwup  = mct_avect_indexra(x2o,'Foxx_lwup' ,perrWith='quiet')
    index_x2o_Faxa_lwdn  = mct_avect_indexra(x2o,'Faxa_lwdn' ,perrWith='quiet')
    index_x2o_Faxa_rain  = mct_avect_indexra(x2o,'Faxa_rain' ,perrWith='quiet')
    index_x2o_Faxa_snow  = mct_avect_indexra(x2o,'Faxa_snow' ,perrWith='quiet')

    index_o2x_So_t       = mct_avect_indexra(o2x,'So_t'      ,perrWith='quiet')
    index_o2x_So_s       = mct_avect_indexra(o2x,'So_s'      ,perrWith='quiet')
    index_o2x_So_u       = mct_avect_indexra(o2x,'So_u'      ,perrWith='quiet')
    index_o2x_So_v       = mct_avect_indexra(o2x,'So_v'      ,perrWith='quiet')
    index_o2x_So_dhdx    = mct_avect_indexra(o2x,'So_dhdx'   ,perrWith='quiet')
    index_o2x_So_dhdy    = mct_avect_indexra(o2x,'So_dhdy'   ,perrWith='quiet')
    index_o2x_So_ssh     = mct_avect_indexra(o2x,'So_ssh'    ,perrWith='quiet')
    index_o2x_Fioo_q     = mct_avect_indexra(o2x,'Fioo_q'    ,perrWith='quiet')

    call mct_aVect_clean(x2o)
    call mct_aVect_clean(o2x)

  end subroutine ocn_cpl_indices_set

end module ocn_cpl_indices
