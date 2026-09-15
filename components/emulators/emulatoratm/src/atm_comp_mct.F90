module atm_comp_mct
   !------------------------------------------------------------------------
   ! MCT cap for the emulated atmosphere.  The work is emulator_mct_cap's;
   ! this adds the atmosphere's field lists and infodata.
   !------------------------------------------------------------------------

   use esmf
   use mct_mod
   use seq_cdata_mod,    only: seq_cdata, seq_cdata_setptrs
   use seq_infodata_mod, only: seq_infodata_type, seq_infodata_getdata, &
                               seq_infodata_putdata
   use seq_timemgr_mod,  only: seq_timemgr_EClockGetData
   use seq_flds_mod,     only: seq_flds_a2x_fields, seq_flds_x2a_fields
   use shr_kind_mod,     only: R8=>SHR_KIND_R8
   use emulator_mct_cap
   use emulator_f2c_api, only: emulator_get_nx, emulator_get_ny, &
                               emulator_create_atm

   implicit none
   private

   public :: atm_init_mct, atm_run_mct, atm_final_mct

   type(emulator_cap) :: cap

CONTAINS

   subroutine atm_init_mct(EClock, cdata, x2a, a2x, NLFilename)
      type(ESMF_Clock),           intent(inout) :: EClock
      type(seq_cdata),            intent(inout) :: cdata
      type(mct_aVect),            intent(inout) :: x2a, a2x
      character(len=*), optional, intent(in)    :: NLFilename

      type(seq_infodata_type), pointer :: infodata
      integer :: phase

      call seq_cdata_setptrs(cdata, infodata=infodata)
      call seq_infodata_getData(infodata, atm_phase=phase)
      if (phase > 1) return

      call seq_infodata_PutData(infodata, atm_prognostic=.true.)
      call emulator_cap_init(cap, 'atm', emulator_create_atm, EClock, cdata, x2a, a2x, &
                             seq_flds_x2a_fields, seq_flds_a2x_fields)
      call seq_infodata_PutData(infodata, &
         atm_nx=emulator_get_nx(cap%handle), &
         atm_ny=emulator_get_ny(cap%handle))
      call put_nextsw_cday(EClock, infodata)
   end subroutine atm_init_mct

   subroutine atm_run_mct(EClock, cdata, x2a, a2x)
      type(ESMF_Clock), intent(inout) :: EClock
      type(seq_cdata),  intent(inout) :: cdata
      type(mct_aVect),  intent(inout) :: x2a, a2x

      type(seq_infodata_type), pointer :: infodata

      call seq_cdata_setptrs(cdata, infodata=infodata)
      call emulator_cap_run(cap, EClock)
      call put_nextsw_cday(EClock, infodata)
   end subroutine atm_run_mct

   subroutine atm_final_mct(EClock, cdata, x2a, a2x)
      type(ESMF_Clock), intent(inout) :: EClock
      type(seq_cdata),  intent(inout) :: cdata
      type(mct_aVect),  intent(inout) :: x2a, a2x

      call emulator_cap_final(cap)
   end subroutine atm_final_mct

   subroutine put_nextsw_cday(EClock, infodata)
      ! The surface components' albedos are for the next shortwave step.
      type(ESMF_Clock),        intent(inout) :: EClock
      type(seq_infodata_type), pointer       :: infodata
      real(R8) :: nextsw_cday
      call seq_timemgr_EClockGetData(EClock, next_cday=nextsw_cday)
      call seq_infodata_PutData(infodata, nextsw_cday=nextsw_cday)
   end subroutine put_nextsw_cday

end module atm_comp_mct
