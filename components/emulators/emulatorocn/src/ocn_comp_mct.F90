module ocn_comp_mct
   !------------------------------------------------------------------------
   ! MCT cap for the emulated ocean.  The work is emulator_mct_cap's; this
   ! adds the ocean's field lists and infodata.
   !------------------------------------------------------------------------

   use esmf
   use mct_mod
   use seq_cdata_mod,    only: seq_cdata, seq_cdata_setptrs
   use seq_infodata_mod, only: seq_infodata_type, seq_infodata_putdata
   use seq_flds_mod,     only: seq_flds_o2x_fields, seq_flds_x2o_fields
   use emulator_mct_cap
   use emulator_f2c_api, only: emulator_get_nx, emulator_get_ny, &
                               emulator_create_ocn

   implicit none
   private

   public :: ocn_init_mct, ocn_run_mct, ocn_final_mct

   type(emulator_cap) :: cap

CONTAINS

   subroutine ocn_init_mct(EClock, cdata, x2o, o2x, NLFilename)
      type(ESMF_Clock),           intent(inout) :: EClock
      type(seq_cdata),            intent(inout) :: cdata
      type(mct_aVect),            intent(inout) :: x2o, o2x
      character(len=*), optional, intent(in)    :: NLFilename

      type(seq_infodata_type), pointer :: infodata

      call seq_cdata_setptrs(cdata, infodata=infodata)
      call seq_infodata_PutData(infodata, ocn_present=.true., &
         ocn_prognostic=.true., ocnrof_prognostic=.false.)
      call emulator_cap_init(cap, 'ocn', emulator_create_ocn, EClock, cdata, x2o, o2x, &
                             seq_flds_x2o_fields, seq_flds_o2x_fields)
      call seq_infodata_PutData(infodata, &
         ocn_nx=emulator_get_nx(cap%handle), &
         ocn_ny=emulator_get_ny(cap%handle))
   end subroutine ocn_init_mct

   subroutine ocn_run_mct(EClock, cdata, x2o, o2x)
      type(ESMF_Clock), intent(inout) :: EClock
      type(seq_cdata),  intent(inout) :: cdata
      type(mct_aVect),  intent(inout) :: x2o, o2x

      call emulator_cap_run(cap, EClock)
   end subroutine ocn_run_mct

   subroutine ocn_final_mct(EClock, cdata, x2o, o2x)
      type(ESMF_Clock), intent(inout) :: EClock
      type(seq_cdata),  intent(inout) :: cdata
      type(mct_aVect),  intent(inout) :: x2o, o2x

      call emulator_cap_final(cap)
   end subroutine ocn_final_mct

end module ocn_comp_mct
