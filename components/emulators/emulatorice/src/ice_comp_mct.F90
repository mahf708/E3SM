module ice_comp_mct
   !------------------------------------------------------------------------
   ! MCT cap for the emulated ocean's sea ice.  The work is
   ! emulator_mct_cap's; this adds the ice's field lists and infodata.  The
   ! component takes its domain from the emulated ocean, which the driver
   ! initializes first.
   !------------------------------------------------------------------------

   use esmf
   use mct_mod
   use seq_cdata_mod,    only: seq_cdata, seq_cdata_setptrs
   use seq_infodata_mod, only: seq_infodata_type, seq_infodata_putdata
   use seq_flds_mod,     only: seq_flds_i2x_fields, seq_flds_x2i_fields
   use emulator_mct_cap
   use emulator_f2c_api, only: emulator_get_nx, emulator_get_ny, &
                               emulator_create_ice

   implicit none
   private

   public :: ice_init_mct, ice_run_mct, ice_final_mct

   type(emulator_cap) :: cap

CONTAINS

   subroutine ice_init_mct(EClock, cdata, x2i, i2x, NLFilename)
      type(ESMF_Clock),           intent(inout) :: EClock
      type(seq_cdata),            intent(inout) :: cdata
      type(mct_aVect),            intent(inout) :: x2i, i2x
      character(len=*), optional, intent(in)    :: NLFilename

      type(seq_infodata_type), pointer :: infodata

      call seq_cdata_setptrs(cdata, infodata=infodata)
      call seq_infodata_PutData(infodata, ice_present=.true., &
         ice_prognostic=.true., iceberg_prognostic=.false.)
      call emulator_cap_init(cap, 'ice', emulator_create_ice, EClock, cdata, x2i, i2x, &
                             seq_flds_x2i_fields, seq_flds_i2x_fields)
      call seq_infodata_PutData(infodata, &
         ice_nx=emulator_get_nx(cap%handle), &
         ice_ny=emulator_get_ny(cap%handle))
   end subroutine ice_init_mct

   subroutine ice_run_mct(EClock, cdata, x2i, i2x)
      type(ESMF_Clock), intent(inout) :: EClock
      type(seq_cdata),  intent(inout) :: cdata
      type(mct_aVect),  intent(inout) :: x2i, i2x

      call emulator_cap_run(cap, EClock)
   end subroutine ice_run_mct

   subroutine ice_final_mct(EClock, cdata, x2i, i2x)
      type(ESMF_Clock), intent(inout) :: EClock
      type(seq_cdata),  intent(inout) :: cdata
      type(mct_aVect),  intent(inout) :: x2i, i2x

      call emulator_cap_final(cap)
   end subroutine ice_final_mct

end module ice_comp_mct
