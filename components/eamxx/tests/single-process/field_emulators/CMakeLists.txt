include (ScreamUtils)

# Emulators of a whole atmosphere process (here P3), working on its fields
if (NOT EAMXX_ENABLE_PROCESS_EMULATORS OR NOT EMULATOR_ENABLE_PYTHON)
  return()
endif()

set (ATM_TIME_STEP 100)
set (NUM_STEPS 3)
set (RUN_T0 2021-10-12-45000)

CreateADUnitTestExec(field_emulators
  LIBS p3)

# The fields P3 updates or computes, given to the test emulator
set (P3_OUTPUTS "T_mid, qv, qc, nc, qr, nr, qi, qm, ni, bm, precip_liq_surf_mass, precip_ice_surf_mass, T_prev_micro_step, qv_prev_micro_step, eff_radius_qc, eff_radius_qi, eff_radius_qr, micro_liq_ice_exchange, micro_vap_liq_exchange, micro_vap_ice_exchange")
string (REPLACE ", " "_before, " P3_OUTPUTS_BEFORE "${P3_OUTPUTS}_before")

# Creates test field_emulators_${CASE}: P3 with the emulators in ${EMULATORS}, in mode ${MODE}
macro (CreateCase CASE EMULATORS MODE)
  # Macro arguments are not variables: make them visible to configure_file
  set (CASE ${CASE})
  set (EMULATORS ${EMULATORS})
  set (MODE ${MODE})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml ${CMAKE_CURRENT_BINARY_DIR}/input_${CASE}.yaml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/output.yaml ${CMAKE_CURRENT_BINARY_DIR}/output_${CASE}.yaml)
  CreateUnitTestFromExec(field_emulators_${CASE} field_emulators
    EXE_ARGS "--args -ifile=input_${CASE}.yaml"
    LABELS p3 physics
    FIXTURES_SETUP field_emulators_${CASE})
endmacro()

# Compares the outputs of tests field_emulators_${SRC} and field_emulators_${TGT}
macro (Compare SRC TGT)
  add_test (NAME field_emulators_${TGT}_vs_${SRC}
    COMMAND ${SCREAM_BASE_DIR}/scripts/compare-nc-files
    -s field_emulators_${SRC}.INSTANT.nsteps_x1.np1.${RUN_T0}.nc
    -t field_emulators_${TGT}.INSTANT.nsteps_x1.np1.${RUN_T0}.nc --tol 0
    -c T_mid=T_mid qv=qv qc=qc qr=qr qi=qi qm=qm nc=nc nr=nr ni=ni bm=bm
       precip_liq_surf_mass=precip_liq_surf_mass precip_ice_surf_mass=precip_ice_surf_mass
       eff_radius_qc=eff_radius_qc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(field_emulators_${TGT}_vs_${SRC} PROPERTIES
    LABELS "p3;physics"
    FIXTURES_REQUIRED "field_emulators_${SRC};field_emulators_${TGT}")
endmacro()

# Stock P3
CreateCase(stock "" replace)

# After P3, an emulator returns P3's outputs: BFB with stock P3
CreateCase(after_identity "identity" after)
Compare(stock after_identity)

# After P3, an emulator returns the outputs as they were before P3: P3 is undone.
# Must be the same as an emulator that replaces P3, and returns its inputs.
CreateCase(after_undo "undo" after)
CreateCase(replace_identity "identity" replace)
Compare(after_undo replace_identity)
