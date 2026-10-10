include (ScreamUtils)

# Emulators run between the kernels computing and applying P3's process rates
if (NOT SCREAM_P3_SMALL_KERNELS OR NOT EAMXX_ENABLE_PROCESS_EMULATORS)
  return()
endif()

set (ATM_TIME_STEP 100)
set (NUM_STEPS 3)
set (RUN_T0 2021-10-12-45000)

CreateADUnitTestExec(p3_process_emulators
  LIBS p3)

# Rates emulated by the test emulators, which return P3's own rates: P3 must be BFB with stock P3
set (WARM_RATES "qc2qr_autoconv_tend, nc2nr_autoconv_tend, ncautr, nc_selfcollect_tend, qc2qr_accret_tend, nc_accret_tend, nr_selfcollect_tend")
set (ICE_RATES  "qi2qv_sublim_tend, qv2qi_vapdep_tend, qc2qi_collect_tend, nc_collect_tend, qr2qi_collect_tend, nr_collect_tend, ni_selfcollect_tend, qc2qi_berg_tend")
set (ALL_RATES  "${WARM_RATES}, qr2qv_evap_tend, nr_evap_tend, ${ICE_RATES}, nr_ice_shed_tend, qc2qi_hetero_freeze_tend, qc2qr_ice_shed_tend, qi2qr_melt_tend, qr2qi_immers_freeze_tend, qv2qi_nucleat_tend, ni2nr_melt_tend, ncshdc, nc2ni_immers_freeze_tend, ni_nucleat_tend, nr2ni_immers_freeze_tend, ni_sublim_tend, ncheti_cnt, qcheti_cnt, nicnt, qicnt, ninuc_cnt, qinuc_cnt, rho_qm_cloud, wetgrowth")
set (ICE_MODEL ${CMAKE_CURRENT_BINARY_DIR}/ice_identity.pt)

set (SED_TENDS "qc_sed_tend, nc_sed_tend, qr_sed_tend, nr_sed_tend, qi_sed_tend, ni_sed_tend, qm_sed_tend, bm_sed_tend")

# Creates test p3_process_emulators_${CASE}, running the process emulators in ${EMULATORS},
# and the sedimentation emulators in the optional third argument
macro (CreateCase CASE EMULATORS)
  # Macro arguments are not variables: make them visible to configure_file
  set (CASE ${CASE})
  set (EMULATORS ${EMULATORS})
  set (SED_EMULATORS "${ARGN}")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml ${CMAKE_CURRENT_BINARY_DIR}/input_${CASE}.yaml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/output.yaml ${CMAKE_CURRENT_BINARY_DIR}/output_${CASE}.yaml)
  CreateUnitTestFromExec(p3_process_emulators_${CASE} p3_process_emulators
    EXE_ARGS "--args -ifile=input_${CASE}.yaml"
    LABELS p3 physics
    FIXTURES_SETUP p3_process_emulators_${CASE})
endmacro()

# Compares the output of test p3_process_emulators_${CASE} with the stock one,
# within the optional tolerance TOL (default 0)
macro (CompareWithStock CASE)
  set (TOL 0)
  if (${ARGC} GREATER 1)
    set (TOL ${ARGV1})
  endif()
  set (SRC_FILE p3_process_emulators_stock.INSTANT.nsteps_x1.np1.${RUN_T0}.nc)
  set (TGT_FILE p3_process_emulators_${CASE}.INSTANT.nsteps_x1.np1.${RUN_T0}.nc)
  add_test (NAME p3_process_emulators_${CASE}_vs_stock
    COMMAND ${SCREAM_BASE_DIR}/scripts/compare-nc-files
    -s ${SRC_FILE} -t ${TGT_FILE} --tol ${TOL}
    -c T_mid=T_mid qv=qv qc=qc qr=qr qi=qi qm=qm nc=nc nr=nr ni=ni bm=bm
       precip_liq_surf_mass=precip_liq_surf_mass precip_ice_surf_mass=precip_ice_surf_mass
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(p3_process_emulators_${CASE}_vs_stock PROPERTIES
    LABELS "p3;physics"
    FIXTURES_REQUIRED "p3_process_emulators_stock;p3_process_emulators_${CASE}")
endmacro()

# Stock P3
CreateCase(stock "")

if (EMULATOR_ENABLE_PYTHON)
  # All the rates, through the python backend
  CreateCase(all_rates "all_rates")
  CompareWithStock(all_rates)

  # All the sedimentation tendencies and surface precipitation. Tendencies are
  # re-applied to the state before sedimentation, x_before + (x - x_before)/dt*dt,
  # which is not BFB, only to round-off
  CreateCase(sedimentation "" "sed_all")
  CompareWithStock(sedimentation 1e-12)

  # The tendencies only: the surface precipitation is diagnosed from them
  CreateCase(sedimentation_diagnosed "" "sed_tends")
  CompareWithStock(sedimentation_diagnosed 1e-12)
endif()

if (EMULATOR_ENABLE_PYTHON AND EMULATOR_ENABLE_LIBTORCH)
  # Two emulators in a chain, of different processes, with different backends
  set (EMULATOR_TORCH_PYTHON "python3" CACHE STRING "Python interpreter with torch, to write TorchScript test models")
  execute_process(COMMAND ${EMULATOR_TORCH_PYTHON} -c "import torch"
    RESULT_VARIABLE NO_TORCH_PYTHON OUTPUT_QUIET ERROR_QUIET)
  if (NO_TORCH_PYTHON)
    message(WARNING "Skipping test p3_process_emulators_chain: '${EMULATOR_TORCH_PYTHON}' cannot import torch.")
  else()
    string(REPLACE "," ";" ICE_RATES_LIST "${ICE_RATES}")
    list(LENGTH ICE_RATES_LIST NUM_ICE_RATES)
    add_custom_command(OUTPUT ${ICE_MODEL}
      COMMAND ${EMULATOR_TORCH_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/make_identity_torchscript.py
              ${ICE_MODEL} ${NUM_ICE_RATES}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/make_identity_torchscript.py
      COMMENT "Writing the TorchScript model of test p3_process_emulators_chain")
    add_custom_target(p3_process_emulators_models DEPENDS ${ICE_MODEL})
    add_dependencies(p3_process_emulators p3_process_emulators_models)

    CreateCase(chain "warm, ice")
    CompareWithStock(chain)
  endif()
endif()
