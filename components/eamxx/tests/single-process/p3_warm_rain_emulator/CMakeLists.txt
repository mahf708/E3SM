include (ScreamUtils)

# The warm-rain emulator replaces P3's warm-rain rates between kernels
if (NOT SCREAM_P3_SMALL_KERNELS)
  return()
endif()

# AD configurable options. The emulator's targets are 100-s mean rates
set (ATM_TIME_STEP 100)
set (NUM_STEPS 3)
set (RUN_T0 2021-10-12-45000)

CreateADUnitTestExec(p3_warm_rain_emulator
  LIBS p3)

# Device-native backend
set (BACKEND kokkos)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_${BACKEND}.yaml)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/output.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/output_${BACKEND}.yaml)
CreateUnitTestFromExec(p3_warm_rain_emulator_${BACKEND} p3_warm_rain_emulator
  EXE_ARGS "--args -ifile=input_${BACKEND}.yaml"
  LABELS p3 physics
  FIXTURES_SETUP p3_warm_rain_emulator_${BACKEND})

if (EAMXX_ENABLE_PYTHON)
  # Python backend, running a numpy implementation of the same model
  set (BACKEND python)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_${BACKEND}.yaml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/output.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/output_${BACKEND}.yaml)
  CreateUnitTestFromExec(p3_warm_rain_emulator_${BACKEND} p3_warm_rain_emulator
    EXE_ARGS "--args -ifile=input_${BACKEND}.yaml"
    LABELS p3 physics
    FIXTURES_SETUP p3_warm_rain_emulator_${BACKEND})

  # Both backends must agree (up to roundoff)
  set (SRC_FILE p3_warm_rain_emulator_kokkos.INSTANT.nsteps_x1.np1.${RUN_T0}.nc)
  set (TGT_FILE p3_warm_rain_emulator_python.INSTANT.nsteps_x1.np1.${RUN_T0}.nc)
  set (TEST_NAME p3_warm_rain_emulator_kokkos_vs_python)
  add_test (NAME ${TEST_NAME}
    COMMAND ${SCREAM_BASE_DIR}/scripts/compare-nc-files
    -s ${SRC_FILE} -t ${TGT_FILE} --tol 1e-10
    -c qc=qc nc=nc qr=qr nr=nr T_mid=T_mid
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(${TEST_NAME} PROPERTIES
    LABELS "p3;physics"
    FIXTURES_REQUIRED "p3_warm_rain_emulator_kokkos;p3_warm_rain_emulator_python")
endif()
