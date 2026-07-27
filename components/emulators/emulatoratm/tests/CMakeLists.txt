# EmulatorAtm tests
#
# These drive the component through the whole per-step path -- import from
# x2a, pack, infer, unpack, export to a2x -- which is the part that no unit
# test of the inference layer can cover on its own.  The python backend is
# needed for the round trip, because the stub deliberately changes nothing.

if(EMULATOR_ENABLE_PYTHON)
  add_executable(test_atm_coupling test_atm_coupling.cpp)
  target_link_libraries(test_atm_coupling PRIVATE emulatoratm)
  target_include_directories(test_atm_coupling PRIVATE ${CATCH2_INCLUDE_DIR})
  target_compile_definitions(test_atm_coupling PRIVATE
    EMULATOR_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../common/tests/fixtures")
  add_test(NAME atm_coupling_tests COMMAND test_atm_coupling)
endif()
