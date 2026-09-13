cmake_minimum_required(VERSION 3.16)

# The emulated atmosphere: ACE on the framework, and its MCT cap.
add_library(emulatoratm
  src/ace_surface.cpp
  src/ace_surface_inputs.cpp
  src/ace_operators.cpp
  src/samudrace_surface.cpp
  src/atm_factory.cpp
)
target_include_directories(emulatoratm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(emulatoratm PUBLIC emulator_common)
target_compile_features(emulatoratm PUBLIC cxx_std_20)
emulator_add_mct_cap(emulatoratm emulatoratm src/atm_comp_mct.F90)

if(BUILD_EMULATOR_TESTS)
  add_subdirectory(tests)
endif()
