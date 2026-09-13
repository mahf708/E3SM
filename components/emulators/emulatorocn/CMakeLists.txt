cmake_minimum_required(VERSION 3.16)

# The emulated ocean: Samudra on the framework, and its MCT cap.
add_library(emulatorocn
  src/ocean_forcing.cpp
  src/ocean_operators.cpp
  src/ocn_factory.cpp
)
target_include_directories(emulatorocn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(emulatorocn PUBLIC emulator_common)
target_compile_features(emulatorocn PUBLIC cxx_std_20)
emulator_add_mct_cap(emulatorocn emulatorocn src/ocn_comp_mct.F90)

if(BUILD_EMULATOR_TESTS)
  add_subdirectory(tests)
endif()
