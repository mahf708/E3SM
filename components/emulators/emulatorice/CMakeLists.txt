cmake_minimum_required(VERSION 3.16)

# The sea ice that goes with an emulated ocean: the ocean's own ice fraction,
# dressed as a sea ice component for the coupler.
add_library(emulatorice
  src/sea_ice_surface.cpp
  src/ice.cpp
  src/ice_factory.cpp
)
target_include_directories(emulatorice PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(emulatorice PUBLIC emulator_common)
target_compile_features(emulatorice PUBLIC cxx_std_20)

if(BUILD_EMULATOR_TESTS)
  add_subdirectory(tests)
endif()
