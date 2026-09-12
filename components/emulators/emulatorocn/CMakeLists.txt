cmake_minimum_required(VERSION 3.16)

# The Samudra ocean's model logic: channels, forcing, stepping, exports.  No
# MCT cap yet; the component and its cap follow the atmosphere's.
add_library(emulatorocn
  src/samudra_channels.cpp
  src/ocean_forcing.cpp
  src/samudra_ocean.cpp
  src/ocn.cpp
)
target_include_directories(emulatorocn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(emulatorocn PUBLIC emulator_common)
target_compile_features(emulatorocn PUBLIC cxx_std_20)

if(BUILD_EMULATOR_TESTS)
  add_subdirectory(tests)
endif()
