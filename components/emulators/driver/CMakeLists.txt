cmake_minimum_required(VERSION 3.16)

set(EMULATOR_DRIVER_SOURCES
  src/emulator_factory.cpp
)

add_library(emulator_driver ${EMULATOR_DRIVER_SOURCES})

# Dispatches emulator_create(kind) to each component library's creator.
target_link_libraries(emulator_driver PUBLIC
  emulator_common
  emulatoratm
  emulatorocn
  emulatorice
)

target_compile_features(emulator_driver PUBLIC cxx_std_20)

if(STANDALONE_MODE)
  add_subdirectory(tests)
endif()
