# Common utilities for emulator components

add_subdirectory(src)
if(BUILD_EMULATOR_TESTS)
  add_subdirectory(tests)
endif()
if(BUILD_EMULATOR_TOOLS)
  add_subdirectory(tools)
endif()
