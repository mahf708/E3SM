cmake_minimum_required(VERSION 3.16)
project(emulators LANGUAGES CXX Fortran)

# Path to externals (relative to this file)
get_filename_component(E3SM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(EXTERNALS_ROOT "${E3SM_ROOT}/externals")

option(BUILD_EMULATOR_TESTS "Build emulator unit tests" OFF)

# Command line tools (currently the inference harness).  On by default in a
# test build, since they are useful for exactly the same kind of poking about.
option(BUILD_EMULATOR_TOOLS "Build emulator command line tools"
  ${BUILD_EMULATOR_TESTS})

# Inference backends.  All are optional and OFF by default: emulator_common
# must build and test with no ML runtime installed.
option(EMULATOR_ENABLE_PYTHON "Build the embedded-Python inference backend" OFF)
option(EMULATOR_ENABLE_TORCH "Build the LibTorch (TorchScript) inference backend" OFF)
option(EMULATOR_ENABLE_ONNXRUNTIME "Build the ONNX Runtime inference backend" OFF)

# Our Find modules for the optional runtimes (FindLibTorch, FindONNXRuntime).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# When built standalone (i.e. this is the top-level project), default to
# standalone mode so we don't try to compile Fortran sources (the project
# only declares CXX).  When included from the full E3SM build Fortran
# is already enabled by the parent project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(STANDALONE_MODE ON CACHE BOOL "Standalone build (CXX only, no Fortran)" FORCE)
endif()

if(BUILD_EMULATOR_TESTS)
  # Catch2 (v2, single header) from ekat by default.  Overridable with
  # -DCATCH2_INCLUDE_DIR=... for environments where Catch2 comes from
  # somewhere else (a module, a system install, a newer ekat layout).
  set(CATCH2_INCLUDE_DIR "${EXTERNALS_ROOT}/ekat/extern/Catch2/single_include"
    CACHE PATH "Directory containing catch2/catch.hpp")

  # Verify Catch2 exists
  if(NOT EXISTS "${CATCH2_INCLUDE_DIR}/catch2/catch.hpp")
    message(FATAL_ERROR "Catch2 not found at ${CATCH2_INCLUDE_DIR}. "
      "Make sure externals/ekat is available, or pass "
      "-DCATCH2_INCLUDE_DIR=<dir containing catch2/catch.hpp>.")
  endif()

  # Enable testing at the top level
  enable_testing()
endif()

# Add subdirectories
add_subdirectory(common)
add_subdirectory(emulatoratm)
add_subdirectory(driver)
