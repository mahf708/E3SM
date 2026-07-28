cmake_minimum_required(VERSION 3.16)
project(emulators LANGUAGES CXX Fortran)

# Path to externals (relative to this file)
get_filename_component(E3SM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(EXTERNALS_ROOT "${E3SM_ROOT}/externals")

option(BUILD_EMULATOR_TESTS "Build emulator unit tests" OFF)

# MPI is what turns InferenceContext from a serial stub into a real
# description of the component's ranks.  Without it everything still builds
# and the unit tests still run; the context just reports rank 0 of 1.
set(EMULATOR_ENABLE_MPI AUTO CACHE STRING
  "Use MPI in the inference context: AUTO, ON or OFF")

if(EMULATOR_ENABLE_MPI STREQUAL "AUTO")
  find_package(MPI QUIET COMPONENTS CXX)
  set(EMULATOR_HAVE_MPI ${MPI_CXX_FOUND})
elseif(EMULATOR_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  set(EMULATOR_HAVE_MPI TRUE)
else()
  set(EMULATOR_HAVE_MPI FALSE)
endif()
# The embedded-Python backend is opt-in: emulator_common builds, and its
# tests pass, with no ML dependency installed at all.
option(EMULATOR_ENABLE_PYTHON
  "Build the embedded-Python inference backend (needs the Python dev headers, \
plus numpy at run time)" OFF)

if(EMULATOR_ENABLE_PYTHON)
  # Development is the embedding half: headers and libpython.
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
endif()

# The python emulators that ship with E3SM.  Baked into the backend as a
# sys.path entry so a run does not have to set PYTHONPATH.
set(EMULATOR_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common/python")

message(STATUS "Emulator inference: MPI=${EMULATOR_HAVE_MPI} "
  "python=${EMULATOR_ENABLE_PYTHON}")

# When built standalone (i.e. this is the top-level project), default to
# standalone mode so we don't try to compile Fortran sources (the project
# only declares CXX).  When included from the full E3SM build Fortran
# is already enabled by the parent project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(STANDALONE_MODE ON CACHE BOOL "Standalone build (CXX only, no Fortran)" FORCE)
endif()

if(BUILD_EMULATOR_TESTS)
  # Catch2 from ekat
  set(CATCH2_INCLUDE_DIR "${EXTERNALS_ROOT}/ekat/extern/Catch2/single_include")

  # Verify Catch2 exists
  if(NOT EXISTS "${CATCH2_INCLUDE_DIR}/catch2/catch.hpp")
    message(FATAL_ERROR "Catch2 not found at ${CATCH2_INCLUDE_DIR}. "
      "Make sure externals/ekat is available.")
  endif()

  # Enable testing at the top level
  enable_testing()
endif()

# Add subdirectories
add_subdirectory(common)
add_subdirectory(emulatoratm)
add_subdirectory(driver)
