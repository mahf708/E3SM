cmake_minimum_required(VERSION 3.16)
project(emulators LANGUAGES CXX)

# Path to externals (relative to this file)
get_filename_component(E3SM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(EXTERNALS_ROOT "${E3SM_ROOT}/externals")

option(BUILD_EMULATOR_TESTS "Build emulator unit tests" OFF)

# When built standalone (i.e. this is the top-level project), default to
# standalone mode so we don't try to compile Fortran sources (the project
# only declares CXX).  When included from the full E3SM build Fortran
# is already enabled by the parent project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(STANDALONE_MODE ON CACHE BOOL "Standalone build (CXX only, no Fortran)" FORCE)
endif()

# yaml-cpp, for emulator specs and component input files, at the commit E3SM
# pins for ekat (externals/ekat/cmake/tpls/EkatBuildYamlCpp.cmake).  In a case
# that also builds EAMxx, ekat has already made the target.  Offline, point
# FETCHCONTENT_SOURCE_DIR_YAML-CPP at a checkout of that commit.
if(NOT TARGET yaml-cpp::yaml-cpp AND NOT TARGET yaml-cpp)
  find_package(yaml-cpp QUIET CONFIG)
endif()
if(NOT TARGET yaml-cpp::yaml-cpp AND NOT TARGET yaml-cpp)
  include(FetchContent)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/e3sm-project/yaml-cpp.git
    GIT_TAG        95088a0a2b6f2dec0b3e6e59020cdcc0d4f3c658)
  FetchContent_MakeAvailable(yaml-cpp)
endif()
if(NOT TARGET yaml-cpp::yaml-cpp)
  add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
endif()

# The specs that ship with the emulators (specs/*.yaml).
set(EMULATOR_SPEC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/specs")

# The component requires MPI for its domain and global inference.
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
# The grid, gather and network-stepping layers are MPI programs: a global
# network runs on one rank for all of them.
if(NOT EMULATOR_HAVE_MPI)
  message(FATAL_ERROR "The emulator framework needs MPI (C++); none was found "
    "or EMULATOR_ENABLE_MPI is OFF.")
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

# The LibTorch backend is the other opt-in: it runs a TorchScript module
# directly, which is what the traced ACE / Samudra checkpoints are.  Point
# CMAKE_PREFIX_PATH (or Torch_DIR) at a libtorch install to enable it.
option(EMULATOR_ENABLE_LIBTORCH
  "Build the LibTorch inference backend (needs a libtorch install)" OFF)

if(EMULATOR_ENABLE_LIBTORCH)
  find_package(Torch REQUIRED)
endif()

# ------------------------------------------------------------------------------
# netCDF
# ------------------------------------------------------------------------------

# The netCDF readers (SCRIP grids now, initial conditions later) are opt-in for
# the same reason the ML backends are: the unit-test loop has to need nothing
# but a compiler.  With this OFF the readers are simply not compiled, and a
# component asked to read a file says the build has no reader.
#
# Serial netCDF-C is all that is wanted here -- small global fields, read on
# each rank during initialization.  Parallel, decomposed I/O is SCORPIO's job.
# Off by default standalone; on by default in the integrated build, where E3SM
# has netCDF as a hard dependency and a real case cannot get a grid without it.
if (STANDALONE_MODE)
  set(_emulator_netcdf_default OFF)
else()
  set(_emulator_netcdf_default ON)
endif()

option(EMULATOR_ENABLE_NETCDF
  "Build the netCDF initial-condition reader (needs a serial netCDF-C install)"
  ${_emulator_netcdf_default})

if(EMULATOR_ENABLE_NETCDF)
  # Three ways to be told where netCDF is, in decreasing order of how much the
  # install tells us about itself:
  #
  #   1. a CMake package config (netCDF-C >= 4.7 installs netCDFConfig.cmake)
  #   2. nc-config on PATH or under NETCDF_ROOT / $NETCDF_DIR
  #   3. a plain header + library search
  #
  # A Cray machine lands on (2) or (3): /opt/cray/pe/netcdf/<ver>/<compiler>
  # ships no CMake config, and its nc-config is a wrapper that refuses to run
  # unless NETCDF_DIR is set -- so we set it for the probe.  Point the build at
  # one with either
  #
  #   -DNETCDF_ROOT=/opt/cray/pe/netcdf/4.9.2.3/gnu/12.3
  #   -DCMAKE_PREFIX_PATH=...   (for an install that has a CMake config)
  set(NETCDF_ROOT "" CACHE PATH
    "Root of a serial netCDF-C install, if it is not findable automatically")

  set(EMULATOR_NETCDF_INCLUDE_DIRS "")
  set(EMULATOR_NETCDF_LIBRARIES "")
  set(EMULATOR_NETCDF_HOW "")

  # In the integrated build E3SM has already found netCDF and exported
  # NETCDF_PATH from the machine config.  Prefer that over probing: it is the
  # netCDF the rest of the executable is linked against, and a second one
  # found independently could be a different library version.
  if (NOT STANDALONE_MODE AND NOT NETCDF_ROOT)
    if (DEFINED ENV{NETCDF_PATH})
      set(NETCDF_ROOT "$ENV{NETCDF_PATH}" CACHE PATH "" FORCE)
    elseif (DEFINED ENV{NETCDF_DIR})
      set(NETCDF_ROOT "$ENV{NETCDF_DIR}" CACHE PATH "" FORCE)
    endif()
    if (NETCDF_ROOT)
      message(STATUS "Emulator netCDF: using E3SM's ${NETCDF_ROOT}")
    endif()
  endif()

  # So find_package(netCDF) takes NETCDF_ROOT as a search prefix rather than
  # warning that it is ignoring it.
  if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
  endif()

  find_package(netCDF QUIET CONFIG)
  if(netCDF_FOUND)
    if(TARGET netCDF::netcdf)
      set(EMULATOR_NETCDF_LIBRARIES netCDF::netcdf)
    else()
      set(EMULATOR_NETCDF_INCLUDE_DIRS "${netCDF_INCLUDE_DIR}")
      set(EMULATOR_NETCDF_LIBRARIES "${netCDF_LIBRARIES}")
    endif()
    set(EMULATOR_NETCDF_HOW "netCDFConfig.cmake (${netCDF_VERSION})")
  endif()

  if(NOT EMULATOR_NETCDF_LIBRARIES)
    find_program(EMULATOR_NC_CONFIG nc-config
      HINTS "${NETCDF_ROOT}/bin" "$ENV{NETCDF_DIR}/bin" "$ENV{NETCDF_ROOT}/bin")
    if(EMULATOR_NC_CONFIG)
      get_filename_component(_nc_bin "${EMULATOR_NC_CONFIG}" DIRECTORY)
      get_filename_component(_nc_prefix "${_nc_bin}" DIRECTORY)
      # NETCDF_DIR is what the Cray wrapper insists on; harmless elsewhere.
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "NETCDF_DIR=${_nc_prefix}"
                ${EMULATOR_NC_CONFIG} --includedir
        OUTPUT_VARIABLE _nc_incdir OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _nc_inc_failed ERROR_QUIET)
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "NETCDF_DIR=${_nc_prefix}"
                ${EMULATOR_NC_CONFIG} --libdir
        OUTPUT_VARIABLE _nc_libdir OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _nc_lib_failed ERROR_QUIET)
      if(NOT _nc_inc_failed AND NOT _nc_lib_failed
         AND EXISTS "${_nc_incdir}/netcdf.h")
        find_library(EMULATOR_NETCDF_LIBRARY netcdf
          HINTS "${_nc_libdir}" NO_DEFAULT_PATH)
        if(EMULATOR_NETCDF_LIBRARY)
          set(EMULATOR_NETCDF_INCLUDE_DIRS "${_nc_incdir}")
          set(EMULATOR_NETCDF_LIBRARIES "${EMULATOR_NETCDF_LIBRARY}")
          set(EMULATOR_NETCDF_HOW "${EMULATOR_NC_CONFIG}")
        endif()
      endif()
    endif()
  endif()

  if(NOT EMULATOR_NETCDF_LIBRARIES)
    find_path(EMULATOR_NETCDF_INCLUDE_DIR netcdf.h
      HINTS "${NETCDF_ROOT}/include" "$ENV{NETCDF_DIR}/include"
            "$ENV{NETCDF_ROOT}/include")
    find_library(EMULATOR_NETCDF_LIBRARY netcdf
      HINTS "${NETCDF_ROOT}/lib" "${NETCDF_ROOT}/lib64"
            "$ENV{NETCDF_DIR}/lib" "$ENV{NETCDF_ROOT}/lib")
    if(EMULATOR_NETCDF_INCLUDE_DIR AND EMULATOR_NETCDF_LIBRARY)
      set(EMULATOR_NETCDF_INCLUDE_DIRS "${EMULATOR_NETCDF_INCLUDE_DIR}")
      set(EMULATOR_NETCDF_LIBRARIES "${EMULATOR_NETCDF_LIBRARY}")
      set(EMULATOR_NETCDF_HOW "${EMULATOR_NETCDF_LIBRARY}")
    endif()
  endif()

  if(NOT EMULATOR_NETCDF_LIBRARIES)
    message(FATAL_ERROR
      "EMULATOR_ENABLE_NETCDF=ON but no serial netCDF-C was found.  Pass "
      "-DNETCDF_ROOT=/path/to/netcdf (a prefix with include/netcdf.h and "
      "lib/libnetcdf), or -DCMAKE_PREFIX_PATH=... for an install that ships "
      "netCDFConfig.cmake.  On Perlmutter: "
      "-DNETCDF_ROOT=/opt/cray/pe/netcdf/4.9.2.3/gnu/12.3")
  endif()

  # One interface target so the components do not each repeat the search.
  add_library(emulator_netcdf INTERFACE)
  if(EMULATOR_NETCDF_INCLUDE_DIRS)
    target_include_directories(emulator_netcdf SYSTEM INTERFACE
      ${EMULATOR_NETCDF_INCLUDE_DIRS})
  endif()
  target_link_libraries(emulator_netcdf INTERFACE ${EMULATOR_NETCDF_LIBRARIES})

  message(STATUS "Emulator netCDF: ${EMULATOR_NETCDF_HOW}")
endif()

message(STATUS "Emulator inference: MPI=${EMULATOR_HAVE_MPI} "
  "python=${EMULATOR_ENABLE_PYTHON} libtorch=${EMULATOR_ENABLE_LIBTORCH}")

# The python package's own unit tests need an interpreter with numpy and
# nothing else, whether or not the embedded backend was built.
if(BUILD_EMULATOR_TESTS)
  # MPI tests run on one rank by default, so they run wherever ctest does.
  # Decompositions are only really exercised on several ranks, which needs a
  # launcher that works where ctest runs (inside an allocation on a cluster).
  set(EMULATOR_TEST_MPI_RANKS 0 CACHE STRING
    "If greater than 1, also run MPI tests on this many ranks via MPIEXEC")
  function(emulator_add_mpi_test name target)
    add_test(NAME ${name} COMMAND ${target})
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT "MPICH_GPU_SUPPORT_ENABLED=0")
    if(EMULATOR_TEST_MPI_RANKS GREATER 1)
      add_test(NAME ${name}_np${EMULATOR_TEST_MPI_RANKS}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
                ${EMULATOR_TEST_MPI_RANKS} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(${name}_np${EMULATOR_TEST_MPI_RANKS} PROPERTIES
        ENVIRONMENT "MPICH_GPU_SUPPORT_ENABLED=0")
    endif()
  endfunction()

  if(NOT Python3_EXECUTABLE)
    find_package(Python3 QUIET COMPONENTS Interpreter)
  endif()
  if(Python3_EXECUTABLE)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
      RESULT_VARIABLE EMULATOR_NUMPY_MISSING OUTPUT_QUIET ERROR_QUIET)
    if(EMULATOR_NUMPY_MISSING)
      message(STATUS "Emulator inference: no numpy in ${Python3_EXECUTABLE}, "
        "skipping the python package tests")
    endif()
  endif()
endif()


if(BUILD_EMULATOR_TESTS)
  # Catch2, as the v2 single header the tests include.  ekat used to vendor
  # this as a submodule and now pulls it with FetchContent, so the old path
  # is empty on a fresh clone; prefer a local copy wherever one exists and
  # otherwise fetch the one header we actually use.  Set CATCH2_INCLUDE_DIR
  # to point at a directory containing catch2/catch.hpp on a machine with no
  # network access from the build node.
  if(NOT CATCH2_INCLUDE_DIR)
    if(EXISTS "${EXTERNALS_ROOT}/ekat/extern/Catch2/single_include/catch2/catch.hpp")
      set(CATCH2_INCLUDE_DIR
        "${EXTERNALS_ROOT}/ekat/extern/Catch2/single_include")
    else()
      set(CATCH2_INCLUDE_DIR "${CMAKE_BINARY_DIR}/catch2")
      if(NOT EXISTS "${CATCH2_INCLUDE_DIR}/catch2/catch.hpp")
        message(STATUS "Emulator tests: downloading Catch2 v2.13.10")
        file(DOWNLOAD
          "https://raw.githubusercontent.com/catchorg/Catch2/v2.13.10/single_include/catch2/catch.hpp"
          "${CATCH2_INCLUDE_DIR}/catch2/catch.hpp"
          EXPECTED_HASH
            SHA256=3725c0f0a75f376a5005dde31ead0feb8f7da7507644c201b814443de8355170
          TLS_VERIFY ON
          STATUS EMULATOR_CATCH2_DOWNLOAD)
        list(GET EMULATOR_CATCH2_DOWNLOAD 0 EMULATOR_CATCH2_FAILED)
        if(EMULATOR_CATCH2_FAILED)
          list(GET EMULATOR_CATCH2_DOWNLOAD 1 EMULATOR_CATCH2_ERROR)
          message(FATAL_ERROR "Could not download Catch2: "
            "${EMULATOR_CATCH2_ERROR}.  Point CATCH2_INCLUDE_DIR at a "
            "directory containing catch2/catch.hpp instead.")
        endif()
      endif()
    endif()
  endif()

  if(NOT EXISTS "${CATCH2_INCLUDE_DIR}/catch2/catch.hpp")
    message(FATAL_ERROR
      "No catch2/catch.hpp under CATCH2_INCLUDE_DIR (${CATCH2_INCLUDE_DIR}).")
  endif()
  # One target for the tests to link, whichever way the header was found.
  add_library(Catch2 INTERFACE)
  target_include_directories(Catch2 INTERFACE ${CATCH2_INCLUDE_DIR})
  # Helpers every test shares: stand-in coupler buffers, input files, MPI.
  target_include_directories(Catch2 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common/tests/support)

  # Enable testing at the top level
  enable_testing()
endif()

# In the integrated build each component's MCT cap is compiled only when the
# case uses that component: MPAS-Ocean's cap, say, defines the same
# ocn_comp_mct module.  csm_share and the driver's seq_* modules come through
# emulator_common.
if(NOT STANDALONE_MODE)
  enable_language(Fortran)
  find_package(MPI REQUIRED COMPONENTS Fortran)
  find_package(CsmShare REQUIRED)
endif()
function(emulator_add_mct_cap target comp_name source)
  if(STANDALONE_MODE OR NOT COMP_NAMES MATCHES "${comp_name}")
    return()
  endif()
  target_sources(${target} PRIVATE ${source})
  set_target_properties(${target} PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
  target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/modules)
  target_link_libraries(${target} PUBLIC MPI::MPI_Fortran)
endfunction()

# Add subdirectories
add_subdirectory(common)

# A built case reads the specs it was built with (spec_path in
# cime_config/emulator_settings.py), so copy them on every build that touches
# the emulators.  CIME runs buildlib_cmake only at the first CMake configure;
# a copy made there went stale on an incremental rebuild.
if(NOT STANDALONE_MODE AND DEFINED EXEROOT)
  file(GLOB EMULATOR_SPECS CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/specs/*.yaml)
  set(EMULATOR_SPECS_BUILT ${EXEROOT}/emulators/specs)
  add_custom_target(emulator_specs
    COMMAND ${CMAKE_COMMAND} -E make_directory ${EMULATOR_SPECS_BUILT}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EMULATOR_SPECS}
            ${EMULATOR_SPECS_BUILT}
    COMMENT "Emulator specs -> ${EMULATOR_SPECS_BUILT}")
  add_dependencies(emulator_common emulator_specs)
endif()
add_subdirectory(emulatoratm)
add_subdirectory(driver)
