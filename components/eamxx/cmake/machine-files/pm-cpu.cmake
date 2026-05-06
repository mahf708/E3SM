include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)
common_setup()

set(CMAKE_CXX_FLAGS "-DTHRUST_IGNORE_CUB_VERSION_CHECK" CACHE STRING "" FORCE)

#message(STATUS "pm-cpu CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID} CMAKE_Fortran_COMPILER_VERSION=${CMAKE_Fortran_COMPILER_VERSION}")
if ("${PROJECT_NAME}" STREQUAL "E3SM")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
      set(CMAKE_Fortran_FLAGS "-fallow-argument-mismatch"  CACHE STRING "" FORCE) # only works with gnu v10 and above
    endif()
  endif()
else()
  set(CMAKE_Fortran_FLAGS "-fallow-argument-mismatch"  CACHE STRING "" FORCE) # only works with gnu v10 and above
endif()

# Pin the entire Python install (interpreter + headers + libs) to the
# NERSC python/3.13 conda env. We can't gate on EAMXX_ENABLE_PYTHON here
# because this machine file is processed before that option is declared
# in components/eamxx/CMakeLists.txt. Setting the cache vars when the
# bridge is off is harmless (nothing reads them).
#
# Why all four vars: pybind11's CMake config calls find_package(Python
# COMPONENTS Interpreter Development) internally. Setting only
# Python_EXECUTABLE is treated as a hint - CMake's FindPython may still
# walk PATH and pick a different install (on Perlmutter this lands on
# cray's /opt/cray/pe/python/3.11, producing a build that links cray's
# libpython3.11 even though Python_EXECUTABLE pointed at 3.13). The
# resulting binary then can't see pip-installed packages in the 3.13
# user site-packages at runtime. Forcing Python_ROOT_DIR plus
# Python_FIND_STRATEGY=LOCATION makes FindPython use exactly this
# install for headers and libs, matching the embed target to the
# interpreter we'll actually run.
set(_EAMXX_PY_PREFIX
    "/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python")
set(Python_EXECUTABLE "${_EAMXX_PY_PREFIX}/bin/python"
    CACHE FILEPATH "Python interpreter for EAMxx pybind11 bridge (>=3.9)" FORCE)
set(Python_ROOT_DIR "${_EAMXX_PY_PREFIX}"
    CACHE PATH "Root of the Python install used by EAMxx" FORCE)
set(Python_FIND_STRATEGY LOCATION
    CACHE STRING "Make FindPython prefer Python_ROOT_DIR over PATH" FORCE)
unset(_EAMXX_PY_PREFIX)
