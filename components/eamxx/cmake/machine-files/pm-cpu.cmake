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

##++ LL
## quartz-intel.cmake 
##set(PYTHON_EXECUTABLE "/usr/tce/packages/python/python-3.9.12/bin/python3" CACHE STRING "" FORCE)
##set(PYTHON_LIBRARIES "/usr/lib64/libpython3.9.so.1.0" CACHE STRING "" FORCE)
##option (SCREAM_ENABLE_ML_CORRECTION "Whether to enable ML correction parametrization" ON) 
##set(HDF5_DISABLE_VERSION_CHECK 1 CACHE STRING "" FORCE)
##execute_process(COMMAND source /usr/WS1/e3sm/python_venv/3.9.2/screamML/bin/activate)

if (EAMXX_ENABLE_PYTHON) 
    set(Python_EXECUTABLE "/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python" CACHE STRING "the CMake variable Python_EXECUTABLE must point to a python3 executable, with python version >= 3.9." FORCE)
  if (NOT Python_EXECUTABLE)
     message (FATAL_ERROR "You must set Python_EXECUTABLE to point to a valid python3 interpreter")
  endif()

  execute_process(
     COMMAND "${Python_EXECUTABLE}" -c "import sys; assert sys.version_info >= (3,9), 'Python version must be >= 3.9'"
     RESULT_VARIABLE python_version_check_result
     ERROR_VARIABLE python_version_check_error
)
  if (python_version_check_result)
     message(FATAL_ERROR "Python version check failed: ${python_version_check_error}")
  endif()

endif()
##-- LL
