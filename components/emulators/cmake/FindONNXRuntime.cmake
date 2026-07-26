# FindONNXRuntime.cmake
#
# Locate an ONNX Runtime C/C++ installation (the archives published at
# https://github.com/microsoft/onnxruntime/releases, a package manager
# install, or a site module).
#
# Search hints, in order of precedence:
#   -DONNXRUNTIME_ROOT=<prefix>      (CMake variable)
#   ONNXRUNTIME_ROOT / ORT_ROOT      (environment variables)
#   CMAKE_PREFIX_PATH / system paths
#
# Result variables:
#   ONNXRuntime_FOUND
#   ONNXRuntime_INCLUDE_DIRS
#   ONNXRuntime_LIBRARIES
#   ONNXRuntime_VERSION            (when discoverable from the headers)
#
# Imported target:
#   ONNXRuntime::ONNXRuntime

set(_ort_hints
  ${ONNXRUNTIME_ROOT}
  ${ONNXRuntime_ROOT}
  $ENV{ONNXRUNTIME_ROOT}
  $ENV{ONNXRuntime_ROOT}
  $ENV{ORT_ROOT}
)

find_path(ONNXRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${_ort_hints}
  PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
  DOC "Directory containing onnxruntime_cxx_api.h")

find_library(ONNXRuntime_LIBRARY
  NAMES onnxruntime
  HINTS ${_ort_hints}
  PATH_SUFFIXES lib lib64
  DOC "The onnxruntime shared library")

# The release headers carry the version in ORT_API_VERSION-adjacent macros;
# onnxruntime_c_api.h is the reliable place to look.
set(ONNXRuntime_VERSION "")
if(ONNXRuntime_INCLUDE_DIR AND EXISTS "${ONNXRuntime_INCLUDE_DIR}/onnxruntime_c_api.h")
  file(STRINGS "${ONNXRuntime_INCLUDE_DIR}/onnxruntime_c_api.h" _ort_version_line
    REGEX "^#define ORT_API_VERSION[ \t]+[0-9]+")
  if(_ort_version_line)
    string(REGEX MATCH "[0-9]+" ONNXRuntime_API_VERSION "${_ort_version_line}")
    set(ONNXRuntime_VERSION "api-${ONNXRuntime_API_VERSION}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
  REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
  VERSION_VAR ONNXRuntime_VERSION
  FAIL_MESSAGE "Could not find ONNX Runtime. Pass -DONNXRUNTIME_ROOT=<prefix> \
pointing at an unpacked onnxruntime release (the directory containing \
include/ and lib/), or set the ONNXRUNTIME_ROOT environment variable.")

if(ONNXRuntime_FOUND)
  set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
  set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})

  if(NOT TARGET ONNXRuntime::ONNXRuntime)
    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
      IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
unset(_ort_hints)
