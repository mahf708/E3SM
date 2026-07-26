# FindLibTorch.cmake
#
# Locate LibTorch (the C++ distribution of PyTorch) for the TorchScript
# inference backend.
#
# There are two ways to find LibTorch, selected by LIBTORCH_USE_CMAKE_PACKAGE:
#
#  ON (default) — use PyTorch's own CMake package, `find_package(Torch)`.
#     This is the supported path: it sets the include directories, the C++ ABI
#     flag and the CUDA-related libraries the way the LibTorch build expects.
#     Point at it with -DTORCH_ROOT=<prefix>, -DCMAKE_PREFIX_PATH=<prefix>, or
#         python -c "import torch; print(torch.utils.cmake_prefix_path)"
#
#  OFF — search for the headers and libraries directly.  Use this when
#     TorchConfig.cmake cannot be satisfied, most commonly a CUDA-enabled
#     PyTorch wheel on a host without a CUDA toolkit, where Caffe2Config.cmake
#     aborts with "Your installed Caffe2 version uses CUDA but I cannot find
#     the CUDA libraries".  Only c10 and torch_cpu are required, which is what
#     CPU TorchScript inference needs; set LIBTORCH_LINK_META_LIB=ON to also
#     link the `torch` meta library (needed for CUDA dispatch, and it drags the
#     CUDA runtime dependencies in with it).
#
#     The two paths are not auto-detected in sequence on purpose: a failing
#     TorchConfig.cmake raises a CMake FATAL_ERROR, which even find_package's
#     QUIET cannot suppress, so trying it first would abort configuration
#     before any fallback could run.
#
# Result variables:
#   LibTorch_FOUND
#   LibTorch_INCLUDE_DIRS
#   LibTorch_LIBRARIES
#
# Imported target:
#   LibTorch::LibTorch

option(LIBTORCH_USE_CMAKE_PACKAGE
  "Find LibTorch through PyTorch's own TorchConfig.cmake (see FindLibTorch.cmake)" ON)
option(LIBTORCH_LINK_META_LIB
  "Also link the `torch` meta library when searching for LibTorch directly" OFF)

set(_libtorch_hints
  ${TORCH_ROOT}
  ${LIBTORCH_ROOT}
  ${LibTorch_ROOT}
  $ENV{TORCH_ROOT}
  $ENV{LIBTORCH_ROOT}
)

# --- PyTorch's own CMake package -------------------------------------------
if(LIBTORCH_USE_CMAKE_PACKAGE)
  if(NOT TARGET torch)
    set(_libtorch_saved_prefix ${CMAKE_PREFIX_PATH})
    foreach(hint ${_libtorch_hints})
      if(hint)
        list(APPEND CMAKE_PREFIX_PATH ${hint} ${hint}/share/cmake)
      endif()
    endforeach()
    message(STATUS "Looking for LibTorch via PyTorch's CMake package "
                   "(pass -DLIBTORCH_USE_CMAKE_PACKAGE=OFF to search for the "
                   "libraries directly instead)")
    find_package(Torch REQUIRED)
    set(CMAKE_PREFIX_PATH ${_libtorch_saved_prefix})
    unset(_libtorch_saved_prefix)
  endif()

  set(LibTorch_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}")
  set(LibTorch_LIBRARIES "${TORCH_LIBRARIES}")
  set(LibTorch_FOUND TRUE)

  if(NOT TARGET LibTorch::LibTorch)
    add_library(LibTorch::LibTorch INTERFACE IMPORTED)
    set_target_properties(LibTorch::LibTorch PROPERTIES
      INTERFACE_LINK_LIBRARIES "${TORCH_LIBRARIES}")
    if(TORCH_INCLUDE_DIRS)
      set_target_properties(LibTorch::LibTorch PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}")
    endif()
    if(TORCH_CXX_FLAGS)
      separate_arguments(_libtorch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
      set_target_properties(LibTorch::LibTorch PROPERTIES
        INTERFACE_COMPILE_OPTIONS "${_libtorch_flags}")
      unset(_libtorch_flags)
    endif()
  endif()
  message(STATUS "Found LibTorch via PyTorch's CMake package")
  unset(_libtorch_hints)
  return()
endif()

# --- Direct search ---------------------------------------------------------
find_path(LibTorch_INCLUDE_DIR
  NAMES torch/script.h
  HINTS ${_libtorch_hints}
  PATH_SUFFIXES include
  DOC "Directory containing torch/script.h")

find_path(LibTorch_API_INCLUDE_DIR
  NAMES torch/torch.h
  HINTS ${_libtorch_hints} ${LibTorch_INCLUDE_DIR}
  PATH_SUFFIXES include/torch/csrc/api/include torch/csrc/api/include
  DOC "Directory containing the torch C++ API headers")

find_library(LibTorch_C10_LIBRARY
  NAMES c10
  HINTS ${_libtorch_hints}
  PATH_SUFFIXES lib lib64)

find_library(LibTorch_CPU_LIBRARY
  NAMES torch_cpu
  HINTS ${_libtorch_hints}
  PATH_SUFFIXES lib lib64)

find_library(LibTorch_META_LIBRARY
  NAMES torch
  HINTS ${_libtorch_hints}
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibTorch
  REQUIRED_VARS LibTorch_INCLUDE_DIR LibTorch_API_INCLUDE_DIR
                LibTorch_C10_LIBRARY LibTorch_CPU_LIBRARY
  FAIL_MESSAGE "Could not find LibTorch. Pass -DTORCH_ROOT=<prefix> pointing \
at an unpacked libtorch distribution or a pip-installed torch package (the \
directory containing include/ and lib/).")

if(LibTorch_FOUND)
  set(LibTorch_INCLUDE_DIRS ${LibTorch_INCLUDE_DIR} ${LibTorch_API_INCLUDE_DIR})
  set(LibTorch_LIBRARIES ${LibTorch_CPU_LIBRARY} ${LibTorch_C10_LIBRARY})
  if(LIBTORCH_LINK_META_LIB AND LibTorch_META_LIBRARY)
    list(INSERT LibTorch_LIBRARIES 0 ${LibTorch_META_LIBRARY})
  endif()

  if(NOT TARGET LibTorch::LibTorch)
    add_library(LibTorch::LibTorch INTERFACE IMPORTED)
    set_target_properties(LibTorch::LibTorch PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${LibTorch_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${LibTorch_LIBRARIES}")
  endif()
  message(STATUS "Found LibTorch by direct search: ${LibTorch_LIBRARIES}")
endif()

mark_as_advanced(LibTorch_INCLUDE_DIR LibTorch_API_INCLUDE_DIR
                 LibTorch_C10_LIBRARY LibTorch_CPU_LIBRARY
                 LibTorch_META_LIBRARY)
unset(_libtorch_hints)
