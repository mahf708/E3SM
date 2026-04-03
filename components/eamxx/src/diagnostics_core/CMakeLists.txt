# ====================================================================
# kokkos-diag-utils: Standalone diagnostic expression engine
#
# This is a header-only library with no EAMxx/EKAT dependencies.
# It requires only Kokkos and standard C++17.
#
# Consumers link against this target:
#   target_link_libraries(my_target kokkos_diag_utils)
# ====================================================================

# Header-only interface library
add_library(kokkos_diag_utils INTERFACE)
target_include_directories(kokkos_diag_utils INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

# Kokkos is required for the RPN evaluator and DiagnosticBase
if (TARGET Kokkos::kokkos)
  target_link_libraries(kokkos_diag_utils INTERFACE Kokkos::kokkos)
elseif (TARGET Kokkos::kokkoscore)
  target_link_libraries(kokkos_diag_utils INTERFACE Kokkos::kokkoscore)
endif()

# Tests
if (NOT SCREAM_LIB_ONLY AND NOT SCREAM_ONLY_GENERATE_BASELINES)
  add_subdirectory(tests)
endif()
