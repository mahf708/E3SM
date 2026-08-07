##################################
#       EAMxx I/O library        #
##################################

# This library allows for a simple(r) interaction between EAMxx
# data structures (such as grids, fields, remappers,...) and
# the scorpio interface library

# Vendored, dependency-free parser for the diagnostics DSL. Kept standalone:
# it links against nothing, and nothing in it may depend on EAMxx. See
# edp/README.md.
add_subdirectory(edp)

add_library(eamxx_io
  eamxx_output_manager.cpp
  scorpio_output.cpp
  eamxx_io_utils.cpp
)

target_link_libraries(eamxx_io PUBLIC
  eamxx_data_managers
  eamxx_scorpio_interface
  eamxx_diagnostics
  e3sm_diags_parser)

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()
