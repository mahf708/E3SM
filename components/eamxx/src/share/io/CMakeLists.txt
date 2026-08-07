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

# Translation from a parsed DSL expression to a diagnostic name + params.
# Kept as its own target, depending only on the parser, because every
# diagnostic parameter is a string and so none of this needs EAMxx types.
# That is what lets the translation be unit tested without a model build.
add_library(eamxx_diag_dsl
  eamxx_diag_dsl.cpp
  eamxx_diag_names.cpp
)
target_link_libraries(eamxx_diag_dsl PUBLIC e3sm_diags_parser)
target_include_directories(eamxx_diag_dsl PUBLIC ${SCREAM_SRC_DIR})

add_library(eamxx_io
  eamxx_output_manager.cpp
  scorpio_output.cpp
  eamxx_io_utils.cpp
)

target_link_libraries(eamxx_io PUBLIC
  eamxx_data_managers
  eamxx_scorpio_interface
  eamxx_diagnostics
  eamxx_diag_dsl)

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()
