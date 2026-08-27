##################################
#       EAMxx I/O library        #
##################################

# This library allows for a simple(r) interaction between EAMxx
# data structures (such as grids, fields, remappers,...) and
# the scorpio interface library

add_library(eamxx_io
  eamxx_output_manager.cpp
  scorpio_output.cpp
  eamxx_io_utils.cpp
  eamxx_dexpr_diags.cpp
)

target_link_libraries(eamxx_io PUBLIC
  eamxx_data_managers
  eamxx_scorpio_interface
  eamxx_diagnostics)

# PRIVATE: eamxx_dexpr_diags.hpp exposes no dexpr types, so nothing that links
# eamxx_io needs dexpr's headers -- or its C++20 requirement, which would
# otherwise propagate through INTERFACE_COMPILE_FEATURES to the whole model.
target_link_libraries(eamxx_io PRIVATE dexpr)

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()
