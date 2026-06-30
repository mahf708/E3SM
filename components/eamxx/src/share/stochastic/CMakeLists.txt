add_library(eamxx_stochastic
  spectral_pattern_generator.cpp
)

target_link_libraries(eamxx_stochastic PUBLIC
  eamxx_field
  eamxx_grid
  eamxx_core)

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()
