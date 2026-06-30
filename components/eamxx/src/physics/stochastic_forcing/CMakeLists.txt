add_library(stochastic_forcing
  eamxx_sppt_process_interface.cpp
  eamxx_skebs_process_interface.cpp
)

target_compile_definitions(stochastic_forcing PUBLIC EAMXX_HAS_SPPT EAMXX_HAS_SKEBS)
target_link_libraries(stochastic_forcing eamxx_physics_share scream_share eamxx_stochastic)

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()

if (TARGET eamxx_physics)
  # Add this library to eamxx_physics
  target_link_libraries(eamxx_physics INTERFACE stochastic_forcing)
endif()
