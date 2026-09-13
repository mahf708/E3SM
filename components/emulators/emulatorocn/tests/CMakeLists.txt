# Samudra ocean tests

add_executable(test_samudra_pieces test_samudra_pieces.cpp)
target_link_libraries(test_samudra_pieces PRIVATE emulatorocn Catch2)
add_test(NAME samudra_pieces_tests COMMAND test_samudra_pieces)

# The real Samudra checkpoint on a GPU.  Skips where the files, netCDF,
# libtorch or a GPU are missing.
add_executable(test_samudra_real test_samudra_real.cpp)
target_link_libraries(test_samudra_real PRIVATE emulatorocn Catch2)
emulator_add_mpi_test(samudra_real_tests test_samudra_real)

# SamudrACE coupled in one process through the exchange: the real
# atmosphere and ocean checkpoints on a GPU.  Skips where they are missing.
add_executable(test_samudrace_coupled_real test_samudrace_coupled_real.cpp)
target_link_libraries(test_samudrace_coupled_real PRIVATE emulatorocn emulatoratm Catch2)
emulator_add_mpi_test(samudrace_coupled_real_tests test_samudrace_coupled_real)

# The ocean specs against SamudraOcean, bit for bit, on the real checkpoint.
add_executable(test_ocean_spec_equivalence_real test_ocean_spec_equivalence_real.cpp)
target_link_libraries(test_ocean_spec_equivalence_real PRIVATE emulatorocn Catch2)
emulator_add_mpi_test(ocean_spec_equivalence_real_tests test_ocean_spec_equivalence_real)
