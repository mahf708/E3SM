# Samudra ocean tests

add_executable(test_samudra_pieces test_samudra_pieces.cpp)
target_link_libraries(test_samudra_pieces PRIVATE emulatorocn Catch2)
add_test(NAME samudra_pieces_tests COMMAND test_samudra_pieces)

# The real Samudra checkpoint on a GPU.  Skips where the files, netCDF,
# libtorch or a GPU are missing.
add_executable(test_samudra_real test_samudra_real.cpp)
target_link_libraries(test_samudra_real PRIVATE emulatorocn Catch2)
emulator_add_mpi_test(samudra_real_tests test_samudra_real)
