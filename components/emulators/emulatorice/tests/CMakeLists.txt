# Sea ice tests

add_executable(test_sea_ice_surface test_sea_ice_surface.cpp)
target_link_libraries(test_sea_ice_surface PRIVATE emulatorice Catch2)
add_test(NAME sea_ice_surface_tests COMMAND test_sea_ice_surface)

add_executable(test_emulator_ice test_emulator_ice.cpp)
target_link_libraries(test_emulator_ice PRIVATE emulatorice Catch2)
emulator_add_mpi_test(emulator_ice_tests test_emulator_ice)

# The real Samudra ocean component and its sea ice, through MCT-laid-out
# buffers.  Skips where the files, netCDF, libtorch or a GPU are missing.
add_executable(test_emulator_ocn_ice_real test_emulator_ocn_ice_real.cpp)
target_link_libraries(test_emulator_ocn_ice_real PRIVATE emulatorice emulatorocn Catch2)
emulator_add_mpi_test(emulator_ocn_ice_real_tests test_emulator_ocn_ice_real)

# SamudrACE as three components configured from their input files, sharing
# the exchange.  Skips where the files, netCDF, libtorch or a GPU are missing.
add_executable(test_samudrace_trio_real test_samudrace_trio_real.cpp)
target_link_libraries(test_samudrace_trio_real PRIVATE emulatorice emulatorocn emulatoratm Catch2)
emulator_add_mpi_test(samudrace_trio_real_tests test_samudrace_trio_real)
