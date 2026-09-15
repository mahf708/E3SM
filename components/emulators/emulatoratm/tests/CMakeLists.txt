# EmulatorAtm tests


add_executable(test_ace_channels test_ace_channels.cpp)
target_link_libraries(test_ace_channels PRIVATE emulatoratm Catch2)
add_test(NAME ace_channels_tests COMMAND test_ace_channels)

add_executable(test_ace_surface test_ace_surface.cpp)
target_link_libraries(test_ace_surface PRIVATE emulatoratm Catch2)
add_test(NAME ace_surface_tests COMMAND test_ace_surface)

add_executable(test_ace_surface_inputs test_ace_surface_inputs.cpp)
target_link_libraries(test_ace_surface_inputs PRIVATE emulatoratm Catch2)
add_test(NAME ace_surface_inputs_tests COMMAND test_ace_surface_inputs)

add_executable(test_insolation test_insolation.cpp)
target_link_libraries(test_insolation PRIVATE emulatoratm Catch2)
add_test(NAME insolation_tests COMMAND test_insolation)

# The real ACE2 checkpoint, on a GPU, from its real initial condition; skips (with a warning) if files, libtorch, netCDF or CUDA are missing.
add_executable(test_ace_real_model test_ace_real_model.cpp)
target_link_libraries(test_ace_real_model PRIVATE emulatoratm Catch2)
emulator_add_mpi_test(ace_real_model_tests test_ace_real_model)

# The ACE2 spec with the real checkpoint: a coupled day and a mid-interval restart that must match bit for bit; skips if files, netCDF, libtorch or a GPU are missing.
add_executable(test_ace_atmosphere_real test_ace_atmosphere_real.cpp)
target_link_libraries(test_ace_atmosphere_real PRIVATE emulatoratm Catch2)
emulator_add_mpi_test(ace_atmosphere_real_tests test_ace_atmosphere_real)

# The ACE2 spec as a component with the real checkpoint, through MCT-laid-out buffers.
add_executable(test_emulator_atm_real test_emulator_atm_real.cpp)
target_link_libraries(test_emulator_atm_real PRIVATE emulatoratm Catch2)
emulator_add_mpi_test(emulator_atm_real_tests test_emulator_atm_real)

