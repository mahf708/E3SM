# Sea ice tests

add_executable(test_sea_ice_surface test_sea_ice_surface.cpp)
target_link_libraries(test_sea_ice_surface PRIVATE emulatorice Catch2)
add_test(NAME sea_ice_surface_tests COMMAND test_sea_ice_surface)
