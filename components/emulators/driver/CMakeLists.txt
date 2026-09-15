add_library(emulator_driver src/emulator_factory.cpp)
target_link_libraries(emulator_driver PUBLIC emulatoratm)
target_compile_features(emulator_driver PUBLIC cxx_std_20)

if(STANDALONE_MODE AND EMULATOR_ENABLE_NETCDF)
  add_executable(emulatoratm_driver src/atmosphere_main.cpp)
  target_link_libraries(emulatoratm_driver PRIVATE emulatoratm)
endif()
