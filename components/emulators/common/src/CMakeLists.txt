# Common source library
cmake_minimum_required(VERSION 3.16)

set(EMULATOR_COMMON_CXX_SOURCES
  emulator.cpp
  emulator_c_api.cpp
  component_settings.cpp
  config/yaml_config.cpp
  fields/field_list.cpp
  fields/field_set.cpp
  fields/mask_set.cpp
  fields/channel_layout.cpp
  fields/channel_layout_yaml.cpp
  fields/coupler_binding.cpp
  grid/horizontal_grid.cpp
  grid/scrip_reader.cpp
  grid/global_gather.cpp
  grid/grid_field_reader.cpp
  coupling/calendar.cpp
  physics/insolation.cpp
  model/model_fields.cpp
  model/emulated_model.cpp
  model/common_operators.cpp
  coupling/restart_store.cpp
  coupling/long_step_clock.cpp
  coupling/interval_state.cpp
  coupling/network_stepper.cpp
  coupling/exchange.cpp
  inference/tensor.cpp
  inference/fpe_guard.cpp
  inference/inference_config.cpp
  inference/inference_context.cpp
  inference/inference_backend.cpp
  inference/stub_inference_backend.cpp
  inference/create_inference_backend.cpp
)

if(EMULATOR_ENABLE_PYTHON)
  list(APPEND EMULATOR_COMMON_CXX_SOURCES
    inference/python_interpreter.cpp
    inference/python_inference_backend.cpp
  )
endif()

if(EMULATOR_ENABLE_LIBTORCH)
  list(APPEND EMULATOR_COMMON_CXX_SOURCES
    inference/libtorch_inference_backend.cpp
  )
endif()

set(EMULATOR_COMMON_F90_SOURCES
  emulator_f2c_api.F90
  emulator_f_api.F90
  emulator_handle.F90
)

# What every component's MCT cap does; it needs the driver's modules.
if(NOT STANDALONE_MODE)
  list(APPEND EMULATOR_COMMON_F90_SOURCES emulator_mct_cap.F90)
endif()

add_library(emulator_common
  ${EMULATOR_COMMON_CXX_SOURCES}
  ${EMULATOR_COMMON_F90_SOURCES}
)

if(NOT STANDALONE_MODE)
  target_link_libraries(emulator_common PUBLIC csm_share MPI::MPI_Fortran)
endif()

target_compile_features(emulator_common PUBLIC cxx_std_20)

# Place generated Fortran .mod files in a predictable directory so that
# downstream Fortran targets (e.g. the driver test) can find them.
set_target_properties(emulator_common PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)

target_include_directories(emulator_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/inference
  ${CMAKE_CURRENT_SOURCE_DIR}/fields
  ${CMAKE_CURRENT_SOURCE_DIR}/grid
  ${CMAKE_CURRENT_SOURCE_DIR}/coupling
  ${CMAKE_CURRENT_SOURCE_DIR}/config
  ${CMAKE_CURRENT_SOURCE_DIR}/model
)
target_link_libraries(emulator_common PUBLIC yaml-cpp::yaml-cpp)
target_compile_definitions(emulator_common PUBLIC
  EMULATOR_SPEC_DIR="${EMULATOR_SPEC_DIR}")

target_include_directories(emulator_common PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/modules
)

if(EMULATOR_HAVE_MPI)
  target_compile_definitions(emulator_common PUBLIC EMULATOR_HAVE_MPI)
  target_link_libraries(emulator_common PUBLIC MPI::MPI_CXX)
endif()

if(EMULATOR_ENABLE_PYTHON)
  target_compile_definitions(emulator_common PUBLIC EMULATOR_ENABLE_PYTHON)
  target_link_libraries(emulator_common PUBLIC Python3::Python)
  # Where the emulator python package lives, so a run does not have to set
  # PYTHONPATH.  Anything the user names in `inference.python_path` still wins.
  target_compile_definitions(emulator_common PRIVATE
    EMULATOR_PYTHON_PACKAGE_DIR="${EMULATOR_PYTHON_PACKAGE_DIR}")
endif()

if(EMULATOR_ENABLE_LIBTORCH)
  target_compile_definitions(emulator_common PUBLIC EMULATOR_ENABLE_LIBTORCH)
  # PUBLIC because TORCH_LIBRARIES carries the ABI flag libtorch was built
  # with (-D_GLIBCXX_USE_CXX11_ABI) on its interface, and every translation
  # unit that ends up in the same binary has to agree with it.
  target_link_libraries(emulator_common PUBLIC "${TORCH_LIBRARIES}")
endif()

if(EMULATOR_ENABLE_NETCDF)
  target_compile_definitions(emulator_common PUBLIC EMULATOR_HAVE_NETCDF)
  target_link_libraries(emulator_common PUBLIC emulator_netcdf)
endif()
