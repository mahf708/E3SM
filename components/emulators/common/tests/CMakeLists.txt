# Common tests
# Test for emulator_registry
add_executable(test_emulator_registry test_emulator_registry.cpp)
target_link_libraries(test_emulator_registry PRIVATE emulator_common Catch2)
add_test(NAME emulator_registry_tests COMMAND test_emulator_registry)

# Test for emulator
add_executable(test_emulator test_emulator.cpp)
target_link_libraries(test_emulator PRIVATE emulator_common Catch2)
add_test(NAME emulator_tests COMMAND test_emulator)

# Test for InferenceConfig and utilities
add_executable(test_inference_config test_inference_config.cpp)
target_link_libraries(test_inference_config PRIVATE emulator_common Catch2)
add_test(NAME inference_config_tests COMMAND test_inference_config)

# Test for Tensor / TensorMap
add_executable(test_inference_tensor test_inference_tensor.cpp)
target_link_libraries(test_inference_tensor PRIVATE emulator_common Catch2)
add_test(NAME inference_tensor_tests COMMAND test_inference_tensor)

# Test for StubBackend
add_executable(test_inference_stub_backend test_inference_stub_backend.cpp)
target_link_libraries(test_inference_stub_backend PRIVATE emulator_common Catch2)
add_test(NAME inference_stub_backend_tests COMMAND test_inference_stub_backend)

# Test for InferenceContext
add_executable(test_inference_context test_inference_context.cpp)
target_link_libraries(test_inference_context PRIVATE emulator_common Catch2)
add_test(NAME inference_context_tests COMMAND test_inference_context)

# Specs and input files: YAML with errors that name the key
add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE emulator_common Catch2)
add_test(NAME config_tests COMMAND test_config)

# The spec-driven model and the common operators, with a toy network
add_executable(test_emulated_model test_emulated_model.cpp)
target_link_libraries(test_emulated_model PRIVATE emulator_common Catch2)
emulator_add_mpi_test(emulated_model_tests test_emulated_model)

# The generic component: input files, domains
add_executable(test_emulator_component test_emulator_component.cpp)
target_link_libraries(test_emulator_component PRIVATE emulator_common Catch2)
emulator_add_mpi_test(emulator_component_tests test_emulator_component)

# Test for the component field layer: FieldList, FieldSet, CouplerBinding
add_executable(test_fields test_fields.cpp)
target_link_libraries(test_fields PRIVATE emulator_common Catch2)
add_test(NAME fields_tests COMMAND test_fields)

# Test for the grid, decomposition and coupler domain
add_executable(test_grid test_grid.cpp)
target_link_libraries(test_grid PRIVATE emulator_common Catch2)
add_test(NAME grid_tests COMMAND test_grid)

# Test for GlobalGather: one rank here, and EMULATOR_TEST_MPI_RANKS more
add_executable(test_global_gather test_global_gather.cpp)
target_link_libraries(test_global_gather PRIVATE emulator_common Catch2)
emulator_add_mpi_test(global_gather_tests test_global_gather)

# Test for the SCRIP reader: writes its own small files, and reads the real 180x360 Gaussian grid where that file exists
if(EMULATOR_ENABLE_NETCDF)
  add_executable(test_scrip_reader test_scrip_reader.cpp)
  target_link_libraries(test_scrip_reader PRIVATE emulator_common Catch2)
  add_test(NAME scrip_reader_tests COMMAND test_scrip_reader)
endif()

# Restart files: written on some ranks, read back on others
add_executable(test_restart_file test_restart_file.cpp)
target_link_libraries(test_restart_file PRIVATE emulator_common Catch2)
emulator_add_mpi_test(restart_file_tests test_restart_file)

# Test for the long-step primitives: clock, interval mean, brackets, restart
add_executable(test_long_step test_long_step.cpp)
target_link_libraries(test_long_step PRIVATE emulator_common Catch2)
add_test(NAME long_step_tests COMMAND test_long_step)

# Test for NetworkStepper: a toy network across ranks
add_executable(test_network_stepper test_network_stepper.cpp)
target_link_libraries(test_network_stepper PRIVATE emulator_common Catch2)
emulator_add_mpi_test(network_stepper_tests test_network_stepper)

# Test for FpeGuard, which every ML backend wraps its model calls in
add_executable(test_fpe_guard test_fpe_guard.cpp)
target_link_libraries(test_fpe_guard PRIVATE emulator_common Catch2)
add_test(NAME fpe_guard_tests COMMAND test_fpe_guard)

# Test for the embedded-Python backend
if(EMULATOR_ENABLE_PYTHON)
  add_executable(test_inference_python_backend test_inference_python_backend.cpp)
  target_link_libraries(test_inference_python_backend PRIVATE emulator_common Catch2)
  target_compile_definitions(test_inference_python_backend PRIVATE
    EMULATOR_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
  add_test(NAME inference_python_backend_tests COMMAND test_inference_python_backend)
endif()

# The python package is pure numpy, so it is tested directly rather than through the C++ bridge, with no MPI, launcher or torch needed.
if(Python3_EXECUTABLE AND NOT EMULATOR_NUMPY_MISSING)
  add_test(NAME inference_python_package_tests
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_e3sm_emulator.py)
  set_tests_properties(inference_python_package_tests PROPERTIES
    ENVIRONMENT
      "PYTHONPATH=${EMULATOR_PYTHON_PACKAGE_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/python")
endif()

if(Python3_EXECUTABLE AND NOT EMULATOR_NUMPY_MISSING)
  add_test(NAME inference_example_tests
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_example.py)
  set_tests_properties(inference_example_tests PROPERTIES
    ENVIRONMENT
      "PYTHONPATH=${EMULATOR_PYTHON_PACKAGE_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/python")
endif()

# Test for the LibTorch backend: TorchScript fixtures are written by a Python interpreter with torch (EMULATOR_TORCH_PYTHON), skipped if none is found, and its torch version should match this libtorch.
if(EMULATOR_ENABLE_LIBTORCH)
  set(EMULATOR_TORCH_PYTHON "" CACHE FILEPATH
    "Python interpreter with torch, used to write the TorchScript test fixtures")

  set(EMULATOR_TORCH_PYTHON_FOUND "")
  foreach(_torch_py_candidate
      "${EMULATOR_TORCH_PYTHON}"
      "${Python3_EXECUTABLE}"
      "python3")
    if(_torch_py_candidate AND NOT EMULATOR_TORCH_PYTHON_FOUND)
      execute_process(COMMAND ${_torch_py_candidate} -c
          "import torch; print(torch.__version__)"
        RESULT_VARIABLE _torch_py_failed
        OUTPUT_VARIABLE _torch_py_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(NOT _torch_py_failed)
        set(EMULATOR_TORCH_PYTHON_FOUND "${_torch_py_candidate}")
        set(EMULATOR_TORCH_PYTHON_VERSION "${_torch_py_version}")
      endif()
    endif()
  endforeach()

  if(NOT EMULATOR_TORCH_PYTHON_FOUND)
    message(STATUS "Emulator tests: no python with torch found, skipping the "
      "libtorch backend test (it needs torch.jit.save to write its "
      "fixtures). Set -DEMULATOR_TORCH_PYTHON=/path/to/python to enable it.")
  else()
    message(STATUS "Emulator tests: writing the TorchScript fixtures with "
      "${EMULATOR_TORCH_PYTHON_FOUND} (torch ${EMULATOR_TORCH_PYTHON_VERSION}); "
      "libtorch ${Torch_VERSION} has to be able to load what it saves")

    set(EMULATOR_TORCH_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/torch_fixtures)
    set(EMULATOR_TORCH_FIXTURES
      ${EMULATOR_TORCH_FIXTURE_DIR}/libtorch_affine.pt
      ${EMULATOR_TORCH_FIXTURE_DIR}/libtorch_tuple.pt
      ${EMULATOR_TORCH_FIXTURE_DIR}/libtorch_channels.pt
      ${EMULATOR_TORCH_FIXTURE_DIR}/libtorch_noise.pt)

    add_custom_command(
      OUTPUT ${EMULATOR_TORCH_FIXTURES}
      COMMAND ${EMULATOR_TORCH_PYTHON_FOUND}
              ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_libtorch_fixture.py
              ${EMULATOR_TORCH_FIXTURE_DIR}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_libtorch_fixture.py
      COMMENT "Writing the TorchScript test fixtures"
      VERBATIM)
    add_custom_target(libtorch_test_fixtures DEPENDS ${EMULATOR_TORCH_FIXTURES})

    add_executable(test_inference_libtorch_backend
      test_inference_libtorch_backend.cpp)
    add_dependencies(test_inference_libtorch_backend libtorch_test_fixtures)
    target_link_libraries(test_inference_libtorch_backend PRIVATE
      emulator_common Catch2)
    target_compile_definitions(test_inference_libtorch_backend PRIVATE
      EMULATOR_TEST_TORCH_FIXTURE_DIR="${EMULATOR_TORCH_FIXTURE_DIR}")
    add_test(NAME inference_libtorch_backend_tests
      COMMAND test_inference_libtorch_backend)
  endif()
endif()
