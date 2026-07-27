# Common tests

# Helper: one Catch2 executable per test file, linked against emulator_common.
function(add_emulator_test test_name source)
  add_executable(${test_name} ${source})
  target_link_libraries(${test_name} PRIVATE emulator_common)
  target_include_directories(${test_name} PRIVATE ${CATCH2_INCLUDE_DIR})
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Test for emulator_registry
add_emulator_test(test_emulator_registry test_emulator_registry.cpp)

# Test for emulator
add_emulator_test(test_emulator test_emulator.cpp)

# ---------------------------------------------------------------------------
# Inference layer (always available: no ML dependency required)
# ---------------------------------------------------------------------------

# Test for the Tensor/TensorMap/TensorSpec data container
add_emulator_test(test_inference_tensor test_inference_tensor.cpp)

# Test for InferenceConfig and utilities
add_emulator_test(test_inference_config test_inference_config.cpp)

# Test for the backend registry and the InferenceBackend contract
add_emulator_test(test_inference_registry test_inference_registry.cpp)

# Test for StubBackend
add_emulator_test(test_inference_stub_backend test_inference_stub_backend.cpp)

# Test for the execution-policy layer (serial: local_replica needs no MPI)
add_emulator_test(test_inference_executor test_inference_executor.cpp)

# ---------------------------------------------------------------------------
# Parallel test
#
# The one test here that runs on several ranks: it checks that a component
# communicator reaches the inference layer intact and that a local replica
# behaves on ragged local column counts.
# ---------------------------------------------------------------------------
if(EMULATOR_INFERENCE_HAS_MPI AND MPIEXEC_EXECUTABLE)
  add_executable(test_inference_context_mpi test_inference_context_mpi.cpp)
  target_link_libraries(test_inference_context_mpi PRIVATE emulator_common
                                                           MPI::MPI_CXX)
  add_test(NAME test_inference_context_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_inference_context_mpi>
            ${MPIEXEC_POSTFLAGS})
elseif(EMULATOR_INFERENCE_HAS_MPI)
  message(STATUS "Emulator tests: skipping the parallel context test "
                 "(no mpiexec found)")
endif()

# ---------------------------------------------------------------------------
# Optional backends
#
# Each of these needs a model artifact.  The artifacts are generated at
# configure time by small scripts under fixtures/ so that no binary blobs are
# committed; if generation fails (e.g. no `torch` in the environment) the
# corresponding test is skipped with a message instead of failing the build.
# ---------------------------------------------------------------------------

set(EMULATOR_TEST_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
file(MAKE_DIRECTORY ${EMULATOR_TEST_FIXTURE_DIR})

# Generate a model artifact with a python script.  Sets <out_var> to the
# artifact path on success and to an empty string on failure.
function(emulator_generate_fixture out_var script artifact)
  if(NOT Python3_EXECUTABLE)
    find_package(Python3 QUIET COMPONENTS Interpreter)
  endif()
  set(${out_var} "" PARENT_SCOPE)
  if(NOT Python3_EXECUTABLE)
    message(STATUS "Emulator tests: no python interpreter; cannot generate ${artifact}")
    return()
  endif()

  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${script}
            ${EMULATOR_TEST_FIXTURE_DIR}/${artifact}
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(rc EQUAL 0 AND EXISTS ${EMULATOR_TEST_FIXTURE_DIR}/${artifact})
    set(${out_var} ${EMULATOR_TEST_FIXTURE_DIR}/${artifact} PARENT_SCOPE)
  else()
    message(STATUS "Emulator tests: could not generate ${artifact} "
                   "(${script}): ${err}")
  endif()
endfunction()

if(EMULATOR_ENABLE_PYTHON)
  add_emulator_test(test_inference_python_backend
    test_inference_python_backend.cpp)
  # The python-side emulators used by the test live in fixtures/.
  target_compile_definitions(test_inference_python_backend PRIVATE
    EMULATOR_TEST_PYTHON_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
endif()

if(EMULATOR_ENABLE_TORCH)
  emulator_generate_fixture(TORCH_TEST_MODEL make_torch_model.py
    test_affine.pt)
  if(TORCH_TEST_MODEL)
    add_emulator_test(test_inference_torch_backend
      test_inference_torch_backend.cpp)
    target_compile_definitions(test_inference_torch_backend PRIVATE
      EMULATOR_TEST_TORCH_MODEL="${TORCH_TEST_MODEL}")
  else()
    message(STATUS "Emulator tests: skipping the torch backend test "
                   "(no TorchScript fixture; needs the `torch` python package)")
  endif()
endif()

if(EMULATOR_ENABLE_ONNXRUNTIME)
  emulator_generate_fixture(ONNX_TEST_MODEL make_onnx_model.py test_affine.onnx)
  if(ONNX_TEST_MODEL)
    add_emulator_test(test_inference_onnx_backend
      test_inference_onnx_backend.cpp)
    target_compile_definitions(test_inference_onnx_backend PRIVATE
      EMULATOR_TEST_ONNX_MODEL="${ONNX_TEST_MODEL}")
  else()
    message(STATUS "Emulator tests: skipping the onnx backend test "
                   "(no ONNX fixture; needs the `onnx` python package)")
  endif()
endif()
