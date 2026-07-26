# Common source library
cmake_minimum_required(VERSION 3.16)

set(EMULATOR_COMMON_CXX_SOURCES
  emulator.cpp
  emulator_c_api.cpp
  inference/tensor.cpp
  inference/inference_config.cpp
  inference/inference_backend.cpp
  inference/inference_backend_registry.cpp
  inference/create_inference_backend.cpp
  inference/stub_inference_backend.cpp
)

set(EMULATOR_COMMON_F90_SOURCES
  emulator_f2c_api.F90
  emulator_f_api.F90
  emulator_handle.F90
)

# Optional inference backends.  Each one is opt-in: emulator_common must keep
# building (and its tests passing) with no ML dependency installed at all.
if(EMULATOR_ENABLE_PYTHON)
  list(APPEND EMULATOR_COMMON_CXX_SOURCES
    inference/python_interpreter.cpp
    inference/python_inference_backend.cpp)
endif()

if(EMULATOR_ENABLE_TORCH)
  list(APPEND EMULATOR_COMMON_CXX_SOURCES
    inference/torch_inference_backend.cpp)
endif()

if(EMULATOR_ENABLE_ONNXRUNTIME)
  list(APPEND EMULATOR_COMMON_CXX_SOURCES
    inference/onnx_inference_backend.cpp)
endif()

add_library(emulator_common
  ${EMULATOR_COMMON_CXX_SOURCES}
  ${EMULATOR_COMMON_F90_SOURCES}
)

target_compile_features(emulator_common PUBLIC cxx_std_17)

# Place generated Fortran .mod files in a predictable directory so that
# downstream Fortran targets (e.g. the driver test) can find them.
set_target_properties(emulator_common PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)

target_include_directories(emulator_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/inference
)

target_include_directories(emulator_common PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/modules
)

# ---------------------------------------------------------------------------
# Optional backend dependencies
# ---------------------------------------------------------------------------

if(EMULATOR_ENABLE_PYTHON)
  # Development.Embed provides libpython, which embedding needs
  # (Development.Module alone is for *building* extension modules).
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed)
  target_link_libraries(emulator_common PUBLIC Python3::Python)
  target_compile_definitions(emulator_common PUBLIC EMULATOR_ENABLE_PYTHON)
  message(STATUS "Emulator inference: python backend ON (${Python3_EXECUTABLE})")
endif()

if(EMULATOR_ENABLE_TORCH)
  find_package(LibTorch REQUIRED)
  target_link_libraries(emulator_common PUBLIC LibTorch::LibTorch)
  target_compile_definitions(emulator_common PUBLIC EMULATOR_ENABLE_TORCH)
  message(STATUS "Emulator inference: torch backend ON")
endif()

if(EMULATOR_ENABLE_ONNXRUNTIME)
  find_package(ONNXRuntime REQUIRED)
  target_link_libraries(emulator_common PUBLIC ONNXRuntime::ONNXRuntime)
  target_compile_definitions(emulator_common PUBLIC EMULATOR_ENABLE_ONNXRUNTIME)
  message(STATUS "Emulator inference: onnxruntime backend ON")
endif()
