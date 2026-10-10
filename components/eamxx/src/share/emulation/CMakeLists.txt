# Emulators of physics process rates, run through the inference backends of
# components/emulators. The stub backend is always built; the python and
# libtorch ones are enabled with EMULATOR_ENABLE_PYTHON and EMULATOR_ENABLE_LIBTORCH
# (the latter needs CMAKE_PREFIX_PATH or Torch_DIR to point at a LibTorch install).
if (NOT TARGET emulator_inference)
  add_subdirectory(${SCREAM_BASE_DIR}/../emulators/common/src/inference
                   ${CMAKE_CURRENT_BINARY_DIR}/emulator_inference)
endif()

add_library(eamxx_process_emulator eamxx_process_emulator.cpp)
target_link_libraries(eamxx_process_emulator PUBLIC eamxx_core ekat::Pack emulator_inference)
target_compile_definitions(eamxx_process_emulator PUBLIC EAMXX_HAS_PROCESS_EMULATORS)

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()
