#==============================================================================
# Common Unit Tests
#==============================================================================

message(STATUS "  Configuring common unit tests...")

# Configuration parsing tests
EmulatorUnitTest(test_emulator_config
  SOURCES test_emulator_config.cpp
  LIBS emulator_common
  LABELS config
)

# Coupling fields tests
EmulatorUnitTest(test_coupling_fields
  SOURCES test_coupling_fields.cpp
  LIBS emulator_common
  LABELS coupling
)

# Inference backend tests
add_subdirectory(inference)
