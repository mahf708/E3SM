# Command line tools for the emulator common library

# Harness for driving any inference backend from a config file; see
# inference_demo.cpp for what it is good for.
add_executable(emulator_inference_demo inference_demo.cpp)
target_link_libraries(emulator_inference_demo PRIVATE emulator_common)
