#!/bin/bash
#
# Reusable argument parsing for test scripts
#
# Usage: source this file, then call parse_test_args "$@"
# After calling, the following variables will be set:
#   CLEAN_ONLY, BUILD_ONLY, VERBOSE, CMAKE_EXTRA_ARGS
#

# Default values
CLEAN_ONLY=false
BUILD_ONLY=false
VERBOSE=false
CMAKE_EXTRA_ARGS=()

parse_test_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean-only)
                CLEAN_ONLY=true
                shift
                ;;
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --with-python)
                CMAKE_EXTRA_ARGS+=("-DEMULATOR_ENABLE_PYTHON=ON")
                shift
                ;;
            --with-onnx)
                CMAKE_EXTRA_ARGS+=("-DEMULATOR_ENABLE_ONNXRUNTIME=ON")
                shift
                ;;
            --onnx-root)
                CMAKE_EXTRA_ARGS+=("-DEMULATOR_ENABLE_ONNXRUNTIME=ON" "-DONNXRUNTIME_ROOT=$2")
                shift 2
                ;;
            --with-torch)
                CMAKE_EXTRA_ARGS+=("-DEMULATOR_ENABLE_TORCH=ON")
                shift
                ;;
            --torch-root)
                CMAKE_EXTRA_ARGS+=("-DEMULATOR_ENABLE_TORCH=ON" "-DTORCH_ROOT=$2")
                shift 2
                ;;
            -D*)
                # Pass any other CMake definition straight through.
                CMAKE_EXTRA_ARGS+=("$1")
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --clean-only       Remove build directory and exit"
                echo "  --build-only       Build without running tests"
                echo "  --verbose          Show verbose test output"
                echo "  --with-python      Enable the embedded-python inference backend"
                echo "  --with-onnx        Enable the ONNX Runtime inference backend"
                echo "  --onnx-root DIR    Enable ONNX Runtime and use this install prefix"
                echo "  --with-torch       Enable the LibTorch inference backend"
                echo "  --torch-root DIR   Enable LibTorch and use this install prefix"
                echo "  -D<VAR>=<VALUE>    Pass a definition through to CMake"
                echo "  --help             Show this help message"
                echo ""
                echo "The inference backends are optional; with none of them enabled"
                echo "only the dependency-free stub backend is built and tested."
                exit 0
                ;;
            *)
                echo "ERROR: Unknown option: $1" >&2
                exit 1
                ;;
        esac
    done
}
