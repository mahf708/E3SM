/**
 * @file create_inference_backend.hpp
 * @brief Convenience entry points for creating inference backends.
 *
 * These are thin wrappers around BackendRegistry.  Including this header
 * gives a caller everything needed to build and drive a backend (config,
 * tensors, registry).
 */

#ifndef E3SM_EMULATOR_CREATE_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_CREATE_INFERENCE_BACKEND_HPP

#include <memory>
#include <string>

#include "inference_backend.hpp"
#include "inference_backend_registry.hpp"
#include "inference_config.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Create a backend using the key in `config.backend`.
 * @throws InferenceError if that backend is not registered
 */
std::shared_ptr<InferenceBackend> create_backend(const InferenceConfig &config);

/**
 * @brief Create a backend by registry key ("stub", "python", "torch", "onnx").
 * @throws InferenceError if the backend is not registered
 */
std::shared_ptr<InferenceBackend> create_backend(const std::string &key,
                                                 const InferenceConfig &config);

/**
 * @brief Create a built-in backend by enum.
 *
 * Unknown enum values fall back to the dependency-free stub backend, so a
 * caller that has not been taught about a newer enumerator still runs.
 */
std::shared_ptr<InferenceBackend> create_backend(BackendType type,
                                                 const InferenceConfig &config);

/**
 * @brief Create a backend and initialize it in one step.
 * @throws InferenceError if creation or initialization fails
 */
std::shared_ptr<InferenceBackend>
create_and_init_backend(const InferenceConfig &config);

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_CREATE_INFERENCE_BACKEND_HPP
