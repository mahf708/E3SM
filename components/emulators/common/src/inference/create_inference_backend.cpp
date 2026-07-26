/**
 * @file create_inference_backend.cpp
 * @brief Convenience entry points for creating inference backends.
 */

#include "create_inference_backend.hpp"

namespace emulator {
namespace inference {

std::shared_ptr<InferenceBackend>
create_backend(const InferenceConfig &config) {
  return BackendRegistry::instance().create(config.backend, config);
}

std::shared_ptr<InferenceBackend> create_backend(const std::string &key,
                                                 const InferenceConfig &config) {
  return BackendRegistry::instance().create(key, config);
}

std::shared_ptr<InferenceBackend> create_backend(BackendType type,
                                                 const InferenceConfig &config) {
  // backend_type_name() maps unrecognized enum values to "stub", so a caller
  // built against an older enum still gets a working backend.
  return BackendRegistry::instance().create(backend_type_name(type), config);
}

std::shared_ptr<InferenceBackend>
create_and_init_backend(const InferenceConfig &config) {
  auto backend = create_backend(config);
  backend->initialize();
  return backend;
}

} // namespace inference
} // namespace emulator
