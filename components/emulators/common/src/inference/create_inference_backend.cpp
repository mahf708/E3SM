/**
 * @file create_inference_backend.cpp
 * @brief Factory for creating inference backends.
 *
 * A switch rather than a registry: there are two backends, and a registry
 * that has to be populated explicitly (a static library may drop an object
 * file whose symbols nothing references, silently un-registering a backend)
 * is more machinery than two names justify.
 */

#include "create_inference_backend.hpp"

#include "inference_error.hpp"
#include "stub_inference_backend.hpp"

#ifdef EMULATOR_ENABLE_PYTHON
#include "python_inference_backend.hpp"
#endif

namespace emulator {
namespace inference {

std::vector<std::string> available_backends() {
  std::vector<std::string> names{"stub"};
#ifdef EMULATOR_ENABLE_PYTHON
  names.push_back("python");
#endif
  return names;
}

namespace {

std::string joined_names() {
  std::string s;
  for (const auto &n : available_backends()) {
    if (!s.empty()) {
      s += ", ";
    }
    s += n;
  }
  return s;
}

} // namespace

std::shared_ptr<InferenceBackend>
create_backend(const InferenceConfig &config, const InferenceContext &context) {
  std::shared_ptr<InferenceBackend> backend;

  if (config.backend.empty() || config.backend == "stub" ||
      config.backend == "none") {
    backend = std::make_shared<StubBackend>(config, context);
  } else if (config.backend == "python") {
#ifdef EMULATOR_ENABLE_PYTHON
    backend = std::make_shared<PythonBackend>(config, context);
#else
    EMULATOR_INFER_REQUIRE(
        false, "The 'python' inference backend was not built. Reconfigure "
               "with -DEMULATOR_ENABLE_PYTHON=ON (it needs the Python "
               "development headers, plus numpy at run time).");
#endif
  } else {
    EMULATOR_INFER_REQUIRE(false, "Unknown inference backend '"
                                      << config.backend
                                      << "'. This build has: " << joined_names()
                                      << ".");
  }

  backend->initialize();
  return backend;
}

std::shared_ptr<InferenceBackend> create_backend(BackendType type,
                                                 const InferenceConfig &config) {
  InferenceConfig copy = config;
  switch (type) {
  case BackendType::PYTHON:
    copy.backend = "python";
    break;
  case BackendType::STUB:
  default:
    copy.backend = "stub";
    break;
  }
  return create_backend(copy, InferenceContext());
}

} // namespace inference
} // namespace emulator
