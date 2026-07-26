/**
 * @file inference_backend_registry.cpp
 * @brief Backend registry plus registration of the compiled-in backends.
 */

#include "inference_backend_registry.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

#include "stub_inference_backend.hpp"

#ifdef EMULATOR_ENABLE_PYTHON
#include "python_inference_backend.hpp"
#endif
#ifdef EMULATOR_ENABLE_TORCH
#include "torch_inference_backend.hpp"
#endif
#ifdef EMULATOR_ENABLE_ONNXRUNTIME
#include "onnx_inference_backend.hpp"
#endif

namespace emulator {
namespace inference {

namespace {

/// Register every backend compiled into this library.
void register_builtins(BackendRegistry &registry) {
  registry.register_backend("stub", [](const InferenceConfig &config) {
    return std::make_shared<StubBackend>(config);
  });

#ifdef EMULATOR_ENABLE_PYTHON
  registry.register_backend("python", [](const InferenceConfig &config) {
    return std::make_shared<PythonBackend>(config);
  });
#endif
#ifdef EMULATOR_ENABLE_TORCH
  registry.register_backend("torch", [](const InferenceConfig &config) {
    return std::make_shared<TorchBackend>(config);
  });
  registry.register_backend("libtorch", [](const InferenceConfig &config) {
    return std::make_shared<TorchBackend>(config);
  });
#endif
#ifdef EMULATOR_ENABLE_ONNXRUNTIME
  registry.register_backend("onnx", [](const InferenceConfig &config) {
    return std::make_shared<OnnxBackend>(config);
  });
  registry.register_backend("onnxruntime", [](const InferenceConfig &config) {
    return std::make_shared<OnnxBackend>(config);
  });
#endif
}

/// Which CMake option provides a backend that is not compiled in.
const char *build_option_for(const std::string &key) {
  if (key == "python") {
    return "EMULATOR_ENABLE_PYTHON";
  }
  if (key == "torch" || key == "libtorch") {
    return "EMULATOR_ENABLE_TORCH";
  }
  if (key == "onnx" || key == "onnxruntime") {
    return "EMULATOR_ENABLE_ONNXRUNTIME";
  }
  return nullptr;
}

} // namespace

BackendRegistry &BackendRegistry::instance() {
  static BackendRegistry registry;
  // Populate `registry` directly (not through instance()) to avoid recursing
  // into this function while the static is still being initialized.
  static const bool builtins_done = [](BackendRegistry &r) {
    register_builtins(r);
    return true;
  }(registry);
  (void)builtins_done;
  return registry;
}

std::string BackendRegistry::normalize(const std::string &key) {
  std::string out;
  out.reserve(key.size());
  for (char c : key) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      out.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return out;
}

void BackendRegistry::register_backend(const std::string &key, Factory factory,
                                       bool overwrite) {
  const std::string norm = normalize(key);
  EMULATOR_INFER_REQUIRE(!norm.empty(),
                         "Cannot register an inference backend with an empty "
                         "key.");
  EMULATOR_INFER_REQUIRE(static_cast<bool>(factory),
                         "Cannot register a null factory for inference "
                         "backend '"
                             << key << "'.");
  EMULATOR_INFER_REQUIRE(overwrite || m_factories.find(norm) ==
                                          m_factories.end(),
                         "Inference backend '"
                             << norm
                             << "' is already registered. Pass "
                                "overwrite=true to replace it.");
  m_factories[norm] = std::move(factory);
}

bool BackendRegistry::unregister_backend(const std::string &key) {
  return m_factories.erase(normalize(key)) > 0;
}

bool BackendRegistry::has(const std::string &key) const {
  return m_factories.find(normalize(key)) != m_factories.end();
}

std::vector<std::string> BackendRegistry::available() const {
  std::vector<std::string> keys;
  keys.reserve(m_factories.size());
  for (const auto &kv : m_factories) {
    keys.push_back(kv.first);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

std::string BackendRegistry::available_string() const {
  std::ostringstream oss;
  const auto keys = available();
  for (std::size_t i = 0; i < keys.size(); ++i) {
    oss << (i ? ", " : "") << keys[i];
  }
  const std::string s = oss.str();
  return s.empty() ? "<none>" : s;
}

std::shared_ptr<InferenceBackend>
BackendRegistry::create(const std::string &key,
                        const InferenceConfig &config) const {
  const std::string norm = normalize(key);
  const auto it = m_factories.find(norm);

  if (it == m_factories.end()) {
    const char *option = build_option_for(norm);
    std::ostringstream hint;
    if (option != nullptr) {
      hint << " That backend exists but was not compiled in; configure with -D"
           << option << "=ON.";
    }
    EMULATOR_INFER_REQUIRE(false, "Unknown inference backend '"
                                      << key << "'. Available: "
                                      << available_string() << "."
                                      << hint.str());
  }

  auto backend = it->second(config);
  EMULATOR_INFER_REQUIRE(backend != nullptr,
                         "Factory for inference backend '" << norm
                                                           << "' returned "
                                                              "null.");
  return backend;
}

} // namespace inference
} // namespace emulator
