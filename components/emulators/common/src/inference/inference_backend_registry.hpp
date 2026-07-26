/**
 * @file inference_backend_registry.hpp
 * @brief String-keyed factory registry for inference backends.
 */

#ifndef E3SM_EMULATOR_INFERENCE_BACKEND_REGISTRY_HPP
#define E3SM_EMULATOR_INFERENCE_BACKEND_REGISTRY_HPP

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "inference_backend.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Registry mapping backend keys to factory functions.
 *
 * Built-in backends (those compiled into emulator_common, which depends on
 * which optional dependencies were enabled) are registered on first use.
 * Anything else — a site-specific runtime, a research prototype, a mock in a
 * test — can be added at run time:
 *
 * ```cpp
 * BackendRegistry::instance().register_backend(
 *     "my_runtime", [](const InferenceConfig& c) {
 *       return std::make_shared<MyBackend>(c);
 *     });
 * ```
 *
 * Registration is explicit rather than relying on static initializers in each
 * backend translation unit: emulator_common is a static library, and a linker
 * is free to drop an object file whose symbols are never referenced, which
 * would silently un-register a backend.
 */
class BackendRegistry {
public:
  /// @brief Creates a backend from a configuration.
  using Factory =
      std::function<std::shared_ptr<InferenceBackend>(const InferenceConfig &)>;

  /// @brief Global registry instance (built-ins already registered).
  static BackendRegistry &instance();

  BackendRegistry(const BackendRegistry &) = delete;
  BackendRegistry &operator=(const BackendRegistry &) = delete;

  /**
   * @brief Register a factory under `key` (compared case-insensitively).
   * @param overwrite Allow replacing an existing entry (e.g. to shadow a
   *                  built-in backend in a test)
   * @throws InferenceError if the key exists and overwrite is false
   */
  void register_backend(const std::string &key, Factory factory,
                        bool overwrite = false);

  /// @brief Remove a registration; returns true if one was present.
  bool unregister_backend(const std::string &key);

  /// @brief True if a factory is registered under `key`.
  bool has(const std::string &key) const;

  /// @brief Registered keys, sorted.
  std::vector<std::string> available() const;

  /// @brief Comma-separated available() for diagnostics.
  std::string available_string() const;

  /**
   * @brief Create a backend.
   * @throws InferenceError if the key is unknown (message lists the known
   *         keys and which build options add the missing ones)
   */
  std::shared_ptr<InferenceBackend> create(const std::string &key,
                                           const InferenceConfig &config) const;

private:
  BackendRegistry() = default;

  /// Lowercase, trimmed form of a key.
  static std::string normalize(const std::string &key);

  std::map<std::string, Factory> m_factories;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_BACKEND_REGISTRY_HPP
