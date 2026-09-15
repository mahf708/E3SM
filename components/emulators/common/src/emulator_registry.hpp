/**
 * @file emulator_registry.hpp
 * @brief Singleton registry for managing emulator instances.
 */

#ifndef E3SM_EMULATOR_REGISTRY_HPP
#define E3SM_EMULATOR_REGISTRY_HPP

#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace emulator {

/**
 * @brief Singleton registry for managing emulator instances.
 *
 * Type-safe storage and retrieval of emulator instances by name, allowing
 * multiple instances of the same type with different names.
 */
class EmulatorRegistry {
public:
  /**
   * @brief Get the singleton instance.
   * @return Reference to the global EmulatorRegistry
   */
  static EmulatorRegistry &instance() {
    static EmulatorRegistry r;
    return r;
  }

  // Prevent copying and moving
  EmulatorRegistry(const EmulatorRegistry &) = delete;
  EmulatorRegistry &operator=(const EmulatorRegistry &) = delete;
  EmulatorRegistry(EmulatorRegistry &&) = delete;
  EmulatorRegistry &operator=(EmulatorRegistry &&) = delete;

  /**
   * @brief Create and register a new emulator instance under a unique name.
   *
   * @param name Unique name for this instance.
   * @throws std::runtime_error if an instance with the same name already exists.
   */
  template <typename T, typename... Args>
  T &create(const std::string &name, Args &&...args) {
    if (m_objects.find(name) != m_objects.end()) {
      throw std::runtime_error(
          "Error! Object with name '" + name +
          "' was already created in the emulator registry.\n");
    }

    auto ptr = std::make_shared<T>(std::forward<Args>(args)...);
    m_objects[name] = std::any(ptr);

    return *ptr;
  }

  /**
   * @brief Get a const reference to an existing emulator by name.
   *
   * @throws std::runtime_error if no instance with the given name exists
   * @throws std::bad_any_cast if the type doesn't match
   */
  template <typename T> const T &get(const std::string &name) const {
    auto it = m_objects.find(name);
    if (it == m_objects.end()) {
      throw std::runtime_error("Error! Object with name '" + name +
                               "' not found in the emulator registry.\n");
    }
    return *std::any_cast<const std::shared_ptr<T> &>(it->second);
  }

  /**
   * @brief Get a mutable reference to an existing emulator by name.
   *
   * @throws std::runtime_error if no instance with the given name exists
   * @throws std::bad_any_cast if the type doesn't match
   */
  template <typename T> T &get_mut(const std::string &name) {
    auto it = m_objects.find(name);
    if (it == m_objects.end()) {
      throw std::runtime_error("Error! Object with name '" + name +
                               "' not found in the emulator registry.\n");
    }
    return *std::any_cast<const std::shared_ptr<T> &>(it->second);
  }

  /// True if an instance with this name exists in the registry.
  bool has(const std::string &name) const {
    return m_objects.find(name) != m_objects.end();
  }

  /**
   * @brief Remove a named object from the registry.
   *
   * Erases the entry; if no other shared_ptr references remain, the object
   * is destroyed immediately.
   *
   * @return true if the entry was found and removed.
   */
  bool remove_by_name(const std::string &name) {
    return m_objects.erase(name) > 0;
  }

  /**
   * @brief Remove all objects from the registry.
   *
   * Should be called during shutdown to release resources.
   */
  void clean_up() { m_objects.clear(); }

private:
  EmulatorRegistry() = default;

  std::unordered_map<std::string, std::any> m_objects; ///< Name-indexed storage
};

/**
 * @brief Convenience function to clean up the global emulator registry.
 *
 * Equivalent to EmulatorRegistry::instance().clean_up().
 */
inline void cleanup_emulator_registry() {
  EmulatorRegistry::instance().clean_up();
}

} // namespace emulator

#endif // E3SM_EMULATOR_REGISTRY_HPP
