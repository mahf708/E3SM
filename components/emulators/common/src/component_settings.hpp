/**
 * @file component_settings.hpp
 * @brief A component's `key: value` input file (atm_in, ocn_in, ice_in).
 */

#ifndef E3SM_EMULATOR_COMPONENT_SETTINGS_HPP
#define E3SM_EMULATOR_COMPONENT_SETTINGS_HPP

#include <map>
#include <string>

namespace emulator {

/**
 * @brief `key: value` lines, trimmed; blank lines and `#` comments skipped.
 *
 * A file that cannot be opened reads as no settings, which a component
 * treats as "no model configured".
 */
class ComponentSettings {
public:
  ComponentSettings() = default;
  static ComponentSettings read(const std::string &path);

  std::string get(const std::string &key, const std::string &fallback) const;
  bool has(const std::string &key) const { return m_values.count(key) > 0; }
  void set(const std::string &key, const std::string &value) {
    m_values[key] = value;
  }

private:
  std::map<std::string, std::string> m_values;
};

} // namespace emulator

#endif // E3SM_EMULATOR_COMPONENT_SETTINGS_HPP
