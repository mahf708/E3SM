/**
 * @file yaml_config.hpp
 * @brief Reading emulator specs and input files, with errors that say where.
 */

#ifndef E3SM_EMULATOR_CONFIG_YAML_CONFIG_HPP
#define E3SM_EMULATOR_CONFIG_YAML_CONFIG_HPP

#include <yaml-cpp/yaml.h>

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace emulator {
namespace config {

/**
 * @brief A YAML node and the path that reached it, e.g.
 *        "samudra-e3smv3-ocean.yaml: network.inputs".
 *
 * Every accessor names that path when a key is missing, has the wrong type,
 * or is not one the reader knows: a spec is written by hand, and a typo must
 * stop the run with the key's name rather than be silently ignored.
 */
class Section {
public:
  Section(YAML::Node node, std::string where)
      : m_node(std::move(node)), m_where(std::move(where)) {}

  /// Load a whole file.  @throws std::runtime_error naming the file
  static Section load_file(const std::string &path);
  /**
   * @brief Load a spec, following `extends: <file>`: the named file, relative
   *        to this one, is loaded first and this file's top-level keys
   *        replace its.  A variant spec then states only what differs.
   */
  static Section load_spec(const std::string &path);
  /// Parse a string, for tests.  `name` stands in for the file name.
  static Section load_string(const std::string &text, const std::string &name);

  const std::string &where() const { return m_where; }
  const YAML::Node &node() const { return m_node; }
  bool has(const std::string &key) const;

  /// @throws std::invalid_argument if missing or not a map
  Section section(const std::string &key) const;
  /// A map, or an empty section if the key is absent.
  Section optional_section(const std::string &key) const;

  std::string string(const std::string &key) const;
  std::string string_or(const std::string &key, const std::string &fallback) const;
  double number(const std::string &key) const;
  double number_or(const std::string &key, double fallback) const;
  long long integer(const std::string &key) const;
  long long integer_or(const std::string &key, long long fallback) const;
  bool boolean_or(const std::string &key, bool fallback) const;

  /**
   * @brief A list of names, with ranges expanded: "T_{0..7}" is T_0 .. T_7.
   * Absent is an empty list.
   */
  std::vector<std::string> names(const std::string &key) const;
  /// The list's entries, each a section, for lists of maps.
  std::vector<Section> list(const std::string &key) const;
  /// The map's keys, in file order.
  std::vector<std::string> keys() const;

  /// @throws std::invalid_argument naming every key not in `allowed`
  void only(std::initializer_list<const char *> allowed) const;

private:
  YAML::Node child(const std::string &key) const;
  YAML::Node m_node;
  std::string m_where;
};

/// Expand "stem{a..b}suffix" into the names from a to b; other names as is.
std::vector<std::string> expand_range(const std::string &name);

} // namespace config
} // namespace emulator

#endif // E3SM_EMULATOR_CONFIG_YAML_CONFIG_HPP
