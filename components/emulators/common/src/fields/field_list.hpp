/**
 * @file field_list.hpp
 * @brief The ordered field names of one coupler attribute vector.
 */

#ifndef E3SM_EMULATOR_FIELDS_FIELD_LIST_HPP
#define E3SM_EMULATOR_FIELDS_FIELD_LIST_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace emulator {
namespace fields {

/**
 * @brief Field names in attribute-vector order, e.g. from `seq_flds_x2a_fields`.
 *
 * The position of a name is its row in the attribute vector, so a FieldList
 * is what turns "Sa_tbot" into the index the coupler's buffer is laid out by.
 * Names are unique and non-empty; a list that breaks either rule is refused
 * when it is built, not when the missing field is first looked up.
 */
class FieldList {
public:
  FieldList() = default;

  /// @throws std::invalid_argument on an empty or duplicated name.
  explicit FieldList(std::vector<std::string> names);

  /**
   * @brief Parse the coupler's colon-separated form, `"Sa_z:Sa_u:Sa_v"`.
   *
   * Whitespace around each name is dropped, and so is anything from the
   * first NUL: a list arriving from Fortran is often a fixed-length buffer.
   * An empty string is an empty list.
   */
  static FieldList parse(std::string_view colon_separated);

  std::size_t size() const { return m_names.size(); }
  bool empty() const { return m_names.empty(); }
  const std::string &name(std::size_t index) const { return m_names.at(index); }
  const std::vector<std::string> &names() const { return m_names; }

  /// Row of `name`, or nothing if the list does not have it.
  std::optional<std::size_t> find(std::string_view name) const;
  bool contains(std::string_view name) const { return find(name).has_value(); }

  /**
   * @brief The list's names that differ from `name` only in case.
   *
   * For error messages: `So_T` asked of a list holding `So_t` is a typo,
   * and saying so is worth more than "not found".
   */
  std::vector<std::string> near_misses(std::string_view name) const;

  /// The colon-separated form, the inverse of parse().
  std::string to_string() const;

private:
  std::vector<std::string> m_names;
  std::unordered_map<std::string, std::size_t> m_index;
};

} // namespace fields
} // namespace emulator

#endif // E3SM_EMULATOR_FIELDS_FIELD_LIST_HPP
