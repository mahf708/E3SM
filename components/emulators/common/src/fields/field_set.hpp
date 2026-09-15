/**
 * @file field_set.hpp
 * @brief Component-owned storage: one contiguous array per named field.
 */

#ifndef E3SM_EMULATOR_FIELDS_FIELD_SET_HPP
#define E3SM_EMULATOR_FIELDS_FIELD_SET_HPP

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace emulator {
namespace fields {

/**
 * @brief Named, equal-length, contiguous field arrays on a component's columns.
 *
 * This is where a component keeps what it exchanges with the coupler, and is
 * deliberately *not* the coupler's buffer: MCT lays an attribute vector out
 * as `rAttr(nfields, npoints)`, so one field there is a stride-`nfields` walk
 * rather than an array.  Copying at the seam (see CouplerBinding) gives every
 * field a `std::span<double>`, matching the C++ coupler's `FieldBuffer`.
 * Adding a field after a span has been taken is safe, since each field has
 * its own vector.
 */
class FieldSet {
public:
  explicit FieldSet(std::size_t npoints = 0) : m_npoints(npoints) {}

  FieldSet(const FieldSet &) = delete;
  FieldSet &operator=(const FieldSet &) = delete;
  FieldSet(FieldSet &&) = default;
  FieldSet &operator=(FieldSet &&) = default;

  /**
   * @brief Add a field, filled with `fill`.
   * @throws std::invalid_argument if the name is empty or already present.
   */
  std::span<double> add(const std::string &name, double fill = 0.0);

  bool contains(std::string_view name) const;

  /// @throws std::out_of_range naming the field and what the set does hold.
  std::span<double> get(std::string_view name);
  std::span<const double> get(std::string_view name) const;

  /// Field names, in the order they were added.
  const std::vector<std::string> &names() const { return m_names; }

  std::size_t size() const { return m_names.size(); }
  std::size_t npoints() const { return m_npoints; }

private:
  std::size_t index_of(std::string_view name) const;

  std::size_t m_npoints = 0;
  std::vector<std::string> m_names;
  std::vector<std::vector<double>> m_data;
  std::unordered_map<std::string, std::size_t> m_index;
};

} // namespace fields
} // namespace emulator

#endif // E3SM_EMULATOR_FIELDS_FIELD_SET_HPP
