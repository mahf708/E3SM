/**
 * @file mask_set.hpp
 * @brief Named binary masks over a component's cells.
 */

#ifndef E3SM_EMULATOR_FIELDS_MASK_SET_HPP
#define E3SM_EMULATOR_FIELDS_MASK_SET_HPP

#include <cstddef>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace emulator {
namespace fields {

/**
 * @brief Where each output channel is valid, by name.
 *
 * A mask belongs to a channel, not to a component: different output channels
 * can be valid over different subsets of cells.  An export names the mask
 * that bounds it (FieldSpec::mask), and CouplerBinding writes zero outside
 * it.
 *
 * Masks are binary, for the same reason the coupler domain's is: a fraction
 * belongs in a field, and which threshold turns it into a mask is a model
 * decision to be made where the model is configured.
 *
 * Masks live in a std::map, so a span handed out for one stays valid when
 * others are added.
 */
class MaskSet {
public:
  explicit MaskSet(std::size_t npoints = 0) : m_npoints(npoints) {}

  /**
   * @brief Add or replace a mask.
   * @throws std::invalid_argument on a wrong length or a value other than
   *         0 and 1, naming the count and the first offender
   */
  void set(const std::string &name, std::span<const double> values);

  bool contains(std::string_view name) const;

  /// @throws std::out_of_range naming the masks there are
  std::span<const double> get(std::string_view name) const;

  /// Cells where the mask is 1.
  std::size_t count(std::string_view name) const;

  std::vector<std::string> names() const;
  std::size_t npoints() const { return m_npoints; }

private:
  std::size_t m_npoints = 0;
  std::map<std::string, std::vector<double>, std::less<>> m_masks;
};

} // namespace fields
} // namespace emulator

#endif // E3SM_EMULATOR_FIELDS_MASK_SET_HPP
