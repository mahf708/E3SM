/**
 * @file grid_field_reader.hpp
 * @brief Read whole-grid 2D fields, such as an emulator's initial
 *        condition, from a netCDF file.
 */

#ifndef E3SM_EMULATOR_GRID_GRID_FIELD_READER_HPP
#define E3SM_EMULATOR_GRID_GRID_FIELD_READER_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace emulator {
namespace grid {

/// One variable as read, and what in it is not a usable number.
struct GridField {
  std::string name;
  std::vector<double> values; ///< ny * nx, row-major: j * nx + i
  /// NaN or infinite values.
  std::size_t non_finite = 0;
  /// Finite values at or above 1e30, or equal to the variable's _FillValue.
  /// A netCDF fill value of 9.97e36 is finite and passes a finiteness check;
  /// one of those in a global network's input poisons every cell.
  std::size_t fill_like = 0;

  std::size_t unusable() const { return non_finite + fill_like; }
};

/**
 * @brief Read `names`, each a `(lat, lon)` or `(time=1, lat, lon)` variable
 *        of `ny` by `nx`, as doubles.
 *
 * Nothing is filled or clipped here: what to do with an unusable value is a
 * per-channel decision (a surface fraction may be zero-filled; a
 * temperature may not), so the counts are reported and the caller decides.
 *
 * @throws std::runtime_error naming the file and variable if a variable is
 *         missing or has another shape, or if this build has no netCDF
 */
std::vector<GridField> read_grid_fields(const std::string &path,
                                        const std::vector<std::string> &names,
                                        int ny, int nx);

} // namespace grid
} // namespace emulator

#endif // E3SM_EMULATOR_GRID_GRID_FIELD_READER_HPP
