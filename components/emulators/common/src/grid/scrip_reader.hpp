/**
 * @file scrip_reader.hpp
 * @brief Read a HorizontalGrid from a SCRIP grid file.
 */

#ifndef E3SM_EMULATOR_GRID_SCRIP_READER_HPP
#define E3SM_EMULATOR_GRID_SCRIP_READER_HPP

#include <string>

#include "horizontal_grid.hpp"

namespace emulator {
namespace grid {

/// Whether this build can read SCRIP files (it needs netCDF).
bool have_scrip_reader();

/**
 * @brief Read `grid_center_lat`, `grid_center_lon`, `grid_area`,
 *        `grid_imask` and `grid_dims` from a SCRIP file.
 *
 * - Coordinates are converted to degrees according to their `units`
 *   attribute (`degrees`, `degrees_north`/`_east` or `radians`); a
 *   coordinate variable with no units, or units this does not recognise, is
 *   an error rather than a guess.
 * - `grid_area` is taken as radians squared (SCRIP calls it steradians).
 *   Units of square degrees are refused: the coupler's areas are solid
 *   angles, and a grid in square degrees is 3283 times too large.
 * - `grid_imask` is optional and defaults to 1 everywhere.
 * - `grid_dims` is in Fortran order, `[nx, ny]`, as SCRIP writes it.  A
 *   rank-1 grid is `[ncells]` and becomes nx = ncells, ny = 1.
 *
 * The grid is validated before it is returned, and, when `expect_global`, it
 * must cover the sphere (HorizontalGrid::validate).
 *
 * @throws std::runtime_error naming the file and variable on any read
 *         failure, including when this build has no netCDF
 */
HorizontalGrid read_scrip(const std::string &path, bool expect_global = true);

} // namespace grid
} // namespace emulator

#endif // E3SM_EMULATOR_GRID_SCRIP_READER_HPP
