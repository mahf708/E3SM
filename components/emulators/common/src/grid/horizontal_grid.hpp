/**
 * @file horizontal_grid.hpp
 * @brief A component's global horizontal grid, and how it is split and
 *        presented to the coupler.
 */

#ifndef E3SM_EMULATOR_GRID_HORIZONTAL_GRID_HPP
#define E3SM_EMULATOR_GRID_HORIZONTAL_GRID_HPP

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace emulator {
namespace grid {

/**
 * @brief Cell centres, areas and mask of a whole grid, in SCRIP order.
 *
 * Units are the coupler's: latitude and longitude in degrees, area as solid
 * angle in radians squared (SCRIP calls it steradians); a reader converts
 * once and nothing downstream converts again. `nx` and `ny` are the logical
 * shape a model sees, with cell `i` at row `i / nx`, column `i % nx` (an
 * unstructured grid has `ny == 1`).
 */
struct HorizontalGrid {
  std::string name; ///< Where it came from, for messages
  int nx = 0;
  int ny = 0;
  std::vector<double> lat;  ///< [degrees north]
  std::vector<double> lon;  ///< [degrees east]
  std::vector<double> area; ///< [radians^2]
  std::vector<int> imask;   ///< 1 where the grid is active, 0 where not

  std::size_t size() const { return lat.size(); }

  /**
   * @brief Refuse a grid that would fail later, somewhere less clear.
   *
   * Checks that every array has nx*ny entries, latitudes are within
   * [-90, 90], longitudes are finite, areas are positive and the mask is
   * binary.  With `expect_global`, also that the areas sum to 4 pi to within
   * `global_area_tolerance` (relative): a grid in square degrees, or one that
   * silently lost a row, is off by far more than that.
   *
   * @throws std::invalid_argument naming the first problem and the grid
   */
  void validate(bool expect_global = false,
                double global_area_tolerance = 1e-6) const;

  /// Sum of `area`, in radians^2.
  double total_area() const;
};

/**
 * @brief Which global cells this rank owns: contiguous blocks, in order.
 *
 * Rank r owns the r-th of `nranks` contiguous blocks, the first
 * `ncells % nranks` blocks one cell larger. Global ids are 1-based, as MCT's
 * gsMap expects. Contiguity also lets a model that consumes `[ny, nx]`
 * fields gather each rank's cells in row order cheaply.
 */
class Decomposition {
public:
  Decomposition() = default;

  /// @throws std::invalid_argument on nranks < 1 or rank outside [0, nranks)
  static Decomposition contiguous_blocks(std::size_t ncells, int nranks,
                                         int rank);

  std::size_t num_global() const { return m_num_global; }
  std::size_t num_local() const { return m_count; }
  /// First global cell (0-based) this rank owns.
  std::size_t offset() const { return m_offset; }
  int rank() const { return m_rank; }
  int nranks() const { return m_nranks; }

  /// 1-based global ids of this rank's cells, for the gsMap.
  std::vector<int> global_ids() const;

  /// This rank's slice of a global array.
  /// @throws std::invalid_argument if `global.size()` is not num_global()
  template <typename T>
  std::vector<T> local(std::span<const T> global) const {
    check_global_size(global.size());
    return std::vector<T>(global.begin() + m_offset,
                          global.begin() + m_offset + m_count);
  }
  template <typename T>
  std::vector<T> local(const std::vector<T> &global) const {
    return local(std::span<const T>(global));
  }

private:
  void check_global_size(std::size_t n) const;

  std::size_t m_num_global = 0;
  std::size_t m_offset = 0;
  std::size_t m_count = 0;
  int m_rank = 0;
  int m_nranks = 1;
};

/**
 * @brief The domain a component hands the coupler for its own cells.
 *
 * `mask` and `frac` are what the MCT driver checks between components. A
 * component that covers every cell (the atmosphere) reports mask and frac of
 * exactly 1; a partial-coverage surface component (the ocean, and the sea
 * ice that shares its grid) reports a **binary** mask with frac equal to it,
 * because seq_domain_mct maps the ocean mask onto the atmosphere grid and
 * requires it to equal one minus the land fraction, which a continuous frac
 * fails on coastal cells. A continuous field is therefore refused as a mask
 * here rather than thresholded: the threshold is a model decision, made
 * where the model is configured.
 */
struct Domain {
  std::vector<int> global_ids; ///< 1-based
  std::vector<double> lat;
  std::vector<double> lon;
  std::vector<double> area;
  std::vector<double> mask;
  std::vector<double> frac;

  std::size_t size() const { return lat.size(); }

  /// Every cell active: mask and frac of 1.  The grid's own imask is ignored.
  static Domain full(const HorizontalGrid &grid, const Decomposition &decomp);

  /**
   * @brief A partial surface: mask and frac from `surface_mask`.
   *
   * @param surface_mask one value per *global* cell, each exactly 0 or 1
   * @throws std::invalid_argument if the mask has the wrong size or any
   *         value other than 0 and 1, with the count and the first offender
   */
  static Domain masked(const HorizontalGrid &grid, const Decomposition &decomp,
                       std::span<const double> surface_mask);
};

} // namespace grid
} // namespace emulator

#endif // E3SM_EMULATOR_GRID_HORIZONTAL_GRID_HPP
