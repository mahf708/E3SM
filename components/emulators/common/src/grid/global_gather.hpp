/**
 * @file global_gather.hpp
 * @brief Moving a decomposed field to and from the whole-grid array a
 *        global model sees.
 */

#ifndef E3SM_EMULATOR_GRID_GLOBAL_GATHER_HPP
#define E3SM_EMULATOR_GRID_GLOBAL_GATHER_HPP

#include <mpi.h>

#include <cstddef>
#include <span>
#include <vector>

#include "horizontal_grid.hpp"

namespace emulator {
namespace grid {

/**
 * @brief Gather local columns into a global array on one rank, and scatter
 *        back.
 *
 * ACE and Samudra are global models: one `[1, C, ny, nx]` tensor per step,
 * not a column at a time.  The coupler decomposes the grid across ranks, so
 * every step the inputs are gathered to the rank that runs the model and the
 * outputs scattered back.  Every rank must call gather() and scatter() the
 * same number of times, in the same order -- they are collectives.
 *
 * The global array is in the grid's own cell order (SCRIP order), which for
 * a structured grid is row-major: cell `j * nx + i`.  Whether that is the
 * latitude order a model was trained on is the model adapter's business, not
 * this class's.
 *
 * Built from a Decomposition on every rank; the constructor checks, with one
 * collective, that the ranks' blocks tile the grid in rank order.
 */
class GlobalGather {
public:
  /// @throws std::invalid_argument if the blocks do not tile the grid
  GlobalGather(MPI_Comm comm, const Decomposition &decomp, int root = 0);

  bool is_root() const { return m_rank == m_root; }
  std::size_t num_global() const { return m_num_global; }
  std::size_t num_local() const { return m_num_local; }

  /**
   * @param local  this rank's num_local() values
   * @param global num_global() values on the root; ignored (may be empty)
   *               elsewhere
   */
  void gather(std::span<const double> local, std::span<double> global) const;

  /// The inverse of gather(): `global` is read on the root only.
  void scatter(std::span<const double> global, std::span<double> local) const;

private:
  void check(std::span<const double> local, std::size_t global_size,
             const char *what) const;

  MPI_Comm m_comm;
  int m_root;
  int m_rank = 0;
  std::size_t m_num_global = 0;
  std::size_t m_num_local = 0;
  std::vector<int> m_counts;
  std::vector<int> m_displs;
};

} // namespace grid
} // namespace emulator

#endif // E3SM_EMULATOR_GRID_GLOBAL_GATHER_HPP
