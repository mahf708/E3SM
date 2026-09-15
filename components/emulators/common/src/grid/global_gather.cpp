/**
 * @file global_gather.cpp
 * @brief Implementation of GlobalGather.
 */

#include "global_gather.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace emulator {
namespace grid {

GlobalGather::GlobalGather(MPI_Comm comm, const Decomposition &decomp,
                           int root)
    : m_comm(comm), m_root(root), m_num_global(decomp.num_global()),
      m_num_local(decomp.num_local()) {
  int size = 0;
  MPI_Comm_rank(comm, &m_rank);
  MPI_Comm_size(comm, &size);
  if (root < 0 || root >= size) {
    throw std::invalid_argument("Gather root " + std::to_string(root) +
                                " is not a rank of this communicator.");
  }
  if (m_num_global > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("A grid of " + std::to_string(m_num_global) +
                                " cells is too large for MPI int counts.");
  }

  // Every rank learns every block's (offset, count), so a bad layout fails
  // everywhere rather than hanging somewhere.
  const int mine[2] = {static_cast<int>(decomp.offset()),
                       static_cast<int>(decomp.num_local())};
  std::vector<int> all(2 * static_cast<std::size_t>(size));
  MPI_Allgather(mine, 2, MPI_INT, all.data(), 2, MPI_INT, comm);

  m_counts.resize(static_cast<std::size_t>(size));
  m_displs.resize(static_cast<std::size_t>(size));
  std::size_t expect_offset = 0;
  for (int r = 0; r < size; ++r) {
    const auto i = static_cast<std::size_t>(r);
    m_displs[i] = all[2 * i];
    m_counts[i] = all[2 * i + 1];
    if (static_cast<std::size_t>(m_displs[i]) != expect_offset) {
      throw std::invalid_argument(
          "Rank " + std::to_string(r) + "'s block starts at cell " +
          std::to_string(m_displs[i]) + " but the blocks before it end at " +
          std::to_string(expect_offset) +
          ". The decomposition does not tile the grid in rank order.");
    }
    expect_offset += static_cast<std::size_t>(m_counts[i]);
  }
  if (expect_offset != m_num_global) {
    throw std::invalid_argument(
        "The ranks' blocks cover " + std::to_string(expect_offset) +
        " cells of a " + std::to_string(m_num_global) + "-cell grid.");
  }
}

void GlobalGather::check(std::span<const double> local,
                         std::size_t global_size, const char *what) const {
  if (local.size() != m_num_local) {
    throw std::invalid_argument(std::string(what) + ": " +
                                std::to_string(local.size()) +
                                " local values for " +
                                std::to_string(m_num_local) + " columns.");
  }
  if (is_root() && global_size != m_num_global) {
    throw std::invalid_argument(std::string(what) + ": a global array of " +
                                std::to_string(global_size) + " for " +
                                std::to_string(m_num_global) + " cells.");
  }
}

void GlobalGather::gather(std::span<const double> local,
                          std::span<double> global) const {
  check(local, global.size(), "gather");
  MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
              is_root() ? global.data() : nullptr, m_counts.data(),
              m_displs.data(), MPI_DOUBLE, m_root, m_comm);
}

void GlobalGather::scatter(std::span<const double> global,
                           std::span<double> local) const {
  check(std::span<const double>(local.data(), local.size()), global.size(),
        "scatter");
  MPI_Scatterv(is_root() ? global.data() : nullptr, m_counts.data(),
               m_displs.data(), MPI_DOUBLE, local.data(),
               static_cast<int>(local.size()), MPI_DOUBLE, m_root, m_comm);
}

} // namespace grid
} // namespace emulator
