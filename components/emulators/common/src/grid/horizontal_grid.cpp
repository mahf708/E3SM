/**
 * @file horizontal_grid.cpp
 * @brief Implementation of HorizontalGrid, Decomposition and Domain.
 */

#include "horizontal_grid.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace emulator {
namespace grid {

namespace {

[[noreturn]] void fail(const std::string &grid, const std::string &what) {
  throw std::invalid_argument("Grid '" + grid + "': " + what);
}

} // namespace

// ===========================================================================
// HorizontalGrid
// ===========================================================================

double HorizontalGrid::total_area() const {
  double sum = 0.0;
  for (const double a : area) {
    sum += a;
  }
  return sum;
}

void HorizontalGrid::validate(bool expect_global,
                              double global_area_tolerance) const {
  if (nx <= 0 || ny <= 0) {
    fail(name, "nx and ny must be positive; got " + std::to_string(nx) +
                   " and " + std::to_string(ny) + ".");
  }
  const auto n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  auto check_size = [&](std::size_t got, const char *what) {
    if (got != n) {
      fail(name, std::string(what) + " has " + std::to_string(got) +
                     " entries but nx*ny is " + std::to_string(n) + ".");
    }
  };
  check_size(lat.size(), "lat");
  check_size(lon.size(), "lon");
  check_size(area.size(), "area");
  check_size(imask.size(), "imask");

  for (std::size_t i = 0; i < n; ++i) {
    if (!(lat[i] >= -90.0 && lat[i] <= 90.0)) {
      std::ostringstream oss;
      oss << "latitude " << lat[i] << " at cell " << i
          << " is outside [-90, 90]. If the file is in radians its units "
             "attribute should say so.";
      fail(name, oss.str());
    }
    if (!std::isfinite(lon[i])) {
      fail(name, "longitude at cell " + std::to_string(i) + " is not finite.");
    }
    if (!(area[i] > 0.0) || !std::isfinite(area[i])) {
      std::ostringstream oss;
      oss << "area " << area[i] << " at cell " << i << " is not positive.";
      fail(name, oss.str());
    }
    if (imask[i] != 0 && imask[i] != 1) {
      fail(name, "imask at cell " + std::to_string(i) + " is " +
                     std::to_string(imask[i]) + ", not 0 or 1.");
    }
  }

  if (expect_global) {
    const double sphere = 4.0 * std::numbers::pi;
    const double total = total_area();
    if (std::abs(total - sphere) > global_area_tolerance * sphere) {
      std::ostringstream oss;
      oss.precision(12);
      oss << "areas sum to " << total << " but a global grid covers 4 pi = "
          << sphere << " radians^2 (relative error "
          << std::abs(total - sphere) / sphere
          << "). Areas in square degrees sum to about 41253.";
      fail(name, oss.str());
    }
  }
}

// ===========================================================================
// Decomposition
// ===========================================================================

Decomposition Decomposition::contiguous_blocks(std::size_t ncells, int nranks,
                                               int rank) {
  if (nranks < 1) {
    throw std::invalid_argument("A decomposition needs at least one rank; got " +
                                std::to_string(nranks) + ".");
  }
  if (rank < 0 || rank >= nranks) {
    throw std::invalid_argument("Rank " + std::to_string(rank) +
                                " is outside [0, " + std::to_string(nranks) +
                                ").");
  }
  const auto size = static_cast<std::size_t>(nranks);
  const auto r = static_cast<std::size_t>(rank);
  const std::size_t base = ncells / size;
  const std::size_t extra = ncells % size;

  Decomposition d;
  d.m_num_global = ncells;
  d.m_offset = r * base + std::min(r, extra);
  d.m_count = base + (r < extra ? 1 : 0);
  d.m_rank = rank;
  d.m_nranks = nranks;
  return d;
}

std::vector<int> Decomposition::global_ids() const {
  std::vector<int> ids(m_count);
  for (std::size_t i = 0; i < m_count; ++i) {
    ids[i] = static_cast<int>(m_offset + i + 1);
  }
  return ids;
}

void Decomposition::check_global_size(std::size_t n) const {
  if (n != m_num_global) {
    throw std::invalid_argument(
        "A global array of " + std::to_string(n) +
        " values handed to a decomposition of " +
        std::to_string(m_num_global) + " cells.");
  }
}

// ===========================================================================
// Domain
// ===========================================================================

namespace {

Domain geometry(const HorizontalGrid &grid, const Decomposition &decomp) {
  if (grid.size() != decomp.num_global()) {
    throw std::invalid_argument(
        "Grid '" + grid.name + "' has " + std::to_string(grid.size()) +
        " cells but the decomposition was built for " +
        std::to_string(decomp.num_global()) + ".");
  }
  Domain d;
  d.global_ids = decomp.global_ids();
  d.lat = decomp.local(grid.lat);
  d.lon = decomp.local(grid.lon);
  d.area = decomp.local(grid.area);
  return d;
}

} // namespace

Domain Domain::full(const HorizontalGrid &grid, const Decomposition &decomp) {
  Domain d = geometry(grid, decomp);
  d.mask.assign(d.size(), 1.0);
  d.frac.assign(d.size(), 1.0);
  return d;
}

Domain Domain::masked(const HorizontalGrid &grid, const Decomposition &decomp,
                      std::span<const double> surface_mask) {
  if (surface_mask.size() != grid.size()) {
    throw std::invalid_argument(
        "Grid '" + grid.name + "': a surface mask of " +
        std::to_string(surface_mask.size()) + " values for " +
        std::to_string(grid.size()) + " cells.");
  }
  std::size_t nonbinary = 0;
  std::size_t first = 0;
  for (std::size_t i = 0; i < surface_mask.size(); ++i) {
    if (surface_mask[i] != 0.0 && surface_mask[i] != 1.0) {
      if (nonbinary == 0) {
        first = i;
      }
      ++nonbinary;
    }
  }
  if (nonbinary > 0) {
    std::ostringstream oss;
    oss << "Grid '" << grid.name << "': the surface mask has " << nonbinary
        << " cells that are neither 0 nor 1 (the first is cell " << first
        << ", value " << surface_mask[first]
        << "). The coupler needs a binary mask with frac equal to it: a "
           "continuous fraction beside a binary mask fails seq_domain_mct's "
           "check on every coastal cell. Threshold it where the model is "
           "configured.";
    throw std::invalid_argument(oss.str());
  }

  Domain d = geometry(grid, decomp);
  d.mask = decomp.local(surface_mask);
  d.frac = d.mask;
  return d;
}

} // namespace grid
} // namespace emulator
