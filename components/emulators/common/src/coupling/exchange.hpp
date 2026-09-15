/**
 * @file exchange.hpp
 * @brief Fields passed directly between emulated components in one process.
 */

#ifndef E3SM_EMULATOR_COUPLING_EXCHANGE_HPP
#define E3SM_EMULATOR_COUPLING_EXCHANGE_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "horizontal_grid.hpp"

namespace emulator {
namespace coupling {

/**
 * @brief Named fields one emulated component publishes and another reads.
 *
 * Some exchanges between emulators have no MCT field, or lose what the
 * emulators need on the way through the coupler.  SamudrACE drives its ocean
 * with the atmosphere emulator's own flux channels rather than the coupler's
 * bulk-formula fluxes, and its atmosphere reads the ocean's unmerged SST and
 * its own sea-ice fraction.  In E3SM both components run in one executable,
 * so this passes them in memory.
 *
 * Every field is on one rank's cells, and the components exchanging it must
 * share the grid and its decomposition: the size is checked on every
 * publish, and the publish count lets a reader see whether a value is fresh.
 */
class Exchange {
public:
  /// Store a copy.  A name's size is fixed by its first publish.
  /// @throws std::invalid_argument on a size that differs from that
  void publish(std::string_view name, std::span<const double> values);

  bool has(std::string_view name) const;
  /// @throws std::out_of_range naming the field and what has been published
  std::span<const double> get(std::string_view name) const;
  /// How many times `name` has been published; 0 if never.
  std::int64_t publishes(std::string_view name) const;

  /// The process's exchange, shared by the components built into it.
  static Exchange &process();

private:
  struct Entry {
    std::vector<double> values;
    std::int64_t publishes = 0;
  };
  std::map<std::string, Entry, std::less<>> m_fields;
};

/**
 * @brief A component's domain, as another component in the process takes it.
 *
 * The sea ice that goes with an emulated ocean has no grid of its own: it is
 * the ocean's ice.  Taking the ocean's published domain, rather than reading
 * the same files again, makes the coupler's ocean-ice domain check a
 * tautology instead of a coincidence, and a component on other ranks finds
 * nothing rather than a plausible, mis-indexed grid.
 */
struct SharedDomain {
  grid::Domain domain;
  int nx = 0;
  int ny = 0;
  std::size_t num_global = 0;
};

/// Publish `component`.domain.{global_ids,lat,lon,area,mask,frac,shape}.
void publish_domain(Exchange &exchange, std::string_view component,
                    const SharedDomain &shared);
bool has_domain(const Exchange &exchange, std::string_view component);
/// @throws std::out_of_range if `component` has published no domain here
SharedDomain shared_domain(const Exchange &exchange,
                           std::string_view component);

} // namespace coupling
} // namespace emulator

#endif // E3SM_EMULATOR_COUPLING_EXCHANGE_HPP
