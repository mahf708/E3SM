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
 * For data that has no MCT field, or that the coupler would degrade in
 * transit; since both components run in one process, this passes it in
 * memory instead. Every field is on one rank's cells, and the components
 * exchanging it must share the grid decomposition: the size is checked on
 * every publish, and the publish count lets a reader see whether a value
 * is fresh.
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
 * Lets a component reuse another's published domain instead of reading grid
 * files again, so the two are guaranteed consistent; a component that has
 * published none finds nothing rather than a mismatched grid.
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
