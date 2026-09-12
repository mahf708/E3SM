/**
 * @file exchange.cpp
 * @brief Implementation of Exchange.
 */

#include "exchange.hpp"

#include <stdexcept>

namespace emulator {
namespace coupling {

void Exchange::publish(std::string_view name, std::span<const double> values) {
  auto it = m_fields.find(name);
  if (it == m_fields.end()) {
    it = m_fields.emplace(std::string(name), Entry{}).first;
    it->second.values.resize(values.size());
  } else if (it->second.values.size() != values.size()) {
    throw std::invalid_argument(
        "Exchange field '" + std::string(name) + "' was published with " +
        std::to_string(it->second.values.size()) + " values and now with " +
        std::to_string(values.size()) +
        ". Components sharing a field must share the grid decomposition.");
  }
  std::copy(values.begin(), values.end(), it->second.values.begin());
  ++it->second.publishes;
}

bool Exchange::has(std::string_view name) const {
  return m_fields.find(name) != m_fields.end();
}

std::span<const double> Exchange::get(std::string_view name) const {
  const auto it = m_fields.find(name);
  if (it == m_fields.end()) {
    std::string held;
    for (const auto &[n, _] : m_fields) {
      held += (held.empty() ? "" : ", ") + n;
    }
    throw std::out_of_range("Nothing has published '" + std::string(name) +
                            "' to the exchange; it holds: " +
                            (held.empty() ? "nothing" : held) +
                            ". Is the publishing component in this run?");
  }
  return it->second.values;
}

std::int64_t Exchange::publishes(std::string_view name) const {
  const auto it = m_fields.find(name);
  return it == m_fields.end() ? 0 : it->second.publishes;
}

Exchange &Exchange::process() {
  static Exchange exchange;
  return exchange;
}

} // namespace coupling
} // namespace emulator
