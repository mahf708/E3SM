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

namespace {

std::string key(std::string_view component, const char *what) {
  return std::string(component) + ".domain." + what;
}

} // namespace

void publish_domain(Exchange &exchange, std::string_view component,
                    const SharedDomain &shared) {
  const auto &d = shared.domain;
  const std::vector<double> ids(d.global_ids.begin(), d.global_ids.end());
  exchange.publish(key(component, "global_ids"), ids);
  exchange.publish(key(component, "lat"), d.lat);
  exchange.publish(key(component, "lon"), d.lon);
  exchange.publish(key(component, "area"), d.area);
  exchange.publish(key(component, "mask"), d.mask);
  exchange.publish(key(component, "frac"), d.frac);
  const std::vector<double> shape{static_cast<double>(shared.nx),
                                  static_cast<double>(shared.ny),
                                  static_cast<double>(shared.num_global)};
  exchange.publish(key(component, "shape"), shape);
}

bool has_domain(const Exchange &exchange, std::string_view component) {
  return exchange.has(key(component, "shape"));
}

SharedDomain shared_domain(const Exchange &exchange,
                           std::string_view component) {
  const auto copy = [&](const char *what) {
    const auto v = exchange.get(key(component, what));
    return std::vector<double>(v.begin(), v.end());
  };
  SharedDomain s;
  const auto shape = copy("shape");
  s.nx = static_cast<int>(shape.at(0));
  s.ny = static_cast<int>(shape.at(1));
  s.num_global = static_cast<std::size_t>(shape.at(2));
  const auto ids = copy("global_ids");
  s.domain.global_ids.assign(ids.begin(), ids.end());
  s.domain.lat = copy("lat");
  s.domain.lon = copy("lon");
  s.domain.area = copy("area");
  s.domain.mask = copy("mask");
  s.domain.frac = copy("frac");
  return s;
}

} // namespace coupling
} // namespace emulator
