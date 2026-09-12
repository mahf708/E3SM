/**
 * @file mask_set.cpp
 * @brief Implementation of MaskSet.
 */

#include "mask_set.hpp"

#include <sstream>
#include <stdexcept>

namespace emulator {
namespace fields {

void MaskSet::set(const std::string &name, std::span<const double> values) {
  if (name.empty()) {
    throw std::invalid_argument("A mask needs a name.");
  }
  if (values.size() != m_npoints) {
    throw std::invalid_argument(
        "Mask '" + name + "' has " + std::to_string(values.size()) +
        " values for " + std::to_string(m_npoints) + " points.");
  }
  std::size_t bad = 0;
  std::size_t first = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i] != 0.0 && values[i] != 1.0) {
      if (bad++ == 0) {
        first = i;
      }
    }
  }
  if (bad > 0) {
    std::ostringstream oss;
    oss << "Mask '" << name << "' has " << bad
        << " values that are neither 0 nor 1 (the first at point " << first
        << " is " << values[first]
        << "). A mask is binary; threshold a fraction where the model is "
           "configured.";
    throw std::invalid_argument(oss.str());
  }
  m_masks[name].assign(values.begin(), values.end());
}

bool MaskSet::contains(std::string_view name) const {
  return m_masks.find(name) != m_masks.end();
}

std::span<const double> MaskSet::get(std::string_view name) const {
  const auto it = m_masks.find(name);
  if (it == m_masks.end()) {
    std::string held;
    for (const auto &[n, _] : m_masks) {
      held += (held.empty() ? "" : ", ") + n;
    }
    throw std::out_of_range("No mask '" + std::string(name) +
                            "'; the masks are: " +
                            (held.empty() ? "none" : held) + ".");
  }
  return it->second;
}

std::size_t MaskSet::count(std::string_view name) const {
  std::size_t n = 0;
  for (const double v : get(name)) {
    n += v == 1.0 ? 1 : 0;
  }
  return n;
}

std::vector<std::string> MaskSet::names() const {
  std::vector<std::string> out;
  for (const auto &[n, _] : m_masks) {
    out.push_back(n);
  }
  return out;
}

} // namespace fields
} // namespace emulator
