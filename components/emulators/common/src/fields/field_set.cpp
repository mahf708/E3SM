/**
 * @file field_set.cpp
 * @brief Implementation of FieldSet.
 */

#include "field_set.hpp"

#include <stdexcept>

namespace emulator {
namespace fields {

std::span<double> FieldSet::add(const std::string &name, double fill) {
  if (name.empty()) {
    throw std::invalid_argument("A field needs a name.");
  }
  const auto [it, inserted] = m_index.emplace(name, m_names.size());
  if (!inserted) {
    throw std::invalid_argument("Field '" + name +
                                "' is already in this field set.");
  }
  m_names.push_back(name);
  m_data.emplace_back(m_npoints, fill);
  return m_data.back();
}

bool FieldSet::contains(std::string_view name) const {
  return m_index.find(std::string(name)) != m_index.end();
}

std::size_t FieldSet::index_of(std::string_view name) const {
  const auto it = m_index.find(std::string(name));
  if (it == m_index.end()) {
    std::string held;
    for (const auto &n : m_names) {
      held += (held.empty() ? "" : ", ") + n;
    }
    throw std::out_of_range("No field '" + std::string(name) +
                            "' in this field set; it holds: " +
                            (held.empty() ? "nothing" : held) + ".");
  }
  return it->second;
}

std::span<double> FieldSet::get(std::string_view name) {
  return m_data[index_of(name)];
}

std::span<const double> FieldSet::get(std::string_view name) const {
  return m_data[index_of(name)];
}

} // namespace fields
} // namespace emulator
