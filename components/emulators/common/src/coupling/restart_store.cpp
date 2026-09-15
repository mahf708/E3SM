/**
 * @file restart_store.cpp
 * @brief Implementation of MemoryRestartStore.
 */

#include "restart_store.hpp"

#include <algorithm>
#include <stdexcept>

namespace emulator {
namespace coupling {

std::string restart_name(std::string_view prefix, std::string_view name) {
  std::string out(prefix);
  if (!out.empty()) {
    out += '.';
  }
  out += name;
  return out;
}

void MemoryRestartStore::write_array(std::string_view name,
                                     std::span<const double> data) {
  m_arrays[std::string(name)].assign(data.begin(), data.end());
}

void MemoryRestartStore::write_int(std::string_view name, std::int64_t value) {
  m_ints[std::string(name)] = value;
}

bool MemoryRestartStore::read_array(std::string_view name,
                                    std::span<double> data) {
  const auto it = m_arrays.find(name);
  if (it == m_arrays.end()) {
    return false;
  }
  if (it->second.size() != data.size()) {
    throw std::runtime_error(
        "Restart array '" + std::string(name) + "' has " +
        std::to_string(it->second.size()) + " values; expected " +
        std::to_string(data.size()) +
        ". The restart was written with a different decomposition or grid.");
  }
  std::copy(it->second.begin(), it->second.end(), data.begin());
  return true;
}

bool MemoryRestartStore::read_int(std::string_view name, std::int64_t &value) {
  const auto it = m_ints.find(name);
  if (it == m_ints.end()) {
    return false;
  }
  value = it->second;
  return true;
}

bool MemoryRestartStore::has(std::string_view name) const {
  return m_arrays.find(name) != m_arrays.end() ||
         m_ints.find(name) != m_ints.end();
}

void MemoryRestartStore::erase(std::string_view name) {
  if (auto it = m_arrays.find(name); it != m_arrays.end()) {
    m_arrays.erase(it);
  }
  if (auto it = m_ints.find(name); it != m_ints.end()) {
    m_ints.erase(it);
  }
}

std::vector<std::string> MemoryRestartStore::names() const {
  std::vector<std::string> out;
  for (const auto &[n, _] : m_arrays) {
    out.push_back(n);
  }
  for (const auto &[n, _] : m_ints) {
    out.push_back(n);
  }
  std::sort(out.begin(), out.end());
  return out;
}

} // namespace coupling
} // namespace emulator
