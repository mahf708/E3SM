/**
 * @file component_settings.cpp
 * @brief Implementation of ComponentSettings.
 */

#include "component_settings.hpp"

#include <fstream>

namespace emulator {

ComponentSettings ComponentSettings::read(const std::string &path) {
  ComponentSettings s;
  if (path.empty()) {
    return s;
  }
  std::ifstream ifs(path);
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto pos = line.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    std::string key = line.substr(0, pos);
    std::string val = line.substr(pos + 1);
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    val.erase(0, val.find_first_not_of(" \t"));
    val.erase(val.find_last_not_of(" \t") + 1);
    s.m_values[key] = val;
  }
  return s;
}

std::string ComponentSettings::get(const std::string &key,
                                   const std::string &fallback) const {
  const auto it = m_values.find(key);
  return it == m_values.end() ? fallback : it->second;
}

bool ComponentSettings::get_bool(const std::string &key, bool fallback) const {
  const auto it = m_values.find(key);
  if (it == m_values.end()) {
    return fallback;
  }
  if (it->second == "true") {
    return true;
  }
  if (it->second == "false") {
    return false;
  }
  throw std::invalid_argument("Setting '" + key + ": " + it->second +
                              "' is neither true nor false.");
}

} // namespace emulator
