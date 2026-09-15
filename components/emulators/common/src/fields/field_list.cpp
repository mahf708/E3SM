/**
 * @file field_list.cpp
 * @brief Implementation of FieldList.
 */

#include "field_list.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace emulator {
namespace fields {

namespace {

std::string_view trim(std::string_view s) {
  const auto is_space = [](char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
  };
  while (!s.empty() && is_space(s.front())) {
    s.remove_prefix(1);
  }
  while (!s.empty() && is_space(s.back())) {
    s.remove_suffix(1);
  }
  return s;
}

bool iequal(std::string_view a, std::string_view b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
           return std::tolower(static_cast<unsigned char>(x)) ==
                  std::tolower(static_cast<unsigned char>(y));
         });
}

} // namespace

FieldList::FieldList(std::vector<std::string> names)
    : m_names(std::move(names)) {
  m_index.reserve(m_names.size());
  for (std::size_t i = 0; i < m_names.size(); ++i) {
    if (m_names[i].empty()) {
      throw std::invalid_argument("Field list entry " + std::to_string(i + 1) +
                                  " is empty.");
    }
    const auto [it, inserted] = m_index.emplace(m_names[i], i);
    if (!inserted) {
      throw std::invalid_argument(
          "Field '" + m_names[i] + "' appears twice in a field list, at " +
          std::to_string(it->second + 1) + " and " + std::to_string(i + 1) +
          ". An attribute vector cannot have two rows with one name.");
    }
  }
}

FieldList FieldList::parse(std::string_view colon_separated) {
  const auto nul = colon_separated.find('\0');
  if (nul != std::string_view::npos) {
    colon_separated = colon_separated.substr(0, nul);
  }
  colon_separated = trim(colon_separated);

  std::vector<std::string> names;
  if (colon_separated.empty()) {
    return FieldList(std::move(names));
  }

  std::size_t start = 0;
  while (true) {
    const auto colon = colon_separated.find(':', start);
    const auto piece = colon_separated.substr(
        start, colon == std::string_view::npos ? std::string_view::npos
                                               : colon - start);
    names.emplace_back(trim(piece));
    if (colon == std::string_view::npos) {
      break;
    }
    start = colon + 1;
  }
  return FieldList(std::move(names));
}

std::optional<std::size_t> FieldList::find(std::string_view name) const {
  const auto it = m_index.find(std::string(name));
  if (it == m_index.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::vector<std::string> FieldList::near_misses(std::string_view name) const {
  std::vector<std::string> out;
  for (const auto &candidate : m_names) {
    if (candidate != name && iequal(candidate, name)) {
      out.push_back(candidate);
    }
  }
  return out;
}

std::string FieldList::to_string() const {
  std::string out;
  for (std::size_t i = 0; i < m_names.size(); ++i) {
    if (i) {
      out += ':';
    }
    out += m_names[i];
  }
  return out;
}

} // namespace fields
} // namespace emulator
