/**
 * @file yaml_config.cpp
 * @brief Implementation of Section.
 */

#include "yaml_config.hpp"

#include <algorithm>
#include <regex>

namespace emulator {
namespace config {

namespace {

std::string join(const std::string &where, const std::string &key) {
  return where + (where.back() == ' ' ? "" : ".") + key;
}

[[noreturn]] void fail(const std::string &where, const std::string &what) {
  throw std::invalid_argument(where + ": " + what);
}

template <typename T>
T scalar(const YAML::Node &node, const std::string &where, const char *type) {
  if (!node.IsScalar()) {
    fail(where, std::string("expected a ") + type);
  }
  try {
    return node.as<T>();
  } catch (const YAML::Exception &) {
    fail(where, std::string("expected a ") + type + ", got '" +
                    node.Scalar() + "'");
  }
}

} // namespace

std::vector<std::string> expand_range(const std::string &name) {
  static const std::regex range(R"(^(.*)\{(-?\d+)\.\.(-?\d+)\}(.*)$)");
  std::smatch m;
  if (!std::regex_match(name, m, range)) {
    return {name};
  }
  const int a = std::stoi(m[2]);
  const int b = std::stoi(m[3]);
  std::vector<std::string> out;
  for (int k = a; a <= b ? k <= b : k >= b; k += a <= b ? 1 : -1) {
    out.push_back(m[1].str() + std::to_string(k) + m[4].str());
  }
  return out;
}

Section Section::load_file(const std::string &path) {
  try {
    const auto slash = path.find_last_of('/');
    return {YAML::LoadFile(path),
            (slash == std::string::npos ? path : path.substr(slash + 1)) + ": "};
  } catch (const YAML::Exception &e) {
    throw std::runtime_error("Could not read YAML file '" + path + "': " +
                             e.what());
  }
}

Section Section::load_string(const std::string &text, const std::string &name) {
  try {
    return {YAML::Load(text), name + ": "};
  } catch (const YAML::Exception &e) {
    throw std::runtime_error("Could not parse " + name + ": " + e.what());
  }
}

bool Section::has(const std::string &key) const {
  return m_node.IsMap() && m_node[key].IsDefined() && !m_node[key].IsNull();
}

YAML::Node Section::child(const std::string &key) const {
  if (!m_node.IsMap()) {
    fail(m_where, "expected a map with '" + key + "'");
  }
  if (!has(key)) {
    fail(join(m_where, key), "required, but missing");
  }
  return m_node[key];
}

Section Section::section(const std::string &key) const {
  auto n = child(key);
  if (!n.IsMap()) {
    fail(join(m_where, key), "expected a map");
  }
  return {n, join(m_where, key)};
}

Section Section::optional_section(const std::string &key) const {
  if (!has(key)) {
    return {YAML::Node(YAML::NodeType::Map), join(m_where, key)};
  }
  return section(key);
}

std::string Section::string(const std::string &key) const {
  return scalar<std::string>(child(key), join(m_where, key), "string");
}

std::string Section::string_or(const std::string &key,
                               const std::string &fallback) const {
  return has(key) ? string(key) : fallback;
}

double Section::number(const std::string &key) const {
  return scalar<double>(child(key), join(m_where, key), "number");
}

double Section::number_or(const std::string &key, double fallback) const {
  return has(key) ? number(key) : fallback;
}

long long Section::integer(const std::string &key) const {
  return scalar<long long>(child(key), join(m_where, key), "integer");
}

long long Section::integer_or(const std::string &key, long long fallback) const {
  return has(key) ? integer(key) : fallback;
}

bool Section::boolean_or(const std::string &key, bool fallback) const {
  return has(key) ? scalar<bool>(child(key), join(m_where, key), "true or false")
                  : fallback;
}

std::vector<std::string> Section::names(const std::string &key) const {
  if (!has(key)) {
    return {};
  }
  const auto n = child(key);
  if (!n.IsSequence()) {
    fail(join(m_where, key), "expected a list of names");
  }
  std::vector<std::string> out;
  for (std::size_t i = 0; i < n.size(); ++i) {
    const auto name = scalar<std::string>(
        n[i], join(m_where, key) + "[" + std::to_string(i) + "]", "name");
    for (auto &e : expand_range(name)) {
      out.push_back(std::move(e));
    }
  }
  return out;
}

std::vector<Section> Section::list(const std::string &key) const {
  if (!has(key)) {
    return {};
  }
  const auto n = child(key);
  if (!n.IsSequence()) {
    fail(join(m_where, key), "expected a list");
  }
  std::vector<Section> out;
  for (std::size_t i = 0; i < n.size(); ++i) {
    out.emplace_back(n[i], join(m_where, key) + "[" + std::to_string(i) + "]");
  }
  return out;
}

std::vector<std::string> Section::keys() const {
  std::vector<std::string> out;
  if (m_node.IsMap()) {
    for (const auto &kv : m_node) {
      out.push_back(kv.first.as<std::string>());
    }
  }
  return out;
}

void Section::only(std::initializer_list<const char *> allowed) const {
  std::string unknown;
  for (const auto &k : keys()) {
    if (std::none_of(allowed.begin(), allowed.end(),
                     [&](const char *a) { return k == a; })) {
      unknown += (unknown.empty() ? "" : ", ") + k;
    }
  }
  if (!unknown.empty()) {
    std::string known;
    for (const char *a : allowed) {
      known += (known.empty() ? "" : ", ") + std::string(a);
    }
    fail(m_where, "unknown key(s) " + unknown + "; known: " + known);
  }
}

} // namespace config
} // namespace emulator
