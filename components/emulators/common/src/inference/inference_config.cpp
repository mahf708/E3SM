/**
 * @file inference_config.cpp
 * @brief Implementation of InferenceConfig.
 */

#include "inference_config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace emulator {
namespace inference {

namespace {

std::string trim(const std::string &s) {
  const char *ws = " \t\r\n";
  const auto b = s.find_first_not_of(ws);
  if (b == std::string::npos) {
    return std::string();
  }
  const auto e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

bool parse_bool(const std::string &raw, const std::string &key) {
  const std::string v = to_lower(trim(raw));
  if (v == "true" || v == "yes" || v == "on" || v == "1" || v == ".true.") {
    return true;
  }
  if (v == "false" || v == "no" || v == "off" || v == "0" || v == ".false.") {
    return false;
  }
  EMULATOR_INFER_REQUIRE(false, "Cannot interpret '" << raw << "' for option '"
                                                     << key
                                                     << "' as a boolean.");
  return false; // unreachable
}

/// Apply one key/value pair to a config; `key` is already trimmed.
void apply_setting(InferenceConfig &cfg, const std::string &key,
                   const std::string &value) {
  if (key == "backend") {
    cfg.backend = value;
  } else if (key == "model_path" || key == "model") {
    cfg.model_path = value;
  } else if (key == "input") {
    cfg.inputs.push_back(TensorSpec::parse(value));
  } else if (key == "output") {
    cfg.outputs.push_back(TensorSpec::parse(value));
  } else if (key == "input_channels") {
    cfg.input_channels = std::stoi(value);
  } else if (key == "output_channels") {
    cfg.output_channels = std::stoi(value);
  } else if (key == "verbose") {
    cfg.verbose = parse_bool(value, key);
  } else if (key.rfind("option.", 0) == 0) {
    cfg.options[key.substr(7)] = value;
  } else {
    cfg.options[key] = value;
  }
}

InferenceConfig parse_lines(const std::string &text, const std::string *prefix) {
  InferenceConfig cfg;
  std::istringstream iss(text);
  std::string line;
  int lineno = 0;

  while (std::getline(iss, line)) {
    ++lineno;
    const std::string stripped = trim(line);
    if (stripped.empty() || stripped[0] == '#' || stripped[0] == '!') {
      continue;
    }

    const auto colon = stripped.find(':');
    EMULATOR_INFER_REQUIRE(colon != std::string::npos,
                           "Malformed inference config line " << lineno << ": '"
                                                             << stripped
                                                             << "' (expected "
                                                                "'key: value')"
                                                                ".");

    std::string key = trim(stripped.substr(0, colon));
    std::string value = trim(stripped.substr(colon + 1));

    // Drop trailing inline comments, but keep '#' inside quotes.
    if (!value.empty() && value.front() != '"' && value.front() != '\'') {
      const auto hash = value.find('#');
      if (hash != std::string::npos) {
        value = trim(value.substr(0, hash));
      }
    } else if (value.size() >= 2 &&
               ((value.front() == '"' && value.back() == '"') ||
                (value.front() == '\'' && value.back() == '\''))) {
      value = value.substr(1, value.size() - 2);
    }

    if (prefix != nullptr) {
      if (key.rfind(*prefix, 0) != 0) {
        continue; // not an inference setting
      }
      key = key.substr(prefix->size());
      if (key.empty()) {
        continue;
      }
    }

    EMULATOR_INFER_REQUIRE(!key.empty(), "Malformed inference config line "
                                             << lineno
                                             << ": empty key in '" << stripped
                                             << "'.");
    try {
      apply_setting(cfg, key, value);
    } catch (const InferenceError &) {
      throw;
    } catch (const std::exception &e) {
      EMULATOR_INFER_REQUIRE(false, "Bad value for '"
                                        << key << "' on line " << lineno
                                        << " ('" << value
                                        << "'): " << e.what());
    }
  }
  return cfg;
}

std::string read_file(const std::string &path) {
  std::ifstream ifs(path);
  EMULATOR_INFER_REQUIRE(ifs.good(), "Cannot open inference config file '"
                                         << path << "'.");
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

} // namespace

bool InferenceConfig::has(const std::string &key) const {
  return options.find(key) != options.end();
}

std::string InferenceConfig::get(const std::string &key,
                                 const std::string &fallback) const {
  const auto it = options.find(key);
  return it == options.end() ? fallback : it->second;
}

int InferenceConfig::get_int(const std::string &key, int fallback) const {
  const auto it = options.find(key);
  if (it == options.end()) {
    return fallback;
  }
  try {
    return std::stoi(it->second);
  } catch (const std::exception &) {
    EMULATOR_INFER_REQUIRE(false, "Option '" << key << "' = '" << it->second
                                             << "' is not an integer.");
  }
  return fallback; // unreachable
}

double InferenceConfig::get_double(const std::string &key,
                                   double fallback) const {
  const auto it = options.find(key);
  if (it == options.end()) {
    return fallback;
  }
  try {
    return std::stod(it->second);
  } catch (const std::exception &) {
    EMULATOR_INFER_REQUIRE(false, "Option '" << key << "' = '" << it->second
                                             << "' is not a number.");
  }
  return fallback; // unreachable
}

bool InferenceConfig::get_bool(const std::string &key, bool fallback) const {
  const auto it = options.find(key);
  return it == options.end() ? fallback : parse_bool(it->second, key);
}

std::string InferenceConfig::get_required(const std::string &key,
                                          const std::string &context) const {
  const auto it = options.find(key);
  EMULATOR_INFER_REQUIRE(it != options.end() && !it->second.empty(),
                         "Missing required option '"
                             << key << "'"
                             << (context.empty() ? "" : " for " + context)
                             << ".");
  return it->second;
}

InferenceConfig &InferenceConfig::set(const std::string &key,
                                      const std::string &value) {
  options[key] = value;
  return *this;
}

InferenceConfig &InferenceConfig::apply(const std::string &key,
                                       const std::string &value) {
  const std::string trimmed_key = trim(key);
  EMULATOR_INFER_REQUIRE(!trimmed_key.empty(),
                         "Cannot apply an inference setting with an empty "
                         "key.");
  try {
    apply_setting(*this, trimmed_key, trim(value));
  } catch (const InferenceError &) {
    throw;
  } catch (const std::exception &e) {
    EMULATOR_INFER_REQUIRE(false, "Bad value for '"
                                      << trimmed_key << "' ('" << value
                                      << "'): " << e.what());
  }
  return *this;
}

InferenceConfig InferenceConfig::from_string(const std::string &text) {
  return parse_lines(text, nullptr);
}

InferenceConfig InferenceConfig::from_file(const std::string &path) {
  return from_string(read_file(path));
}

InferenceConfig
InferenceConfig::from_string_with_prefix(const std::string &text,
                                         const std::string &prefix) {
  return parse_lines(text, &prefix);
}

InferenceConfig
InferenceConfig::from_file_with_prefix(const std::string &path,
                                       const std::string &prefix) {
  return from_string_with_prefix(read_file(path), prefix);
}

std::string InferenceConfig::to_string() const {
  std::ostringstream oss;
  oss << "backend        : " << backend << "\n";
  oss << "model_path     : " << (model_path.empty() ? "<none>" : model_path)
      << "\n";
  oss << "verbose        : " << std::boolalpha << verbose << "\n";
  if (input_channels > 0 || output_channels > 0) {
    oss << "channels       : in=" << input_channels
        << " out=" << output_channels << "\n";
  }
  oss << "inputs         : ";
  if (inputs.empty()) {
    oss << "<unspecified>";
  } else {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      oss << (i ? ", " : "") << inputs[i].to_string();
    }
  }
  oss << "\n";
  oss << "outputs        : ";
  if (outputs.empty()) {
    oss << "<unspecified>";
  } else {
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      oss << (i ? ", " : "") << outputs[i].to_string();
    }
  }
  oss << "\n";
  if (!options.empty()) {
    oss << "options        :\n";
    for (const auto &kv : options) {
      oss << "  " << kv.first << " = " << kv.second << "\n";
    }
  }
  return oss.str();
}

} // namespace inference
} // namespace emulator
