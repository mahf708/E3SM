/**
 * @file channel_layout.cpp
 * @brief Implementation of ChannelLayout.
 */

#include "channel_layout.hpp"

#include <algorithm>
#include <stdexcept>

namespace emulator {
namespace fields {

namespace {

bool has(const std::vector<std::string> &names, std::string_view name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

void unique(const std::string &layout, const std::vector<std::string> &names,
            const char *what) {
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty()) {
      throw std::invalid_argument("Layout '" + layout + "': " + what +
                                  " channel " + std::to_string(i) +
                                  " has no name.");
    }
    if (std::find(names.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                  names.end(), names[i]) != names.end()) {
      throw std::invalid_argument("Layout '" + layout + "': " + what +
                                  " channel '" + names[i] + "' appears twice.");
    }
  }
}

void subset(const std::string &layout, const std::vector<std::string> &names,
            const std::vector<std::string> &of, const char *what,
            const char *where) {
  for (const auto &n : names) {
    if (!has(of, n)) {
      throw std::invalid_argument("Layout '" + layout + "': " + what + " '" +
                                  n + "' is not one of the " + where + ".");
    }
  }
}

} // namespace

const char *to_string(InputSource source) {
  switch (source) {
  case InputSource::Prognostic:
    return "prognostic";
  case InputSource::Coupled:
    return "coupled";
  case InputSource::Boundary:
    return "boundary";
  case InputSource::Forcing:
    return "forcing";
  }
  return "unknown";
}

void ChannelLayout::validate() const {
  if (model_dt <= 0) {
    throw std::invalid_argument("Layout '" + name +
                                "': model_dt must be positive.");
  }
  unique(name, inputs, "input");
  unique(name, outputs, "output");
  subset(name, coupled_inputs, inputs, "coupled input", "inputs");
  subset(name, boundary_inputs, inputs, "boundary input", "inputs");
  subset(name, forcing_inputs, inputs, "forcing input", "inputs");
  subset(name, interval_mean_outputs, outputs, "interval-mean output",
         "outputs");
  for (const auto &in : inputs) {
    const int roles = static_cast<int>(has(coupled_inputs, in)) +
                      static_cast<int>(has(boundary_inputs, in)) +
                      static_cast<int>(has(forcing_inputs, in));
    if (roles > 1) {
      throw std::invalid_argument("Layout '" + name + "': input '" + in +
                                  "' is given more than one source.");
    }
    if (roles == 0 && !has(outputs, in)) {
      throw std::invalid_argument(
          "Layout '" + name + "': input '" + in +
          "' has no source. It is not an output, so it cannot be carried "
          "forward, and it is not listed as coupled, boundary or forcing; "
          "the network would read a channel nothing ever sets.");
    }
  }
}

InputSource ChannelLayout::source(std::string_view input) const {
  if (!has(inputs, input)) {
    throw std::out_of_range("Layout '" + name + "' has no input '" +
                            std::string(input) + "'.");
  }
  if (has(coupled_inputs, input)) {
    return InputSource::Coupled;
  }
  if (has(boundary_inputs, input)) {
    return InputSource::Boundary;
  }
  if (has(forcing_inputs, input)) {
    return InputSource::Forcing;
  }
  return InputSource::Prognostic;
}

coupling::BracketedState::Temporal
ChannelLayout::temporal(std::string_view output) const {
  if (!has(outputs, output)) {
    throw std::out_of_range("Layout '" + name + "' has no output '" +
                            std::string(output) + "'.");
  }
  return has(interval_mean_outputs, output)
             ? coupling::BracketedState::Temporal::IntervalMean
             : coupling::BracketedState::Temporal::Snapshot;
}

std::vector<std::string> ChannelLayout::inputs_from(InputSource s) const {
  std::vector<std::string> out;
  for (const auto &in : inputs) {
    if (source(in) == s) {
      out.push_back(in);
    }
  }
  return out;
}

std::vector<coupling::BracketedState::Temporal>
ChannelLayout::output_temporals() const {
  std::vector<coupling::BracketedState::Temporal> out;
  for (const auto &o : outputs) {
    out.push_back(temporal(o));
  }
  return out;
}

} // namespace fields
} // namespace emulator
