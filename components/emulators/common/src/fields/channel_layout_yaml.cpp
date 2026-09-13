/**
 * @file channel_layout_yaml.cpp
 * @brief Implementation of read_channel_layout.
 */

#include "channel_layout_yaml.hpp"

namespace emulator {
namespace fields {

ChannelLayout read_channel_layout(const config::Section &network) {
  network.only({"name", "timestep", "inputs", "outputs",
                "interval_mean_outputs", "coupled_inputs", "boundary_inputs",
                "forcing_inputs"});
  ChannelLayout l;
  l.name = network.string("name");
  l.model_dt = static_cast<int>(network.integer("timestep"));
  l.inputs = network.names("inputs");
  l.outputs = network.names("outputs");
  l.interval_mean_outputs = network.names("interval_mean_outputs");
  l.coupled_inputs = network.names("coupled_inputs");
  l.boundary_inputs = network.names("boundary_inputs");
  l.forcing_inputs = network.names("forcing_inputs");
  try {
    l.validate();
  } catch (const std::invalid_argument &e) {
    throw std::invalid_argument(network.where() + ": " + e.what());
  }
  return l;
}

} // namespace fields
} // namespace emulator
