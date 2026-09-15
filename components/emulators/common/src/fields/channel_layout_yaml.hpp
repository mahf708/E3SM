/**
 * @file channel_layout_yaml.hpp
 * @brief A network's channel contract, from a spec's `network` section.
 */

#ifndef E3SM_EMULATOR_FIELDS_CHANNEL_LAYOUT_YAML_HPP
#define E3SM_EMULATOR_FIELDS_CHANNEL_LAYOUT_YAML_HPP

#include "channel_layout.hpp"
#include "yaml_config.hpp"

namespace emulator {
namespace fields {

/**
 * @brief Read and validate a layout:
 *
 * ```yaml
 * network:
 *   name: my-network
 *   timestep: 21600                     # seconds
 *   inputs:  [LANDFRAC, PS, "T_{0..7}"] # tensor order
 *   outputs: [PS, "T_{0..7}", LHFLX]
 *   interval_mean_outputs: [LHFLX]      # the rest are snapshots
 *   coupled_inputs:  [LANDFRAC]         # not carried from same-named output
 *   boundary_inputs: []                 # boundary_inputs / forcing_inputs: []
 * ```
 */
ChannelLayout read_channel_layout(const config::Section &network);

} // namespace fields
} // namespace emulator

#endif // E3SM_EMULATOR_FIELDS_CHANNEL_LAYOUT_YAML_HPP
