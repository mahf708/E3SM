/**
 * @file ace_channels.hpp
 * @brief Channel layouts of the ACE-family atmosphere checkpoints.
 */

#ifndef EMULATORATM_ACE_CHANNELS_HPP
#define EMULATORATM_ACE_CHANNELS_HPP

#include <string>
#include <vector>

#include "channel_layout.hpp"

namespace emulator {
namespace atm {

/// ACE2-EAMv3: deterministic, 39 inputs, 44 outputs, 6 h step.
fields::ChannelLayout ace2_eamv3();

/// The atmosphere of SamudrACE-E3SMv3: stochastic, 43 inputs, 51 outputs.
fields::ChannelLayout samudrace_e3smv3();

/// By name, as it appears in configuration.
/// @throws std::invalid_argument listing the known names
fields::ChannelLayout ace_layout(const std::string &name);

std::vector<std::string> ace_layout_names();

} // namespace atm
} // namespace emulator

#endif // EMULATORATM_ACE_CHANNELS_HPP
