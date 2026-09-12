/**
 * @file samudra_channels.hpp
 * @brief The Samudra ocean checkpoint's channel layout.
 */

#ifndef EMULATOROCN_SAMUDRA_CHANNELS_HPP
#define EMULATOROCN_SAMUDRA_CHANNELS_HPP

#include <string>
#include <vector>

#include "channel_layout.hpp"

namespace emulator {
namespace ocn {

/// The ten atmospheric forcing channels, in tensor order.
const std::vector<std::string> &samudra_forcing_names();

/**
 * @brief The ocean of SamudrACE-E3SMv3: 102 inputs, 80 outputs, 5-day step.
 *
 * Inputs are two static fraction channels, the ten forcing channels, the 80
 * state channels (the outputs, in output order), and a trailing copy of the
 * ten forcing channels that the traced graph slices but does not read.  The
 * copies need distinct names here and are called `<name>:next`.
 *
 * Masking of land cells to each channel's training mean, and normalization,
 * happen inside the traced graph.  All outputs are carried as snapshots, as
 * the Fortran emulator did.
 */
fields::ChannelLayout samudra_e3smv3();

} // namespace ocn
} // namespace emulator

#endif // EMULATOROCN_SAMUDRA_CHANNELS_HPP
