/**
 * @file samudra_channels.cpp
 * @brief The Samudra channel table, transcribed from
 *        emulator_comps/eocn/src/eocn_channels_mod.F90 on
 *        mahf708/eocn/add-samudra.
 */

#include "samudra_channels.hpp"

namespace emulator {
namespace ocn {

const std::vector<std::string> &samudra_forcing_names() {
  static const std::vector<std::string> names{
      "TAUX", "TAUY", "surface_precipitation_rate", "frozen_precipitation_rate",
      "FLUS", "FSUS", "FLDS", "FSDS", "LHFLX", "SHFLX"};
  return names;
}

fields::ChannelLayout samudra_e3smv3() {
  fields::ChannelLayout l;
  l.name = "SamudrACE-E3SMv3-ocean";
  l.model_dt = 5 * 86400;

  l.outputs = {"sst", "ssh"};
  for (const char *stem : {"salinityCoarsened_", "temperatureCoarsened_",
                           "velocityZonalCoarsened_",
                           "velocityMeridionalCoarsened_"}) {
    for (int k = 0; k <= 18; ++k) {
      l.outputs.push_back(stem + std::to_string(k));
    }
  }
  l.outputs.push_back("ocean_sea_ice_fraction");
  l.outputs.push_back("iceVolumeTotal");

  l.inputs = {"LANDFRAC", "sea_surface_fraction"};
  l.boundary_inputs = l.inputs;
  for (const auto &f : samudra_forcing_names()) {
    l.inputs.push_back(f);
    l.forcing_inputs.push_back(f);
  }
  l.inputs.insert(l.inputs.end(), l.outputs.begin(), l.outputs.end());
  for (const auto &f : samudra_forcing_names()) {
    l.inputs.push_back(f + ":next");
    l.forcing_inputs.push_back(f + ":next");
  }
  l.validate();
  return l;
}

} // namespace ocn
} // namespace emulator
