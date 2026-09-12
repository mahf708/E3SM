/**
 * @file ace_channels.cpp
 * @brief The ACE channel tables.
 *
 * Transcribed from emulator_comps/eatm/src/eatm_channels_mod.F90 on
 * mahf708/eocn/add-samudra, which was checked against the traced
 * checkpoints.  The order is the tensor's channel order.
 */

#include "ace_channels.hpp"

#include <stdexcept>

namespace emulator {
namespace atm {

namespace {

void levels(std::vector<std::string> &names, const std::string &stem) {
  for (int k = 0; k <= 7; ++k) {
    names.push_back(stem + std::to_string(k));
  }
}

/// What both checkpoints share: the component owns the surface fractions
/// and TS (from the coupler) even though the network predicts TS, keeps
/// PHIS from the initial condition, and computes SOLIN.
fields::ChannelLayout common(const std::string &name,
                             const std::string &water_stem) {
  fields::ChannelLayout l;
  l.name = name;
  l.model_dt = 6 * 3600;
  l.inputs = {"LANDFRAC", "OCNFRAC", "ICEFRAC", "PHIS", "SOLIN", "PS", "TS"};
  levels(l.inputs, "T_");
  levels(l.inputs, water_stem);
  levels(l.inputs, "U_");
  levels(l.inputs, "V_");
  l.outputs = {"PS", "TS"};
  levels(l.outputs, "T_");
  levels(l.outputs, water_stem);
  levels(l.outputs, "U_");
  levels(l.outputs, "V_");
  l.coupled_inputs = {"LANDFRAC", "OCNFRAC", "ICEFRAC", "TS"};
  l.boundary_inputs = {"PHIS"};
  l.forcing_inputs = {"SOLIN"};
  return l;
}

} // namespace

fields::ChannelLayout ace2_eamv3() {
  auto l = common("ACE2-EAMv3", "specific_total_water_");
  for (const char *o : {"LHFLX", "SHFLX", "surface_precipitation_rate",
                        "surface_upward_longwave_flux", "FLUT", "FLDS", "FSDS",
                        "surface_upward_shortwave_flux",
                        "top_of_atmos_upward_shortwave_flux",
                        "tendency_of_total_water_path_due_to_advection"}) {
    l.outputs.push_back(o);
    l.interval_mean_outputs.push_back(o);
  }
  l.validate();
  return l;
}

fields::ChannelLayout samudrace_e3smv3() {
  auto l = common("SamudrACE-E3SMv3", "STW_");
  for (const char *in : {"Qat2m", "Uat10m", "Vat10m", "Tat2m"}) {
    l.inputs.push_back(in);
  }
  for (const char *o : {"LHFLX", "SHFLX", "surface_precipitation_rate",
                        "frozen_precipitation_rate", "FLUS", "FLUT", "FLDS",
                        "FSDS", "FSUS", "FSUTOA", "DTENDTTW", "TAUX",
                        "TAUY"}) {
    l.outputs.push_back(o);
    l.interval_mean_outputs.push_back(o);
  }
  for (const char *o : {"Qat2m", "Uat10m", "Vat10m", "Tat2m"}) {
    l.outputs.push_back(o); // snapshots, and fed back as inputs
  }
  l.validate();
  return l;
}

std::vector<std::string> ace_layout_names() {
  return {"ACE2-EAMv3", "SamudrACE-E3SMv3"};
}

fields::ChannelLayout ace_layout(const std::string &name) {
  if (name == "ACE2-EAMv3") {
    return ace2_eamv3();
  }
  if (name == "SamudrACE-E3SMv3") {
    return samudrace_e3smv3();
  }
  std::string known;
  for (const auto &n : ace_layout_names()) {
    known += (known.empty() ? "" : ", ") + n;
  }
  throw std::invalid_argument("Unknown ACE layout '" + name +
                              "'; known: " + known + ".");
}

} // namespace atm
} // namespace emulator
