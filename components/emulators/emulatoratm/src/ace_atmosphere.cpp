/**
 * @file ace_atmosphere.cpp
 * @brief Implementation of AceAtmosphere.
 */

#include "ace_atmosphere.hpp"

#include "ace_surface_inputs.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace emulator {
namespace atm {

namespace {

bool has_output(const fields::ChannelLayout &l, const std::string &name) {
  return std::find(l.outputs.begin(), l.outputs.end(), name) != l.outputs.end();
}

/// The first of `names` the layout outputs; the two ACE tables spell some
/// channels differently.
std::string output_named(const fields::ChannelLayout &l,
                         std::initializer_list<const char *> names) {
  for (const char *n : names) {
    if (has_output(l, n)) {
      return n;
    }
  }
  return {};
}

std::span<const double> maybe(const fields::FieldSet &f,
                              const std::string &name) {
  if (name.empty() || !f.contains(name)) {
    return {};
  }
  return f.get(name);
}

} // namespace

const std::vector<std::string> &ocean_forcing_channels() {
  static const std::vector<std::string> names{
      "TAUX", "TAUY", "surface_precipitation_rate", "frozen_precipitation_rate",
      "FLUS", "FSUS", "FLDS", "FSDS", "LHFLX", "SHFLX"};
  return names;
}

const std::vector<std::string> &ace_import_names() {
  static const std::vector<std::string> names{"Sf_lfrac", "Sf_ofrac",
                                              "Sf_ifrac", "Sx_t"};
  return names;
}

const std::vector<std::string> &ace_export_names() {
  static const std::vector<std::string> names{
      "Sa_z",       "Sa_u",       "Sa_v",       "Sa_tbot",    "Sa_ptem",
      "Sa_shum",    "Sa_pbot",    "Sa_pslv",    "Sa_dens",    "Sa_topo",
      "Faxa_lwdn",  "Faxa_rainc", "Faxa_rainl", "Faxa_snowc", "Faxa_snowl",
      "Faxa_swndr", "Faxa_swvdr", "Faxa_swndf", "Faxa_swvdf", "Faxa_swnet"};
  return names;
}

AceAtmosphere::AceAtmosphere(
    Config config, MPI_Comm comm, const grid::HorizontalGrid &grid,
    const grid::Decomposition &decomp,
    std::shared_ptr<inference::InferenceBackend> backend)
    : m_config(std::move(config)), m_comm(comm), m_decomp(decomp),
      m_lat(decomp.local(grid.lat)), m_lon(decomp.local(grid.lon)),
      m_gather(comm, decomp),
      m_stepper(m_config.layout, comm, m_gather, grid.nx, grid.ny,
                std::move(backend)),
      m_clock(m_config.layout.model_dt, m_config.coupler_dt),
      m_brackets(m_config.layout.outputs, m_config.layout.output_temporals(),
                 decomp.num_local()),
      m_blended(decomp.num_local()), m_sun(m_config.orbit, m_lat, m_lon),
      m_solin_window(decomp.num_local(), 0.0),
      m_solin_now(decomp.num_local(), 0.0) {
  for (const auto &name : m_config.layout.outputs) {
    m_blended.add(name);
  }
  if ((m_config.publish_ocean_forcing || m_config.surface_from_ocean) &&
      m_config.exchange == nullptr) {
    throw std::invalid_argument(
        "AceAtmosphere: exchanging with an emulated ocean needs an exchange.");
  }
  if (m_config.publish_ocean_forcing) {
    for (const auto &name : ocean_forcing_channels()) {
      if (!has_output(m_config.layout, name)) {
        throw std::invalid_argument(
            "AceAtmosphere: layout '" + m_config.layout.name +
            "' has no '" + name + "' output, so it cannot force an emulated "
            "ocean. SamudrACE-E3SMv3 has all ten.");
      }
    }
  }
}

void AceAtmosphere::set_boundary_and_initial(
    const std::vector<grid::GridField> &ic, bool prognostic_too) {
  const auto &layout = m_config.layout;
  for (const auto &name : layout.inputs) {
    const auto source = layout.source(name);
    const bool wanted = source == fields::InputSource::Boundary ||
                        (prognostic_too && source != fields::InputSource::Forcing);
    if (!wanted) {
      continue;
    }
    const auto it = std::find_if(ic.begin(), ic.end(),
                                 [&](const auto &f) { return f.name == name; });
    if (it == ic.end()) {
      throw std::runtime_error("The ACE initial condition has no '" + name +
                               "'.");
    }
    auto values = it->values;
    const bool fraction = name.find("FRAC") != std::string::npos;
    if (it->unusable() > 0) {
      if (!fraction) {
        throw std::runtime_error(
            "The ACE initial condition's '" + name + "' has " +
            std::to_string(it->non_finite) + " non-finite and " +
            std::to_string(it->fill_like) +
            " fill-like values. Only the surface fractions may be "
            "zero-filled; anything else would poison the whole network.");
      }
      for (auto &v : values) {
        if (!std::isfinite(v) || std::abs(v) >= 1e30) {
          v = 0.0;
        }
      }
    }
    const auto local = m_decomp.local(values);
    auto dest = m_stepper.inputs().get(name);
    std::copy(local.begin(), local.end(), dest.begin());
  }
}

void AceAtmosphere::initialize(
    coupling::ModelTime start,
    const std::vector<grid::GridField> &initial_condition) {
  const auto &layout = m_config.layout;
  set_boundary_and_initial(initial_condition, /*prognostic_too=*/true);

  // The first step is (start, start + dt]; SOLIN is its mean.
  auto solin = m_stepper.inputs().get("SOLIN");
  m_sun.window_mean(start.ymd, start.tod, layout.model_dt, solin);
  std::copy(solin.begin(), solin.end(), m_solin_window.begin());

  // The lower bracket is the initial state wherever an output has an input
  // of the same name, captured before the step overwrites the prognostic
  // inputs; the flux channels have no earlier value, so both brackets hold
  // the first prediction there.
  fields::FieldSet lower(m_decomp.num_local());
  for (const auto &name : layout.outputs) {
    auto dest = lower.add(name);
    if (m_stepper.inputs().contains(name)) {
      const auto from = m_stepper.inputs().get(name);
      std::copy(from.begin(), from.end(), dest.begin());
    }
  }
  m_stepper.step(0);
  for (const auto &name : layout.outputs) {
    if (!m_stepper.inputs().contains(name)) {
      const auto from = m_stepper.prediction().get(name);
      auto dest = lower.get(name);
      std::copy(from.begin(), from.end(), dest.begin());
    }
  }
  m_brackets.set_both(lower);
  m_brackets.advance(m_stepper.prediction());
  m_started = true;
}

void AceAtmosphere::run(coupling::ModelTime now,
                        const fields::FieldSet &imports,
                        fields::FieldSet &exports) {
  if (!m_started) {
    throw std::logic_error("AceAtmosphere::run before initialize or restart.");
  }
  const auto &layout = m_config.layout;
  const auto step = m_clock.on_coupler_step(now);

  if (step.first_call && step.advance) {
    auto &in = m_stepper.inputs();
    SurfaceChannels surface{in.get("LANDFRAC"), in.get("OCNFRAC"),
                            in.get("ICEFRAC"), in.get("TS")};
    SurfaceCouplerInputs from_coupler{
        imports.get("Sf_lfrac"), imports.get("Sf_ofrac"),
        imports.get("Sf_ifrac"), imports.get("Sx_t"), m_brackets.upper("TS")};
    if (m_config.surface_from_ocean) {
      from_coupler.ocean_ice_fraction =
          m_config.exchange->get("ocn.sea_ice_fraction");
      from_coupler.ocean_sst = m_config.exchange->get("ocn.sst");
    }
    compute_surface_inputs(from_coupler, surface);
    auto solin = in.get("SOLIN");
    m_sun.window_mean(now.ymd, now.tod, layout.model_dt, solin);
    std::copy(solin.begin(), solin.end(), m_solin_window.begin());

    m_stepper.step(step.completed_steps);
    m_brackets.advance(m_stepper.prediction());
  }
  compute_exports(now, step.fraction, exports);
}

void AceAtmosphere::initial_exports(coupling::ModelTime start,
                                    fields::FieldSet &exports) {
  if (!m_started) {
    throw std::logic_error("AceAtmosphere::initial_exports before initialize.");
  }
  compute_exports(start, 0.0, exports);
}

void AceAtmosphere::compute_exports(coupling::ModelTime now, double fraction,
                                    fields::FieldSet &exports) {
  const auto &l = m_config.layout;
  m_brackets.blend(fraction, m_blended);
  m_sun.instantaneous(now.ymd, now.tod, m_solin_now);

  const auto &b = m_blended;
  SurfaceInputs in;
  in.ps = b.get("PS");
  in.phis = m_stepper.inputs().get("PHIS");
  in.t_lowest = b.get("T_7");
  in.q_lowest = b.get(output_named(l, {"STW_7", "specific_total_water_7"}));
  in.u_lowest = b.get("U_7");
  in.v_lowest = b.get("V_7");
  in.t_2m = maybe(b, output_named(l, {"Tat2m"}));
  in.q_2m = maybe(b, output_named(l, {"Qat2m"}));
  in.u_10m = maybe(b, output_named(l, {"Uat10m"}));
  in.v_10m = maybe(b, output_named(l, {"Vat10m"}));
  in.flds = b.get("FLDS");
  in.fsds = b.get("FSDS");
  in.fsus = maybe(b, output_named(l, {"FSUS", "surface_upward_shortwave_flux"}));
  in.precip = b.get("surface_precipitation_rate");
  in.frozen_precip = maybe(b, output_named(l, {"frozen_precipitation_rate"}));
  in.solin_now = m_solin_now;
  in.solin_window = m_solin_window;

  SurfaceExports out{exports.get("Sa_z"),       exports.get("Sa_u"),
                     exports.get("Sa_v"),       exports.get("Sa_tbot"),
                     exports.get("Sa_ptem"),    exports.get("Sa_shum"),
                     exports.get("Sa_pbot"),    exports.get("Sa_pslv"),
                     exports.get("Sa_dens"),    exports.get("Sa_topo"),
                     exports.get("Faxa_lwdn"),  exports.get("Faxa_rainc"),
                     exports.get("Faxa_rainl"), exports.get("Faxa_snowc"),
                     exports.get("Faxa_snowl"), exports.get("Faxa_swndr"),
                     exports.get("Faxa_swvdr"), exports.get("Faxa_swndf"),
                     exports.get("Faxa_swvdf"), exports.get("Faxa_swnet")};
  compute_surface_exports(in, m_config.surface, out);

  if (m_config.publish_ocean_forcing) {
    std::vector<double> buffer(m_decomp.num_local());
    for (const auto &name : ocean_forcing_channels()) {
      const auto held = b.get(name);
      std::copy(held.begin(), held.end(), buffer.begin());
      if (name == "surface_precipitation_rate" ||
          name == "frozen_precipitation_rate") {
        const double scale = name == "frozen_precipitation_rate" &&
                                     m_config.surface.frozen_precip_in_m_per_s
                                 ? constants::rhofw
                                 : 1.0;
        for (auto &v : buffer) {
          v = std::max(v, 0.0) * scale;
        }
      }
      m_config.exchange->publish("atm." + name, buffer);
    }
  }
}

void AceAtmosphere::save_to(coupling::RestartStore &store) const {
  m_clock.save_to(store, "atm.clock");
  m_brackets.save_to(store, "atm.state");
  store.write_array("atm.solin_window", m_solin_window);
}

void AceAtmosphere::restart(
    coupling::RestartStore &store,
    const std::vector<grid::GridField> &initial_condition) {
  m_clock.load_from(store, "atm.clock");
  m_brackets.load_from(store, "atm.state");
  if (!store.read_array("atm.solin_window", m_solin_window)) {
    throw std::runtime_error(
        "The atmosphere restart has no 'atm.solin_window'; the held "
        "shortwave cannot be put back on the diurnal cycle without it.");
  }
  set_boundary_and_initial(initial_condition, /*prognostic_too=*/false);
  // The network's next inputs are the upper bracket: the raw prediction.
  for (const auto &name :
       m_config.layout.inputs_from(fields::InputSource::Prognostic)) {
    const auto from = m_brackets.upper(name);
    auto to = m_stepper.inputs().get(name);
    std::copy(from.begin(), from.end(), to.begin());
  }
  m_started = true;
}

} // namespace atm
} // namespace emulator
