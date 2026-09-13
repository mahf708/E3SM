/**
 * @file samudra_ocean.cpp
 * @brief Implementation of SamudraOcean.
 */

#include "samudra_ocean.hpp"

#include "samudra_channels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace emulator {
namespace ocn {

namespace {

const grid::GridField &find_field(const std::vector<grid::GridField> &ic,
                                  const std::string &name) {
  const auto it = std::find_if(ic.begin(), ic.end(),
                               [&](const auto &f) { return f.name == name; });
  if (it == ic.end()) {
    throw std::runtime_error("The ocean initial condition has no '" + name +
                             "'.");
  }
  return *it;
}

} // namespace

const std::vector<std::string> &samudra_export_names() {
  static const std::vector<std::string> names{
      "So_t", "So_s", "So_u", "So_v", "So_ssh", "So_dhdx", "So_dhdy"};
  return names;
}

SamudraOcean::SamudraOcean(
    Config config, MPI_Comm comm, const grid::HorizontalGrid &grid,
    const grid::Decomposition &decomp,
    std::shared_ptr<inference::InferenceBackend> backend)
    : m_config(std::move(config)), m_decomp(decomp), m_gather(comm, decomp),
      m_stepper(m_config.layout, comm, m_gather, grid.nx, grid.ny,
                std::move(backend)),
      m_clock(m_config.layout.model_dt, m_config.coupler_dt),
      m_brackets(m_config.layout.outputs, decomp.num_local()),
      m_window(samudra_forcing_names(), decomp.num_local()),
      m_sample(decomp.num_local()), m_blended(decomp.num_local()),
      m_nx(grid.nx), m_ny(grid.ny),
      m_ocean_mask(decomp.num_local(), 0.0), m_ice_mask(decomp.num_local(), 0.0),
      m_ice_fraction(decomp.num_local(), 0.0) {
  for (const auto &name : samudra_forcing_names()) {
    m_sample.add(name);
  }
  for (const auto &name : m_config.layout.outputs) {
    m_blended.add(name);
  }
  if (m_gather.is_root()) {
    m_global_lat = grid.lat;
  }
  if (m_config.forcing_source == ForcingSource::Atmosphere &&
      m_config.exchange == nullptr) {
    throw std::invalid_argument(
        "SamudraOcean: forcing from the atmosphere needs an exchange.");
  }
}

std::vector<std::string> SamudraOcean::initial_condition_names() const {
  std::vector<std::string> names;
  for (const auto &in : m_config.layout.inputs) {
    if (in.find(":next") == std::string::npos) {
      names.push_back(in);
    }
  }
  names.push_back("mask_2d");
  names.push_back("mask_ocean_sea_ice_fraction");
  return names;
}

void SamudraOcean::load_static(const std::vector<grid::GridField> &ic) {
  for (const auto &name : m_config.layout.inputs_from(fields::InputSource::Boundary)) {
    const auto local = m_decomp.local(find_field(ic, name).values);
    auto dest = m_stepper.inputs().get(name);
    std::copy(local.begin(), local.end(), dest.begin());
  }
  m_ocean_mask = m_decomp.local(find_field(ic, "mask_2d").values);
  if (m_gather.is_root()) {
    m_global_mask = find_field(ic, "mask_2d").values;
  }
  m_ice_mask = m_decomp.local(find_field(ic, "mask_ocean_sea_ice_fraction").values);
  for (auto *mask : {&m_ocean_mask, &m_ice_mask}) {
    for (auto &v : *mask) {
      if (v != 0.0 && v != 1.0) {
        throw std::runtime_error(
            "An ocean mask in the initial condition is not binary.");
      }
    }
  }
}

void SamudraOcean::initialize(coupling::ModelTime start,
                              const std::vector<grid::GridField> &ic) {
  (void)start;
  load_static(ic);
  auto &in = m_stepper.inputs();
  // The initial condition carries the forcing and the state; fill values
  // were zeroed when it was made, and the graph masks land itself.
  for (const auto &name : m_config.layout.inputs) {
    const auto source = m_config.layout.source(name);
    if (source == fields::InputSource::Boundary) {
      continue;
    }
    const auto base = name.substr(0, name.find(":next"));
    const auto &field = find_field(ic, base);
    if (field.unusable() > 0) {
      throw std::runtime_error("The ocean initial condition's '" + base +
                               "' has unusable values.");
    }
    const auto local = m_decomp.local(field.values);
    auto dest = in.get(name);
    std::copy(local.begin(), local.end(), dest.begin());
  }

  fields::FieldSet state(m_decomp.num_local());
  for (const auto &name : m_config.layout.outputs) {
    const auto from = in.get(name);
    auto dest = state.add(name);
    std::copy(from.begin(), from.end(), dest.begin());
  }
  m_brackets.set_both(state);
  m_started = true;
}

void SamudraOcean::run(coupling::ModelTime now,
                       const fields::FieldSet &imports,
                       fields::FieldSet &exports) {
  if (!m_started) {
    throw std::logic_error("SamudraOcean::run before initialize or restart.");
  }
  const auto step = m_clock.on_coupler_step(now);
  if (step.first_call) {
    if (m_config.forcing_source == ForcingSource::Atmosphere) {
      for (const auto &name : samudra_forcing_names()) {
        const auto from = m_config.exchange->get("atm." + name);
        auto to = m_sample.get(name);
        std::copy(from.begin(), from.end(), to.begin());
      }
    } else {
      coupler_forcing_sample(imports, m_config.forcing, m_sample);
    }
    m_window.add(m_sample);
    if (step.advance) {
      auto &in = m_stepper.inputs();
      for (const auto &name : samudra_forcing_names()) {
        m_window.mean(name, in.get(name));
      }
      for (const auto &name : samudra_forcing_names()) {
        // clip_after_mean on the stepper's own channels
        if (name == "surface_precipitation_rate" ||
            name == "frozen_precipitation_rate") {
          for (auto &v : in.get(name)) {
            v = std::max(v, 0.0);
          }
        }
        const auto from = in.get(name);
        auto next = in.get(name + ":next");
        std::copy(from.begin(), from.end(), next.begin());
      }
      m_stepper.step(step.completed_steps);
      m_brackets.advance(m_stepper.prediction());
      m_window.reset();
    }
  }
  compute_exports(exports);
}

void SamudraOcean::initial_exports(fields::FieldSet &exports) {
  compute_exports(exports);
}

void SamudraOcean::compute_exports(fields::FieldSet &exports) {
  // The latest state, held through the window.
  m_brackets.blend(1.0, m_blended);
  const auto sst = m_blended.get("sst");
  const auto sal = m_blended.get("salinityCoarsened_0");
  const auto u = m_blended.get("velocityZonalCoarsened_0");
  const auto v = m_blended.get("velocityMeridionalCoarsened_0");
  const auto ssh = m_blended.get("ssh");
  const auto sif = m_blended.get("ocean_sea_ice_fraction");
  auto so_t = exports.get("So_t"), so_s = exports.get("So_s");
  auto so_u = exports.get("So_u"), so_v = exports.get("So_v");
  auto so_ssh = exports.get("So_ssh");
  const double tfrz = m_config.freezing_sst;
  for (std::size_t i = 0; i < sst.size(); ++i) {
    const bool ocean = m_ocean_mask[i] == 1.0;
    so_t[i] = ocean ? std::max(sst[i], tfrz) : tfrz;
    so_s[i] = ocean ? std::clamp(sal[i], 0.0, 60.0) : m_config.land_salinity;
    so_u[i] = ocean ? u[i] : 0.0;
    so_v[i] = ocean ? v[i] : 0.0;
    so_ssh[i] = ocean ? ssh[i] : 0.0;
    // Bounded by the ice channel's own mask: on the ocean mask it put sea
    // ice in the tropics.
    m_ice_fraction[i] = m_ice_mask[i] == 1.0 ? std::clamp(sif[i], 0.0, 1.0) : 0.0;
  }

  // The slope needs neighbouring rows, which can be on another rank: take
  // it on the whole grid at the root, from the exported So_ssh.  Collective.
  const bool root = m_gather.is_root();
  const std::size_t n = m_gather.num_global();
  std::vector<double> ssh_global(root ? n : 0), dx(root ? n : 0),
      dy(root ? n : 0);
  m_gather.gather(so_ssh, ssh_global);
  if (root) {
    ssh_gradients(ssh_global, m_global_lat, m_global_mask, m_nx, m_ny, dx, dy);
  }
  m_gather.scatter(dx, exports.get("So_dhdx"));
  m_gather.scatter(dy, exports.get("So_dhdy"));

  if (m_config.exchange != nullptr) {
    m_config.exchange->publish("ocn.sst", so_t);
    m_config.exchange->publish("ocn.sea_ice_fraction", m_ice_fraction);
  }
}

void SamudraOcean::save_to(coupling::RestartStore &store) const {
  m_clock.save_to(store, "ocn.clock");
  m_brackets.save_to(store, "ocn.state");
  m_window.save_to(store, "ocn.forcing");
}

void SamudraOcean::restart(coupling::RestartStore &store,
                           const std::vector<grid::GridField> &ic) {
  load_static(ic);
  m_clock.load_from(store, "ocn.clock");
  m_brackets.load_from(store, "ocn.state");
  m_window.load_from(store, "ocn.forcing");
  for (const auto &name :
       m_config.layout.inputs_from(fields::InputSource::Prognostic)) {
    const auto from = m_brackets.upper(name);
    auto to = m_stepper.inputs().get(name);
    std::copy(from.begin(), from.end(), to.begin());
  }
  m_started = true;
}

} // namespace ocn
} // namespace emulator
