/**
 * @file ocean_operators.cpp
 * @brief Implementation of the ocean operators.
 */

#include "ocean_operators.hpp"

#include <algorithm>
#include <stdexcept>

namespace emulator {
namespace ocn {

namespace {

model::FieldRef ref(const config::Section &s, const std::string &key) {
  return model::FieldRef::parse(s.string(key), s.where() + "." + key);
}

std::map<std::string, model::FieldRef>
refs(const config::Section &s, std::initializer_list<const char *> keys) {
  s.only(keys);
  std::map<std::string, model::FieldRef> out;
  for (const char *k : keys) {
    out[k] = ref(s, k);
  }
  return out;
}

void add_statics(model::Declarations &d, const model::FieldRef &r) {
  if (r.set() == model::FieldRef::Set::Statics) {
    d.statics.push_back(r.name());
  }
}

} // namespace

// ---------------------------------------------------------------------------
// ocean.surface_exports
// ---------------------------------------------------------------------------

OceanSurfaceExports::OceanSurfaceExports(const config::Section &o,
                                         const model::ModelInfo &)
    : m_freezing(o.number("freezing_sst")),
      m_land_salinity(o.number("land_salinity")),
      m_ocean_mask(ref(o, "ocean_mask")), m_ice_mask(ref(o, "ice_mask")) {
  o.only({"operator", "freezing_sst", "land_salinity", "ocean_mask",
          "ice_mask", "from", "to"});
  m_from = refs(o.section("from"),
                {"sst", "salinity", "u", "v", "ssh", "ice_fraction"});
  m_to = refs(o.section("to"),
              {"sst", "salinity", "u", "v", "ssh", "ice_fraction"});
}

model::Declarations OceanSurfaceExports::declarations() const {
  model::Declarations d;
  add_statics(d, m_ocean_mask);
  add_statics(d, m_ice_mask);
  for (const auto &[_, r] : m_to) {
    if (r.set() == model::FieldRef::Set::Aux) {
      d.aux.push_back(r.name());
    }
  }
  return d;
}

void OceanSurfaceExports::exports(const model::StepInfo &, model::Fields &f) {
  const auto ocean_mask = m_ocean_mask.read(f);
  const auto ice_mask = m_ice_mask.read(f);
  if (!m_checked) {
    for (const auto mask : {ocean_mask, ice_mask}) {
      for (const double v : mask) {
        if (v != 0.0 && v != 1.0) {
          throw std::runtime_error(
              "ocean.surface_exports: an ocean mask is not binary.");
        }
      }
    }
    m_checked = true;
  }
  const auto sst = m_from.at("sst").read(f);
  const auto sal = m_from.at("salinity").read(f);
  const auto u = m_from.at("u").read(f);
  const auto v = m_from.at("v").read(f);
  const auto ssh = m_from.at("ssh").read(f);
  const auto sif = m_from.at("ice_fraction").read(f);
  auto so_t = m_to.at("sst").write(f);
  auto so_s = m_to.at("salinity").write(f);
  auto so_u = m_to.at("u").write(f);
  auto so_v = m_to.at("v").write(f);
  auto so_ssh = m_to.at("ssh").write(f);
  auto ice = m_to.at("ice_fraction").write(f);
  const double tfrz = m_freezing;
  for (std::size_t i = 0; i < sst.size(); ++i) {
    const bool ocean = ocean_mask[i] == 1.0;
    so_t[i] = ocean ? std::max(sst[i], tfrz) : tfrz;
    so_s[i] = ocean ? std::clamp(sal[i], 0.0, 60.0) : m_land_salinity;
    so_u[i] = ocean ? u[i] : 0.0;
    so_v[i] = ocean ? v[i] : 0.0;
    so_ssh[i] = ocean ? ssh[i] : 0.0;
    ice[i] = ice_mask[i] == 1.0 ? std::clamp(sif[i], 0.0, 1.0) : 0.0;
  }
}

// ---------------------------------------------------------------------------
// ocean.ssh_gradients
// ---------------------------------------------------------------------------

SshGradients::SshGradients(const config::Section &o,
                           const model::ModelInfo &info)
    : m_geometry(info.geometry), m_ssh(ref(o, "ssh")), m_mask(ref(o, "mask")) {
  o.only({"operator", "ssh", "mask", "to"});
  const auto to = o.section("to");
  to.only({"dhdx", "dhdy"});
  m_dhdx = ref(to, "dhdx");
  m_dhdy = ref(to, "dhdy");
  if (m_geometry->grid == nullptr) {
    throw std::invalid_argument(o.where() +
                                ": the SSH slope needs the whole grid.");
  }
}

model::Declarations SshGradients::declarations() const {
  model::Declarations d;
  add_statics(d, m_mask);
  return d;
}

void SshGradients::exports(const model::StepInfo &, model::Fields &f) {
  const auto &gather = *m_geometry->gather;
  const bool root = gather.is_root();
  const std::size_t n = gather.num_global();
  if (!m_have_mask) {
    m_global_mask.assign(root ? n : 0, 0.0);
    gather.gather(m_mask.read(f), m_global_mask);
    m_have_mask = true;
  }
  std::vector<double> ssh_global(root ? n : 0), dx(root ? n : 0),
      dy(root ? n : 0);
  gather.gather(m_ssh.read(f), ssh_global);
  if (root) {
    const auto &g = *m_geometry->grid;
    ssh_gradients(ssh_global, g.lat, m_global_mask, g.nx, g.ny, dx, dy);
  }
  gather.scatter(dx, m_dhdx.write(f));
  gather.scatter(dy, m_dhdy.write(f));
}

// ---------------------------------------------------------------------------
// ocean.coupler_window_mean
// ---------------------------------------------------------------------------

CouplerWindowMean::CouplerWindowMean(const config::Section &o,
                                     const model::ModelInfo &info)
    : WindowMeanForcing(o, info, o.names("channels")) {
  o.only({"operator", "channels", "also_into_suffix", "clip_min_zero",
          "unweight_by_ice_fraction", "unweight_stress", "ocean_albedo"});
  m_options.unweight_by_ice_fraction =
      o.boolean_or("unweight_by_ice_fraction", m_options.unweight_by_ice_fraction);
  m_options.unweight_stress =
      o.boolean_or("unweight_stress", m_options.unweight_stress);
  m_options.ocean_albedo = o.number_or("ocean_albedo", m_options.ocean_albedo);
}

void CouplerWindowMean::fill_sample(model::Fields &f,
                                   fields::FieldSet &sample) {
  coupler_forcing_sample(*f.imports, m_options, sample);
}

void register_ocn_operators() {
  auto &r = model::OperatorRegistry::instance();
  r.add("ocean.surface_exports",
        [](const config::Section &o, const model::ModelInfo &i) {
          return std::make_unique<OceanSurfaceExports>(o, i);
        });
  r.add("ocean.ssh_gradients",
        [](const config::Section &o, const model::ModelInfo &i) {
          return std::make_unique<SshGradients>(o, i);
        });
  r.add("ocean.coupler_window_mean",
        [](const config::Section &o, const model::ModelInfo &i) {
          return std::make_unique<CouplerWindowMean>(o, i);
        });
}

} // namespace ocn
} // namespace emulator
