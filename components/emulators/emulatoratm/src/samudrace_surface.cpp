/**
 * @file samudrace_surface.cpp
 * @brief Implementation of SamudrACE's ocean-to-atmosphere exchange.  The
 *        tests check it against fme's own classes on the same cells.
 */

#include "samudrace_surface.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace emulator {
namespace atm {

namespace {

void need(std::size_t size, std::size_t n, const char *what) {
  if (size != n) {
    throw std::invalid_argument(std::string("samudrace surface: ") + what +
                                " has " + std::to_string(size) +
                                " values for " + std::to_string(n) + " cells.");
  }
}

model::FieldRef ref(const config::Section &s, const std::string &key) {
  return model::FieldRef::parse(s.string(key), s.where() + "." + key);
}

} // namespace

void samudrace_fractions(const OceanToAtmosphereFields &in,
                         std::span<double> ocnfrac,
                         std::span<double> icefrac) {
  const auto n = in.land_fraction.size();
  need(in.ocean_ice_fraction.size(), n, "the ocean's ice fraction");
  need(in.ocean_mask.size(), n, "the ocean mask");
  need(in.ice_mask.size(), n, "the ice mask");
  need(ocnfrac.size(), n, "OCNFRAC");
  need(icefrac.size(), n, "ICEFRAC");
  for (std::size_t i = 0; i < n; ++i) {
    const double s = std::isnan(in.ocean_ice_fraction[i])
                         ? 0.0
                         : in.ocean_ice_fraction[i];
    const double land = in.land_fraction[i];
    const double ice = s * (1.0 - land);
    const double ocean = std::max(1.0 - land - ice, 0.0);
    icefrac[i] = in.ice_mask[i] != 0.0 ? ice : 0.0;
    ocnfrac[i] = in.ocean_mask[i] != 0.0 ? ocean : 0.0;
  }
}

void samudrace_prescribe_ts(std::span<const double> ocnfrac,
                            std::span<const double> sst,
                            std::span<const double> ocean_mask,
                            std::span<double> ts) {
  const auto n = ts.size();
  need(ocnfrac.size(), n, "OCNFRAC");
  need(sst.size(), n, "SST");
  need(ocean_mask.size(), n, "the ocean mask");
  for (std::size_t i = 0; i < n; ++i) {
    const double target = ocean_mask[i] != 0.0 ? sst[i] : 0.0;
    ts[i] = ocnfrac[i] * target + (1.0 - ocnfrac[i]) * ts[i];
  }
}

OceanToAtmosphereOperator::OceanToAtmosphereOperator(
    const config::Section &o, const model::ModelInfo &info)
    : m_land(ref(o, "land_fraction")), m_predicted(ref(o, "predicted_ts")) {
  o.only({"operator", "land_fraction", "ocean", "predicted_ts", "to"});
  const auto oc = o.section("ocean");
  oc.only({"sst", "ice_fraction", "ocean_mask", "ice_mask"});
  m_sst = ref(oc, "sst");
  m_ice = ref(oc, "ice_fraction");
  m_ocean_mask = ref(oc, "ocean_mask");
  m_ice_mask = ref(oc, "ice_mask");
  const auto to = o.section("to");
  to.only({"ocnfrac", "icefrac", "ts"});
  m_ocnfrac = ref(to, "ocnfrac");
  m_icefrac = ref(to, "icefrac");
  m_ts = ref(to, "ts");
  for (const auto *r : {&m_ocnfrac, &m_icefrac, &m_ts}) {
    if (r->set() != model::FieldRef::Set::Inputs) {
      throw std::invalid_argument(to.where() + ": '" + r->to_string() +
                                  "' must be a network input.");
    }
  }
  if (info.layout == nullptr ||
      info.layout->source(m_land.name()) != fields::InputSource::Boundary) {
    throw std::invalid_argument(
        o.where() + ".land_fraction: '" + m_land.to_string() +
        "' must be a boundary input: fme's LANDFRAC is static.");
  }
}

model::Declarations OceanToAtmosphereOperator::declarations() const {
  model::Declarations d;
  d.writes_inputs = {m_ocnfrac.name(), m_icefrac.name(), m_ts.name()};
  d.aux = {kCarried, kSeenSst, kSeenIce};
  return d;
}

bool OceanToAtmosphereOperator::ocean_present(const model::Fields &f) const {
  return m_sst.present(f) && m_ice.present(f) && m_ocean_mask.present(f) &&
         m_ice_mask.present(f);
}

void OceanToAtmosphereOperator::fractions(model::Fields &f) {
  samudrace_fractions({m_land.read(f), m_ice.read(f), m_ocean_mask.read(f),
                       m_ice_mask.read(f)},
                      m_ocnfrac.write(f), m_icefrac.write(f));
}

bool OceanToAtmosphereOperator::ocean_changed(const model::Fields &f) const {
  const auto sst = m_sst.read(f);
  const auto ice = m_ice.read(f);
  const auto seen_sst = f.aux->get(kSeenSst);
  const auto seen_ice = f.aux->get(kSeenIce);
  // NaN never equals itself; the ice fraction's NaN counts as unchanged.
  for (std::size_t i = 0; i < sst.size(); ++i) {
    const bool same_ice = ice[i] == seen_ice[i] ||
                          (std::isnan(ice[i]) && std::isnan(seen_ice[i]));
    if (sst[i] != seen_sst[i] || !same_ice) {
      return true;
    }
  }
  return false;
}

void OceanToAtmosphereOperator::remember_ocean(const model::Fields &f) {
  const auto sst = m_sst.read(f);
  const auto ice = m_ice.read(f);
  std::copy(sst.begin(), sst.end(), f.aux->get(kSeenSst).begin());
  std::copy(ice.begin(), ice.end(), f.aux->get(kSeenIce).begin());
}

void OceanToAtmosphereOperator::initialize(const model::StepInfo &,
                                           model::Fields &f) {
  if (!ocean_present(f)) {
    return; // the initial condition's own fractions and TS for the first step
  }
  fractions(f);
  samudrace_prescribe_ts(m_ocnfrac.read(f), m_sst.read(f),
                         m_ocean_mask.read(f), m_ts.write(f));
}

void OceanToAtmosphereOperator::before_step(const model::StepInfo &,
                                            model::Fields &f) {
  fractions(f);
  const auto carried = f.aux->get(kCarried);
  auto ts = m_ts.write(f);
  std::copy(carried.begin(), carried.end(), ts.begin());
  if (ocean_changed(f)) {
    samudrace_prescribe_ts(m_ocnfrac.read(f), m_sst.read(f),
                           m_ocean_mask.read(f), ts);
  }
}

void OceanToAtmosphereOperator::after_step(const model::StepInfo &,
                                           model::Fields &f) {
  const auto predicted = m_predicted.read(f);
  auto carried = f.aux->get(kCarried);
  std::copy(predicted.begin(), predicted.end(), carried.begin());
  if (ocean_present(f)) {
    fractions(f);
    samudrace_prescribe_ts(m_ocnfrac.read(f), m_sst.read(f),
                           m_ocean_mask.read(f), carried);
    remember_ocean(f);
  }
}

void register_samudrace_operators() {
  model::OperatorRegistry::instance().add(
      "samudrace.ocean_to_atmosphere",
      [](const config::Section &o, const model::ModelInfo &i) {
        return std::make_unique<OceanToAtmosphereOperator>(o, i);
      });
}

} // namespace atm
} // namespace emulator
