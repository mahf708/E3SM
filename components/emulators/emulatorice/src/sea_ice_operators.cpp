/**
 * @file sea_ice_operators.cpp
 * @brief Implementation of the sea ice operator.
 */

#include "sea_ice_operators.hpp"

#include <stdexcept>

namespace emulator {
namespace ice {

SeaIceSurfaceOperator::SeaIceSurfaceOperator(const config::Section &o,
                                             const model::ModelInfo &info)
    : m_geometry(info.geometry),
      m_fraction(model::FieldRef::parse(o.string("ice_fraction"),
                                        o.where() + ".ice_fraction")) {
  o.only({"operator", "ice_fraction", "skin", "ice_thickness", "snow_depth"});
  const auto skin = o.string_or("skin", "prescribed");
  if (skin == "energy_balance") {
    m_skin.mode = SkinOptions::Mode::EnergyBalance;
  } else if (skin != "prescribed") {
    throw std::invalid_argument(o.where() + ".skin: '" + skin +
                                "' is not prescribed or energy_balance.");
  }
  if (o.has("ice_thickness")) {
    const auto t = o.section("ice_thickness");
    t.only({"north", "south"});
    m_skin.thickness_north = t.number_or("north", m_skin.thickness_north);
    m_skin.thickness_south = t.number_or("south", m_skin.thickness_south);
  }
  m_skin.snow_depth = o.number_or("snow_depth", m_skin.snow_depth);
}

void SeaIceSurfaceOperator::exports(const model::StepInfo &info,
                                    model::Fields &f) {
  compute_sea_ice_exports(
      info.now,
      {m_geometry->lat, m_geometry->domain_mask, m_fraction.read(f)},
      *f.imports, *f.exports, m_skin);
}

void register_ice_operators() {
  model::OperatorRegistry::instance().add(
      "sea_ice.surface",
      [](const config::Section &o, const model::ModelInfo &i) {
        return std::make_unique<SeaIceSurfaceOperator>(o, i);
      });
}

} // namespace ice
} // namespace emulator
