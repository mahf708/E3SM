/**
 * @file sea_ice_operators.cpp
 * @brief Implementation of the sea ice operator.
 */

#include "sea_ice_operators.hpp"

namespace emulator {
namespace ice {

SeaIceSurfaceOperator::SeaIceSurfaceOperator(const config::Section &o,
                                             const model::ModelInfo &info)
    : m_geometry(info.geometry),
      m_fraction(model::FieldRef::parse(o.string("ice_fraction"),
                                        o.where() + ".ice_fraction")) {
  o.only({"operator", "ice_fraction"});
}

void SeaIceSurfaceOperator::exports(const model::StepInfo &info,
                                    model::Fields &f) {
  compute_sea_ice_exports(
      info.now,
      {m_geometry->lat, m_geometry->domain_mask, m_fraction.read(f)},
      *f.imports, *f.exports);
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
