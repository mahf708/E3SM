/**
 * @file sea_ice_operators.hpp
 * @brief The sea ice operator: an emulated ocean's ice fraction, reported
 *        to the coupler with a surface (compute_sea_ice_exports).
 */

#ifndef EMULATORICE_SEA_ICE_OPERATORS_HPP
#define EMULATORICE_SEA_ICE_OPERATORS_HPP

#include "operator.hpp"
#include "sea_ice_surface.hpp"

namespace emulator {
namespace ice {

/**
 * @brief `sea_ice.surface`: every sea_ice_export_names() field the coupler
 *        carries, on every call, over the component's domain.
 *
 * ```yaml
 * - operator: sea_ice.surface
 *   ice_fraction: exchange.ocn.sea_ice_fraction
 *   skin: prescribed            # or energy_balance (reads Faxa_lwdn)
 *   ice_thickness: {north: 2.0, south: 1.0}   # m, energy_balance only
 *   snow_depth: 0.2                           # m, energy_balance only
 * ```
 * Reads the sea_ice_import_names() imports; the domain mask and latitudes
 * are the component's.
 */
class SeaIceSurfaceOperator : public model::Operator {
public:
  SeaIceSurfaceOperator(const config::Section &options,
                        const model::ModelInfo &info);
  void exports(const model::StepInfo &info, model::Fields &f) override;

private:
  const model::Geometry *m_geometry;
  model::FieldRef m_fraction;
  SkinOptions m_skin;
};

/// Registers sea_ice.surface.  Idempotent.
void register_ice_operators();

} // namespace ice
} // namespace emulator

#endif // EMULATORICE_SEA_ICE_OPERATORS_HPP
