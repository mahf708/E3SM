/**
 * @file ocean_operators.hpp
 * @brief The emulated ocean's operators: what the coupler receives, the SSH
 *        slope, and forcing averaged from the coupler's merged fluxes.
 */

#ifndef EMULATOROCN_OCEAN_OPERATORS_HPP
#define EMULATOROCN_OCEAN_OPERATORS_HPP

#include <map>
#include <string>
#include <vector>

#include "common_operators.hpp"
#include "ocean_forcing.hpp"
#include "operator.hpp"

namespace emulator {
namespace ocn {

/**
 * @brief `ocean.surface_exports`: the ocean state the coupler receives.
 *
 * On the ocean mask: SST floored at freezing, salinity clamped to [0, 60],
 * velocities and SSH as predicted.  Off it: freezing, `land_salinity`, 0.
 * The sea-ice fraction is clamped to [0, 1] inside the ice channel's own
 * mask and 0 outside it: on the ocean mask it put sea ice in the tropics.
 *
 * ```yaml
 * - operator: ocean.surface_exports
 *   freezing_sst: 271.35
 *   land_salinity: 34.7
 *   ocean_mask: statics.mask_2d
 *   ice_mask: statics.mask_ocean_sea_ice_fraction
 *   from: {sst: state.sst, salinity: state.salinityCoarsened_0,
 *          u: state.velocityZonalCoarsened_0,
 *          v: state.velocityMeridionalCoarsened_0, ssh: state.ssh,
 *          ice_fraction: state.ocean_sea_ice_fraction}
 *   to: {sst: exports.So_t, salinity: exports.So_s, u: exports.So_u,
 *        v: exports.So_v, ssh: exports.So_ssh, ice_fraction: aux.sea_ice_fraction}
 * ```
 * Both masks must be binary.
 */
class OceanSurfaceExports : public model::Operator {
public:
  OceanSurfaceExports(const config::Section &options,
                      const model::ModelInfo &info);
  model::Declarations declarations() const override;
  void exports(const model::StepInfo &, model::Fields &f) override;

private:
  double m_freezing;
  double m_land_salinity;
  model::FieldRef m_ocean_mask, m_ice_mask;
  std::map<std::string, model::FieldRef> m_from, m_to;
  bool m_checked = false;
};

/**
 * @brief `ocean.ssh_gradients`: the sea surface height slope, on the whole
 *        grid at the root, since it needs neighbouring rows (ssh_gradients).
 *
 * ```yaml
 * - operator: ocean.ssh_gradients
 *   ssh: exports.So_ssh
 *   mask: statics.mask_2d
 *   to: {dhdx: exports.So_dhdx, dhdy: exports.So_dhdy}
 * ```
 * Collective: every rank calls it on every exports call.
 */
class SshGradients : public model::Operator {
public:
  SshGradients(const config::Section &options, const model::ModelInfo &info);
  model::Declarations declarations() const override;
  void exports(const model::StepInfo &, model::Fields &f) override;

private:
  const model::Geometry *m_geometry;
  model::FieldRef m_ssh, m_mask, m_dhdx, m_dhdy;
  std::vector<double> m_global_mask; ///< root only, gathered once
  bool m_have_mask = false;
};

/**
 * @brief `ocean.coupler_window_mean`: forcing averaged from the coupler's
 *        merged fluxes (coupler_forcing_sample), for an emulated ocean under
 *        a model atmosphere.
 *
 * ```yaml
 * - operator: ocean.coupler_window_mean
 *   channels: [TAUX, TAUY, surface_precipitation_rate, ...]
 *   also_into_suffix: ":next"
 *   clip_min_zero: [surface_precipitation_rate, frozen_precipitation_rate]
 *   unweight_by_ice_fraction: true
 *   unweight_stress: false
 *   ocean_albedo: 0.06
 * ```
 */
class CouplerWindowMean : public model::WindowMeanForcing {
public:
  CouplerWindowMean(const config::Section &options,
                    const model::ModelInfo &info);

protected:
  void fill_sample(model::Fields &f, fields::FieldSet &sample) override;

private:
  CouplerForcingOptions m_options;
};

/// Registers the ocean operators.  Idempotent.
void register_ocn_operators();

} // namespace ocn
} // namespace emulator

#endif // EMULATOROCN_OCEAN_OPERATORS_HPP
