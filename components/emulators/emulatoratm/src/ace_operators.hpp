/**
 * @file ace_operators.hpp
 * @brief The ACE atmosphere's operators: surface inputs from the coupler (or
 *        an emulated ocean), and the atmospheric state and fluxes it exports.
 */

#ifndef EMULATORATM_ACE_OPERATORS_HPP
#define EMULATORATM_ACE_OPERATORS_HPP

#include <map>
#include <string>

#include "ace_surface.hpp"
#include "ace_surface_inputs.hpp"
#include "operator.hpp"

namespace emulator {
namespace atm {

/**
 * @brief `ace.surface_inputs`: LANDFRAC, OCNFRAC, ICEFRAC and TS before each
 *        network step (compute_surface_inputs).
 *
 * ```yaml
 * - operator: ace.surface_inputs
 *   coupler: {lfrac: imports.Sf_lfrac, ofrac: imports.Sf_ofrac,
 *             ifrac: imports.Sf_ifrac, merged_ts: imports.Sx_t}
 *   emulator_ts: upper.TS
 *   ocean: {ice_fraction: exchange.ocn.sea_ice_fraction, sst: exchange.ocn.sst}
 *   to: {landfrac: inputs.LANDFRAC, ocnfrac: inputs.OCNFRAC,
 *        icefrac: inputs.ICEFRAC, ts: inputs.TS}
 *   tolerance: 0.05
 * ```
 * `ocean` is optional: with it, the non-land part is split by the emulated
 * ocean's own ice fraction and TS takes its SST.
 */
class SurfaceInputsOperator : public model::Operator {
public:
  SurfaceInputsOperator(const config::Section &options,
                        const model::ModelInfo &info);
  model::Declarations declarations() const override;
  void before_step(const model::StepInfo &, model::Fields &f) override;

private:
  model::FieldRef m_lfrac, m_ofrac, m_ifrac, m_sx_t, m_ts_emulator;
  bool m_from_ocean = false;
  model::FieldRef m_ocean_ice, m_ocean_sst;
  model::FieldRef m_landfrac, m_ocnfrac, m_icefrac, m_ts;
  double m_tolerance;
};

/**
 * @brief `ace.surface_exports`: the atmospheric state and fluxes the coupler
 *        receives, on every call (compute_surface_exports).
 *
 * ```yaml
 * - operator: ace.surface_exports
 *   layer: near_surface          # or lowest_level
 *   reference_height: 10.0
 *   cap_humidity: true
 *   frozen_precip_in_m_per_s: false
 *   diurnal_shortwave: true
 *   from: {ps: state.PS, phis: inputs.PHIS, t_lowest: state.T_7, ...}
 *   to: {z: exports.Sa_z, u: exports.Sa_u, ...}
 * ```
 * `from` keys: ps, phis, t_lowest, q_lowest, u_lowest, v_lowest, flds, fsds,
 * precip, solin_now, solin_window, and optionally t_2m, q_2m, u_10m, v_10m,
 * fsus, frozen_precip.  `to` keys: z, u, v, tbot, ptem, shum, pbot, pslv,
 * dens, topo, lwdn, rainc, rainl, snowc, snowl, swndr, swvdr, swndf, swvdf,
 * swnet.
 */
class SurfaceExportsOperator : public model::Operator {
public:
  SurfaceExportsOperator(const config::Section &options,
                         const model::ModelInfo &info);
  void exports(const model::StepInfo &, model::Fields &f) override;

private:
  SurfaceOptions m_options;
  std::map<std::string, model::FieldRef> m_from;
  std::map<std::string, model::FieldRef> m_to;
};

/// Registers the atmosphere operators. Idempotent.
void register_atm_operators();

} // namespace atm
} // namespace emulator

#endif // EMULATORATM_ACE_OPERATORS_HPP
