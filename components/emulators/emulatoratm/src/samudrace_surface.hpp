/**
 * @file samudrace_surface.hpp
 * @brief SamudrACE's ocean-to-atmosphere exchange, as fme computes it.
 */

#ifndef EMULATORATM_SAMUDRACE_SURFACE_HPP
#define EMULATORATM_SAMUDRACE_SURFACE_HPP

#include <span>
#include <string>

#include "operator.hpp"

namespace emulator {
namespace atm {

/// One rank's cells of what the emulated ocean hands the atmosphere.
struct OceanToAtmosphereFields {
  std::span<const double> land_fraction;      ///< LANDFRAC, static
  std::span<const double> ocean_ice_fraction; ///< share of the sea surface; NaN is 0
  std::span<const double> ocean_mask;         ///< mask_2d
  std::span<const double> ice_mask;           ///< mask_ICEFRAC
};

/**
 * @brief fme's surface fractions from the ocean's ice (CoupledStepper
 *        _forcings_from_ocean_with_ocean_fraction, OceanData):
 *
 *   ICEFRAC = s (1 - L)
 *   OCNFRAC = max(1 - L - ICEFRAC, 0)
 *
 * then OCNFRAC = 0 where the ocean mask is 0 and ICEFRAC = 0 where the ice
 * mask is 0.  OCNFRAC is computed from the unmasked ICEFRAC, as fme does.
 */
void samudrace_fractions(const OceanToAtmosphereFields &in,
                         std::span<double> ocnfrac, std::span<double> icefrac);

/**
 * @brief fme's prescribed surface temperature (Prescriber with the
 *        checkpoint's OceanConfig: interpolate, weight OCNFRAC):
 *
 *   TS = OCNFRAC SST + (1 - OCNFRAC) TS
 *
 * with SST taken as 0 where the ocean mask is 0 (where OCNFRAC is 0 too).
 * `ts` holds the atmosphere's TS on entry and the prescribed TS on return.
 */
void samudrace_prescribe_ts(std::span<const double> ocnfrac,
                            std::span<const double> sst,
                            std::span<const double> ocean_mask,
                            std::span<double> ts);

/**
 * @brief `samudrace.ocean_to_atmosphere`: the atmosphere's OCNFRAC, ICEFRAC
 *        and TS inputs from the emulated ocean, with fme's timing.
 *
 * In fme the ocean's state is held over a window.  Each atmosphere step's TS
 * output is blended with the SST once, and that blended output is the next
 * step's TS input; at a window start, when the ocean's state is new, the
 * carried TS is blended again with the new SST.  So:
 *
 *  - after_step: the carried TS (aux) is the prediction blended with the
 *    current SST;
 *  - before_step: the fractions from the current ocean state; the TS input is
 *    the carried TS, blended again only if the ocean's SST or ice fraction
 *    changed since the carried TS was made;
 *  - initialize: if the ocean has published already (it initializes before
 *    the atmosphere in a single process), the initial condition's TS and
 *    fractions are prescribed from it, as fme prescribes the initial SST;
 *    otherwise (the MCT driver initializes the atmosphere first) they are the
 *    initial condition's own until the first step.
 *
 * ```yaml
 * - operator: samudrace.ocean_to_atmosphere
 *   land_fraction: inputs.LANDFRAC
 *   ocean: {sst: exchange.ocn.sst_raw, ice_fraction: exchange.ocn.sea_ice_fraction,
 *           ocean_mask: exchange.ocn.domain.mask, ice_mask: exchange.ocn.ice_mask}
 *   predicted_ts: upper.TS
 *   to: {ocnfrac: inputs.OCNFRAC, icefrac: inputs.ICEFRAC, ts: inputs.TS}
 * ```
 */
class OceanToAtmosphereOperator : public model::Operator {
public:
  OceanToAtmosphereOperator(const config::Section &options,
                            const model::ModelInfo &info);
  model::Declarations declarations() const override;
  void initialize(const model::StepInfo &, model::Fields &f) override;
  void before_step(const model::StepInfo &, model::Fields &f) override;
  void after_step(const model::StepInfo &, model::Fields &f) override;

private:
  bool ocean_present(const model::Fields &f) const;
  void fractions(model::Fields &f);
  /// Whether the ocean's SST or ice differ from those the carried TS used.
  bool ocean_changed(const model::Fields &f) const;
  void remember_ocean(const model::Fields &f);

  model::FieldRef m_land, m_sst, m_ice, m_ocean_mask, m_ice_mask, m_predicted;
  model::FieldRef m_ocnfrac, m_icefrac, m_ts;
  static constexpr const char *kCarried = "samudrace_ts";
  static constexpr const char *kSeenSst = "samudrace_seen_sst";
  static constexpr const char *kSeenIce = "samudrace_seen_ice";
};

/// Registers samudrace.ocean_to_atmosphere.  Idempotent.
void register_samudrace_operators();

} // namespace atm
} // namespace emulator

#endif // EMULATORATM_SAMUDRACE_SURFACE_HPP
