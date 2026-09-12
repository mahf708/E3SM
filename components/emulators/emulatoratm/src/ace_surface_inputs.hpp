/**
 * @file ace_surface_inputs.hpp
 * @brief The surface channels the ACE atmosphere reads from the coupler.
 */

#ifndef EMULATORATM_ACE_SURFACE_INPUTS_HPP
#define EMULATORATM_ACE_SURFACE_INPUTS_HPP

#include <cstddef>
#include <span>

namespace emulator {
namespace atm {

/// Per-cell values at the step the network is about to run.
struct SurfaceCouplerInputs {
  std::span<const double> lfrac; ///< Sf_lfrac
  std::span<const double> ofrac; ///< Sf_ofrac
  std::span<const double> ifrac; ///< Sf_ifrac
  std::span<const double> sx_t;  ///< Sx_t: already fraction-weighted, K
  /// The emulator's own TS from its latest prediction, for the part of the
  /// cell no surface model covers.
  std::span<const double> ts_emulator;
  /// Optional, both or neither: an emulated ocean's own sea-ice fraction and
  /// unmerged SST, published in-process by that component.
  std::span<const double> ocean_ice_fraction;
  std::span<const double> ocean_sst;
};

struct SurfaceChannels {
  std::span<double> landfrac, ocnfrac, icefrac, ts;
};

struct SurfaceInputCounts {
  std::size_t clipped = 0;      ///< cells with a fraction outside [0, 1]
  std::size_t renormalized = 0; ///< cells whose fractions summed above one
  double worst_clip = 0.0;
  double worst_excess = 0.0;
};

/**
 * @brief LANDFRAC, OCNFRAC, ICEFRAC and TS for the network.
 *
 * The rule, from EATM's review, where getting it wrong cost 150 W/m2:
 *
 *  - each fraction is clipped to [0, 1], and if they sum above one they are
 *    renormalized; the *deficit*, 1 - sum, is the part no surface model
 *    covers.  With a stub land model that is all the land: Sf_lfrac arrives
 *    as 0 and Sx_t as 0 K over a quarter of the globe.
 *  - LANDFRAC = lfrac + deficit, so the network sees the real land.
 *  - Sx_t is already weighted by the fractions, so TS = Sx_t + deficit *
 *    TS_emulator.  Reweighting Sx_t instead put 70 K errors on coastlines.
 *  - with an emulated ocean's ice fraction s and SST, the non-land part is
 *    re-split: ICEFRAC = s (1 - L), OCNFRAC = max(1 - L - s (1 - L), 0), and
 *    TS = o SST + (1 - o) TS_emulator with o = (1 - L)(1 - s).
 *
 * @param tolerance largest clip or excess accepted: anything beyond
 *        round-off means Sx_t was merged from different fractions than the
 *        ones the network is being given
 * @throws std::runtime_error beyond the tolerance;
 *         std::invalid_argument on mismatched lengths
 */
SurfaceInputCounts compute_surface_inputs(const SurfaceCouplerInputs &in,
                                          SurfaceChannels &out,
                                          double tolerance = 0.05);

} // namespace atm
} // namespace emulator

#endif // EMULATORATM_ACE_SURFACE_INPUTS_HPP
