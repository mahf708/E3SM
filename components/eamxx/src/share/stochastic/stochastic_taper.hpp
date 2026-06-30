#ifndef EAMXX_STOCHASTIC_TAPER_HPP
#define EAMXX_STOCHASTIC_TAPER_HPP

#include "share/core/eamxx_types.hpp"

#include <Kokkos_Core.hpp>

namespace scream {
namespace stochastic {

/*
 * Vertical tapering mask for stochastic perturbations.
 *
 * Returns a smooth 0 -> 1 -> 0 weight as a function of pressure p (Pa):
 *   - 1 in the free atmosphere,
 *   - ramps to 0 below p_bot_ramp (towards the surface, p -> p_sfc), so that
 *     perturbations vanish in the boundary layer and do not corrupt surface
 *     fluxes,
 *   - ramps to 0 above p_top_ramp (towards the model top), so that
 *     perturbations vanish in the sponge layer / stratosphere.
 *
 * A cosine ramp is used for smoothness. The ramps are expressed relative to the
 * surface pressure (bottom) and an absolute pressure (top), matching the usual
 * SPPT/SKEBS practice. All pressures are in Pa.
 *
 *   p_bot_ramp : perturbation is full strength for p <= p_sfc - p_bot_ramp,
 *                and ramps linearly-in-pressure to 0 at p = p_sfc.
 *   p_top_ramp : perturbation is full strength for p >= p_top_ramp, and ramps
 *                to 0 at p = 0 (model top).
 */
KOKKOS_INLINE_FUNCTION
Real vertical_taper (const Real p, const Real p_sfc,
                     const Real p_bot_ramp, const Real p_top_ramp)
{
  constexpr Real pi = static_cast<Real>(3.14159265358979323846);
  Real w = 1;

  // Bottom (boundary-layer) ramp
  if (p_bot_ramp > 0) {
    const Real p_full = p_sfc - p_bot_ramp;   // full strength at/above this
    if (p >= p_sfc) {
      w = 0;
    } else if (p > p_full) {
      const Real x = (p_sfc - p) / p_bot_ramp;          // 0 at sfc -> 1 at p_full
      w *= static_cast<Real>(0.5) * (static_cast<Real>(1) - Kokkos::cos(pi*x));
    }
  }

  // Top (sponge-layer) ramp
  if (p_top_ramp > 0) {
    if (p <= 0) {
      w = 0;
    } else if (p < p_top_ramp) {
      const Real x = p / p_top_ramp;                    // 0 at top -> 1 at p_top_ramp
      w *= static_cast<Real>(0.5) * (static_cast<Real>(1) - Kokkos::cos(pi*x));
    }
  }

  return w;
}

} // namespace stochastic
} // namespace scream

#endif // EAMXX_STOCHASTIC_TAPER_HPP
