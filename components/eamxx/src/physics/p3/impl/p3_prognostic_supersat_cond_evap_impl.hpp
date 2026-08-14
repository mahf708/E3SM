#ifndef P3_PROGNOSTIC_SUPERSAT_COND_EVAP_IMPL_HPP
#define P3_PROGNOSTIC_SUPERSAT_COND_EVAP_IMPL_HPP

#include "p3_functions.hpp"
#include "share/physics/physics_constants.hpp"

namespace scream {
namespace p3 {

/*
 * Implementation of p3 prognostic supersaturation condensation/evaporation.
 * Clients should NOT #include this file, but include p3_functions.hpp instead.
 */

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::prognostic_supersat_cond_evap(
  const Pack& qc_incld, const Pack& qv, const Pack& qv_sat_l, const Pack& dqsdt,
  const Pack& ab, const Pack& epsc, const Pack& rho, const Pack& omega_mp,
  const Pack& tke_mp, const Scalar& dt, const Scalar& inv_dt,
  Pack& w_updraft, Pack& cond_evap_tend,
  const Mask& context)
{
  constexpr Scalar g      = C::gravit.value;
  constexpr Scalar inv_cp = C::INV_CP.value;
  constexpr Scalar THIRD  = C::THIRD;

  w_updraft      = 0;
  cond_evap_tend = 0;

  if (!context.any()) return;

  // epsc and ab are only meaningful inside the context mask. Set the copies used
  // as denominators below to one elsewhere, so that the reciprocals are finite
  // on the inactive lanes of the pack. Values on active lanes are untouched.
  Pack tau_inv(1), ab_safe(1);
  tau_inv.set(context, epsc); // inverse condensation relaxation timescale [1/s]
  ab_safe.set(context, ab);
  const auto tau = 1 / tau_inv;

  // Updraft speed driving the supersaturation: the resolved part from omega plus
  // a sub-grid part from TKE, assuming isotropic turbulence so w'^2 = (2/3) tke.
  w_updraft.set(context, -omega_mp / (rho * g) + sqrt(max(0, 2 * THIRD * tke_mp)));

  // Supersaturation w.r.t. liquid and its production rate by adiabatic ascent,
  // dqv_sat/dt = dqsdt * dT/dt with dT/dt = -w g / cp.
  Pack ssat_l(0), ssat_src(0);
  ssat_l.set(context, qv - qv_sat_l);
  ssat_src.set(context, -dqsdt * (-w_updraft * g * inv_cp));

  // Step-averaged condensation (>0) / evaporation (<0) rate, from the analytic
  // solution of ds/dt = ssat_src - s/tau over the time step.
  cond_evap_tend.set(context, (ssat_src * epsc * tau +
                               (ssat_l - ssat_src * tau) * inv_dt * epsc * tau *
                               (1 - exp(-tau_inv * dt))) / ab_safe);

  // Evaporation cannot remove more liquid than is present.
  cond_evap_tend.set(context, max(-qc_incld / dt, cond_evap_tend));
}

} // namespace p3
} // namespace scream

#endif // P3_PROGNOSTIC_SUPERSAT_COND_EVAP_IMPL_HPP
