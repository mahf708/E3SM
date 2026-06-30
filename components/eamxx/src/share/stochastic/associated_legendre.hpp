#ifndef EAMXX_ASSOCIATED_LEGENDRE_HPP
#define EAMXX_ASSOCIATED_LEGENDRE_HPP

#include "share/core/eamxx_types.hpp"

#include <Kokkos_Core.hpp>

namespace scream {
namespace stochastic {

/*
 * Device-friendly evaluation of a band-limited real spherical-harmonic field
 * and (optionally) the rotational wind implied by treating that field as a
 * streamfunction.
 *
 * The field is
 *
 *   P(lambda,phi) = sum_{m=0}^{N} sum_{n=max(1,m)}^{N}
 *                     Pbar_n^m(mu) * [ A_{n,m} cos(m lambda) + B_{n,m} sin(m lambda) ]
 *
 * where mu = sin(phi) (phi = latitude, lambda = longitude), Pbar_n^m is the
 * fully-normalized associated Legendre function, and the real coefficients
 * A (cosine) and B (sine, only for m>0) are stored in a flat coefficient view
 * (see indexing helpers below). The degree-0 mode is dropped, so the field has
 * exactly zero global mean.
 *
 * The coefficients are laid out in two padded (N+1)x(N+1) blocks (cosine then
 * sine). Entries with n<m are never used and are expected to be zero. The
 * padded layout keeps device indexing branch-free and is tiny in memory
 * (2*(N+1)^2 reals, e.g. ~2 KB for N=30).
 *
 * NOTE on normalization: the recurrences below implement one self-consistent
 * normalization (Pbar_0^0 = 1, no Condon-Shortley phase). Any overall constant
 * is absorbed into the per-degree noise amplitude sigma_n in the pattern
 * generator, so the precise convention does not matter as long as the value
 * routine and the derivative routine use the *same* convention -- which they
 * do, since the derivative is expressed through the same Pbar_n^m values. The
 * unit test validates the analytic gradient against finite differences.
 */

// Number of reals in the (padded) coefficient view for truncation N.
KOKKOS_INLINE_FUNCTION
constexpr int sph_num_coeffs (const int N) { return 2*(N+1)*(N+1); }

// Flat index of the cosine coefficient A_{n,m}.
KOKKOS_INLINE_FUNCTION
int sph_idx_cos (const int n, const int m, const int N) { return m*(N+1) + n; }

// Flat index of the sine coefficient B_{n,m} (only meaningful for m>0).
KOKKOS_INLINE_FUNCTION
int sph_idx_sin (const int n, const int m, const int N) { return (N+1)*(N+1) + m*(N+1) + n; }

// Evaluate the scalar pattern P(lambda,phi) at a single point.
//   mu  = sin(latitude)
//   lon = longitude in radians
template<typename PsiView>
KOKKOS_INLINE_FUNCTION
Real sph_eval_pattern (const int N, const Real mu, const Real lon, const PsiView& psi)
{
  const Real one_minus_mu2 = (mu*mu < static_cast<Real>(1)) ? (static_cast<Real>(1) - mu*mu)
                                                            : static_cast<Real>(0);
  const Real cphi = Kokkos::sqrt(one_minus_mu2);   // cos(latitude) >= 0

  Real sum  = 0;
  Real Pmm  = 1;   // running sectoral value Pbar_m^m, starts at Pbar_0^0 = 1
  for (int m=0; m<=N; ++m) {
    if (m>0) {
      Pmm *= Kokkos::sqrt( static_cast<Real>(2*m+1) / static_cast<Real>(2*m) ) * cphi;
    }
    const Real cml = Kokkos::cos(static_cast<Real>(m)*lon);
    const Real sml = Kokkos::sin(static_cast<Real>(m)*lon);

    Real prev1 = 0, prev2 = 0;   // Pbar_{n-1}^m, Pbar_{n-2}^m
    for (int n=m; n<=N; ++n) {
      Real Pn;
      if (n==m) {
        Pn = Pmm;
      } else if (n==m+1) {
        Pn = Kokkos::sqrt(static_cast<Real>(2*m+3)) * mu * Pmm;
      } else {
        const Real a = Kokkos::sqrt( (static_cast<Real>(2*n+1)*static_cast<Real>(2*n-1)) /
                                     (static_cast<Real>(n-m)*static_cast<Real>(n+m)) );
        const Real b = Kokkos::sqrt( (static_cast<Real>(2*n+1)*static_cast<Real>(n+m-1)*static_cast<Real>(n-m-1)) /
                                     (static_cast<Real>(2*n-3)*static_cast<Real>(n+m)*static_cast<Real>(n-m)) );
        Pn = a*mu*prev1 - b*prev2;
      }

      if (n>=1) {   // skip the (n=0,m=0) global-mean mode
        const Real A = psi(sph_idx_cos(n,m,N));
        const Real B = (m==0) ? static_cast<Real>(0) : psi(sph_idx_sin(n,m,N));
        sum += Pn * (A*cml + B*sml);
      }

      prev2 = prev1;
      prev1 = Pn;
    }
  }
  return sum;
}

// Evaluate the rotational (non-divergent) wind obtained by treating the
// pattern as a streamfunction psi:
//     u = -(1/a) d(psi)/d(phi)
//     v =  (1/(a cos phi)) d(psi)/d(lambda)
// The cos(phi) singularity at the poles is regularized by flooring cos(phi)
// at cphi_floor. For the band-limited fields used here this only affects a
// negligible cap around each pole.
template<typename PsiView>
KOKKOS_INLINE_FUNCTION
void sph_eval_winds (const int N, const Real mu, const Real lon, const PsiView& psi,
                     const Real a_radius, const Real cphi_floor,
                     Real& u, Real& v)
{
  const Real one_minus_mu2 = (mu*mu < static_cast<Real>(1)) ? (static_cast<Real>(1) - mu*mu)
                                                            : static_cast<Real>(0);
  const Real cphi = Kokkos::sqrt(one_minus_mu2);
  const Real cphi_safe = (cphi > cphi_floor) ? cphi : cphi_floor;

  Real s_lam = 0;   // d(psi)/d(lambda)
  Real s_phi = 0;   // cos(phi) * d(psi)/d(phi)  (pole-safe form)
  Real Pmm   = 1;
  for (int m=0; m<=N; ++m) {
    if (m>0) {
      Pmm *= Kokkos::sqrt( static_cast<Real>(2*m+1) / static_cast<Real>(2*m) ) * cphi;
    }
    const Real cml = Kokkos::cos(static_cast<Real>(m)*lon);
    const Real sml = Kokkos::sin(static_cast<Real>(m)*lon);

    Real prev1 = 0, prev2 = 0;
    for (int n=m; n<=N; ++n) {
      Real Pn;
      if (n==m) {
        Pn = Pmm;
      } else if (n==m+1) {
        Pn = Kokkos::sqrt(static_cast<Real>(2*m+3)) * mu * Pmm;
      } else {
        const Real a = Kokkos::sqrt( (static_cast<Real>(2*n+1)*static_cast<Real>(2*n-1)) /
                                     (static_cast<Real>(n-m)*static_cast<Real>(n+m)) );
        const Real b = Kokkos::sqrt( (static_cast<Real>(2*n+1)*static_cast<Real>(n+m-1)*static_cast<Real>(n-m-1)) /
                                     (static_cast<Real>(2*n-3)*static_cast<Real>(n+m)*static_cast<Real>(n-m)) );
        Pn = a*mu*prev1 - b*prev2;
      }

      if (n>=1) {
        const Real A = psi(sph_idx_cos(n,m,N));
        const Real B = (m==0) ? static_cast<Real>(0) : psi(sph_idx_sin(n,m,N));

        // d/d(lambda): cos(m l)->-m sin(m l), sin(m l)->m cos(m l)
        s_lam += Pn * static_cast<Real>(m) * (-A*sml + B*cml);

        // cos(phi) d(Pbar_n^m)/d(phi) = (1-mu^2) dPbar/dmu
        //   = -n mu Pbar_n^m + C_n^m Pbar_{n-1}^m,  C_n^m = sqrt((2n+1)(n-m)(n+m)/(2n-1))
        // (C_m^m = 0, so the prev1 term vanishes correctly at n=m).
        const Real C = (n>m)
          ? Kokkos::sqrt( (static_cast<Real>(2*n+1)*static_cast<Real>(n-m)*static_cast<Real>(n+m)) /
                          static_cast<Real>(2*n-1) )
          : static_cast<Real>(0);
        const Real dphi = -static_cast<Real>(n)*mu*Pn + C*prev1;
        s_phi += dphi * (A*cml + B*sml);
      }

      prev2 = prev1;
      prev1 = Pn;
    }
  }

  u = -s_phi / (a_radius * cphi_safe);
  v =  s_lam / (a_radius * cphi_safe);
}

} // namespace stochastic
} // namespace scream

#endif // EAMXX_ASSOCIATED_LEGENDRE_HPP
