#include <catch2/catch.hpp>

#include "share/stochastic/associated_legendre.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <vector>

// Pure-math validation of the spherical-harmonic evaluation used by SKEBS/SPPT.
// The functions are KOKKOS_INLINE_FUNCTION and are exercised here on the host.

TEST_CASE ("associated_legendre")
{
  using namespace scream;
  using namespace scream::stochastic;

  const int N = 8;
  const int K = sph_num_coeffs(N);
  Kokkos::View<Real*,Kokkos::HostSpace> psi("psi", K);
  for (int i=0; i<K; ++i) psi(i) = 0;

  // A handful of nonzero modes (cosine and sine, various n,m).
  psi(sph_idx_cos(3,2,N)) =  0.7;
  psi(sph_idx_sin(3,2,N)) = -0.4;
  psi(sph_idx_cos(5,0,N)) =  0.3;   // m=0 (no lon dependence)
  psi(sph_idx_cos(6,4,N)) =  0.5;
  psi(sph_idx_sin(6,4,N)) =  0.2;

  const Real d2r = static_cast<Real>(M_PI/180.0);
  const Real a   = static_cast<Real>(6.371e6);

  SECTION ("m0_is_zonally_symmetric") {
    // With only the m=0 mode active, the pattern must not depend on longitude.
    Kokkos::View<Real*,Kokkos::HostSpace> p0("p0", K);
    for (int i=0;i<K;++i) p0(i)=0;
    p0(sph_idx_cos(4,0,N)) = 1.0;
    const Real mu = std::sin(20.0*d2r);
    const Real ref = sph_eval_pattern(N, mu, Real(0.0), p0);
    for (Real lon : {0.5, 1.7, 3.3, 5.9}) {
      REQUIRE(std::abs(sph_eval_pattern(N, mu, Real(lon), p0) - ref) < 1e-12);
    }
  }

  SECTION ("analytic_gradient_matches_finite_difference") {
    const std::vector<Real> lats = {-70,-45,-10,15,40,75};
    const std::vector<Real> lons = {0.3, 1.1, 2.5, 4.0, 5.5};
    const Real h = 1e-4;   // radians

    for (Real latd : lats) {
      const Real phi  = latd*d2r;
      const Real mu   = std::sin(phi);
      const Real cphi = std::sqrt(1 - mu*mu);
      for (Real lon : lons) {
        Real u, v;
        sph_eval_winds(N, mu, lon, psi, a, Real(1e-8), u, v);

        // Finite-difference the pattern (treated as a streamfunction).
        const Real psi_phip = sph_eval_pattern(N, std::sin(phi+h), lon, psi);
        const Real psi_phim = sph_eval_pattern(N, std::sin(phi-h), lon, psi);
        const Real dpsi_dphi = (psi_phip - psi_phim)/(2*h);

        const Real psi_lonp = sph_eval_pattern(N, mu, lon+h, psi);
        const Real psi_lonm = sph_eval_pattern(N, mu, lon-h, psi);
        const Real dpsi_dlon = (psi_lonp - psi_lonm)/(2*h);

        // Compare the un-scaled gradients (O(1) magnitudes) to avoid tiny numbers.
        //   a*u        == -dpsi/dphi
        //   a*cphi*v   ==  dpsi/dlon
        const Real lhs_u = a*u;
        const Real lhs_v = a*cphi*v;
        REQUIRE(std::abs(lhs_u - (-dpsi_dphi)) < 1e-4*(1 + std::abs(dpsi_dphi)));
        REQUIRE(std::abs(lhs_v -   dpsi_dlon ) < 1e-4*(1 + std::abs(dpsi_dlon)));
      }
    }
  }

  SECTION ("finite_near_poles") {
    for (Real latd : {-89.99, -89.9, 89.9, 89.99}) {
      const Real mu = std::sin(latd*d2r);
      Real u, v;
      sph_eval_winds(N, mu, Real(2.0), psi, a, Real(1e-3), u, v);
      REQUIRE(std::isfinite(u));
      REQUIRE(std::isfinite(v));
      REQUIRE(std::isfinite(sph_eval_pattern(N, mu, Real(2.0), psi)));
    }
  }
}
