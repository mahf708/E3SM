#include "share/stochastic/spectral_pattern_generator.hpp"
#include "share/stochastic/associated_legendre.hpp"
#include "share/field/field_identifier.hpp"
#include "share/field/field_layout.hpp"

#include <ekat_units.hpp>

#include <cmath>
#include <random>

namespace scream {
namespace stochastic {

std::uint64_t SpectralPatternGenerator::hash_seed (std::uint64_t x)
{
  // splitmix64
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  x =  x ^ (x >> 31);
  return x;
}

void SpectralPatternGenerator::
initialize (const std::string& name_prefix,
            const std::shared_ptr<const AbstractGrid>& grid,
            const Params& params,
            const ekat::Comm& comm)
{
  m_params = params;
  m_comm   = comm;
  m_prefix = name_prefix;

  EKAT_REQUIRE_MSG (m_params.truncation >= 1,
      "Error! [SpectralPatternGenerator] truncation must be >= 1.\n");
  EKAT_REQUIRE_MSG (m_params.nmin >= 1 && m_params.nmin <= m_params.truncation,
      "Error! [SpectralPatternGenerator] need 1 <= nmin <= truncation.\n");
  EKAT_REQUIRE_MSG (m_params.decorr_time > 0,
      "Error! [SpectralPatternGenerator] decorrelation time must be > 0.\n");

  const int N = m_params.truncation;
  m_num_coeffs = sph_num_coeffs(N);

  // Per-degree stationary standard deviation: power-law spectrum, normalized so
  // that the realized pattern variance equals stddev^2. The (2n+1) factor is the
  // number of order-m modes contributing at degree n.
  m_sigma.assign(N+1, 0);
  double norm = 0;
  for (int n=m_params.nmin; n<=N; ++n) {
    const double shape = std::pow(static_cast<double>(n)*(n+1), -m_params.power);
    m_sigma[n] = static_cast<Real>(shape);
    norm += shape * (2.0*n + 1.0);
  }
  EKAT_REQUIRE_MSG (norm > 0,
      "Error! [SpectralPatternGenerator] degenerate spectrum normalization.\n");
  const double g2 = (m_params.stddev*m_params.stddev) / norm;
  for (int n=m_params.nmin; n<=N; ++n) {
    m_sigma[n] = static_cast<Real>(std::sqrt(g2) * std::sqrt(m_sigma[n]));
  }

  // Replicated coefficient field (no COL tag -> written/read identically by
  // every rank, classified as Vector0D). A unique dimension name avoids IO
  // dim-name collisions across schemes.
  using namespace ShortFieldTagsNames;
  const auto& gname = grid->name();
  FieldLayout layout ({CMP}, {m_num_coeffs}, {m_prefix+"_nmode"});
  FieldIdentifier fid (m_prefix+"_spectral_coeffs", layout, ekat::units::none, gname);
  m_coeffs = Field(fid);
  m_coeffs.allocate_view();
  m_coeffs.deep_copy(0);

  // Per-coefficient stddev (0 for unused/out-of-band padded entries), on device.
  m_sigma_dev  = view_1d<Real>("stoch_sigma", m_num_coeffs);
  m_noise_dev  = view_1d<Real>("stoch_noise", m_num_coeffs);
  m_noise_host = Kokkos::create_mirror_view(m_noise_dev);

  auto sigma_h = Kokkos::create_mirror_view(m_sigma_dev);
  const int block = (N+1)*(N+1);
  for (int k=0; k<m_num_coeffs; ++k) {
    const bool is_sin = (k >= block);
    const int  kk = is_sin ? k - block : k;
    const int  m  = kk / (N+1);
    const int  n  = kk % (N+1);
    const bool valid = (n>=m) && (n>=m_params.nmin) && (n<=N) && (!is_sin || m>=1);
    sigma_h(k) = valid ? m_sigma[n] : static_cast<Real>(0);
  }
  Kokkos::deep_copy(m_sigma_dev, sigma_h);

  m_step = 0;
}

void SpectralPatternGenerator::set_columns (const Field& lat_deg, const Field& lon_deg)
{
  const auto lat = lat_deg.get_view<const Real*>();
  const auto lon = lon_deg.get_view<const Real*>();
  m_ncols = lat.extent_int(0);

  m_sin_lat = view_1d<Real>("stoch_sin_lat", m_ncols);
  m_lon     = view_1d<Real>("stoch_lon_rad", m_ncols);

  const Real deg2rad = static_cast<Real>(M_PI/180.0);
  auto sin_lat = m_sin_lat;
  auto lon_rad = m_lon;
  Kokkos::parallel_for("stoch_setup_columns", m_ncols, KOKKOS_LAMBDA (const int i) {
    sin_lat(i) = Kokkos::sin(lat(i)*deg2rad);
    lon_rad(i) = lon(i)*deg2rad;
  });
  Kokkos::fence();
}

void SpectralPatternGenerator::advance (const double dt)
{
  const Real alpha = static_cast<Real>(std::exp(-dt/m_params.decorr_time));
  const Real beta  = std::sqrt(std::max(static_cast<Real>(0), static_cast<Real>(1) - alpha*alpha));

  // Draw one independent N(0,1) per coefficient on the host. The per-coefficient
  // seed depends on (base_seed, member_id, step, flat-index), so the noise is
  // identical on every rank and exactly reproducible after a restart. We use the
  // standardized mt19937_64 engine plus an explicit Box-Muller transform (rather
  // than the non-portable std::normal_distribution) for cross-compiler BFB.
  const std::int64_t step = m_step;
  const int base = m_params.base_seed;
  const int member = m_params.member_id;
  for (int k=0; k<m_num_coeffs; ++k) {
    std::uint64_t key = static_cast<std::uint64_t>(static_cast<std::uint32_t>(base));
    key ^= 0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(member + 1);
    key ^= 0xD1B54A32D192ED03ull * static_cast<std::uint64_t>(step + 1);
    key ^= 0xCBF29CE484222325ull * static_cast<std::uint64_t>(k + 1);
    std::mt19937_64 eng(hash_seed(key));
    auto u01 = [&]() -> double { return (eng() >> 11) * (1.0/9007199254740992.0); };
    const double u1 = std::max(u01(), 1.0e-300);
    const double u2 = u01();
    m_noise_host(k) = static_cast<Real>(std::sqrt(-2.0*std::log(u1)) * std::cos(2.0*M_PI*u2));
  }
  Kokkos::deep_copy(m_noise_dev, m_noise_host);

  // Device-authoritative AR(1) update: c = alpha*c + beta*sigma*eta. Out-of-band
  // entries have sigma=0 and so remain identically zero.
  const auto c     = m_coeffs.get_view<Real*>();
  const auto sigma = m_sigma_dev;
  const auto noise = m_noise_dev;
  Kokkos::parallel_for("stoch_ar1", m_num_coeffs, KOKKOS_LAMBDA (const int k) {
    c(k) = alpha*c(k) + beta*sigma(k)*noise(k);
  });
  Kokkos::fence();

  ++m_step;
}

void SpectralPatternGenerator::evaluate_pattern (const view_1d<Real>& pattern) const
{
  EKAT_REQUIRE_MSG (m_ncols>0, "Error! [SpectralPatternGenerator] set_columns() not called.\n");
  const int N = m_params.truncation;
  const auto psi = m_coeffs.get_view<const Real*>();
  auto sin_lat = m_sin_lat;
  auto lon     = m_lon;
  Kokkos::parallel_for("stoch_eval_pattern", m_ncols, KOKKOS_LAMBDA (const int i) {
    pattern(i) = sph_eval_pattern(N, sin_lat(i), lon(i), psi);
  });
}

void SpectralPatternGenerator::
evaluate_winds (const Real a_radius, const Real cphi_floor,
                const view_1d<Real>& du, const view_1d<Real>& dv) const
{
  EKAT_REQUIRE_MSG (m_ncols>0, "Error! [SpectralPatternGenerator] set_columns() not called.\n");
  const int N = m_params.truncation;
  const auto psi = m_coeffs.get_view<const Real*>();
  auto sin_lat = m_sin_lat;
  auto lon     = m_lon;
  Kokkos::parallel_for("stoch_eval_winds", m_ncols, KOKKOS_LAMBDA (const int i) {
    Real u, v;
    sph_eval_winds(N, sin_lat(i), lon(i), psi, a_radius, cphi_floor, u, v);
    du(i) = u;
    dv(i) = v;
  });
}

} // namespace stochastic
} // namespace scream
