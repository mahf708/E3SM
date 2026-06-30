#ifndef EAMXX_SPECTRAL_PATTERN_GENERATOR_HPP
#define EAMXX_SPECTRAL_PATTERN_GENERATOR_HPP

#include "share/field/field.hpp"
#include "share/grid/abstract_grid.hpp"
#include "share/core/eamxx_types.hpp"

#include <ekat_comm.hpp>
#include <ekat_parameter_list.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace scream {
namespace stochastic {

/*
 * SpectralPatternGenerator
 *
 * Generates a smooth, mean-zero, spatially- and temporally-correlated random
 * field on the sphere, suitable as the forcing pattern for stochastic-physics
 * schemes (SPPT, SKEBS, ...). The pattern is represented as a truncated
 * spherical-harmonic expansion whose complex coefficients evolve in time as an
 * AR(1) ("red noise") process:
 *
 *     c_k(t+dt) = alpha * c_k(t) + sqrt(1-alpha^2) * sigma_n(k) * eta_k(t)
 *
 * with alpha = exp(-dt/tau) (tau = decorrelation time) and eta ~ N(0,1).
 *
 * Design goals (all critical for climate-scale, ensemble use):
 *   - DECOMPOSITION INDEPENDENT / BFB: the coefficient evolution is done on the
 *     host with a per-coefficient seed derived from (base_seed, member_id,
 *     step, mode), so every MPI rank produces *identical* coefficients, and the
 *     realized pattern is a pure function of (lat,lon). Splitting the columns
 *     across ranks cannot change any column's value.
 *   - RESTARTABLE: the coefficient state lives in a Field (see
 *     get_coefficient_field()) that the owning atm process registers in the
 *     RESTART group, plus an integer step counter the process stores in its
 *     restart_extra_data. A run chained from a restart is bit-for-bit
 *     continuous.
 *   - ENSEMBLE DISTINCT: different member_id values give statistically
 *     identical but independent realizations.
 *
 * The per-degree variance follows a power law sigma_n^2 ~ (n(n+1))^(-p),
 * band-limited to [nmin, N], normalized so that the realized pattern has the
 * requested stationary standard deviation.
 *
 * This class is NOT an AtmosphereProcess; it is a reusable utility owned by the
 * stochastic-forcing atm processes.
 */

class SpectralPatternGenerator
{
public:
  using KT = KokkosTypes<DefaultDevice>;
  template<typename T> using view_1d = typename KT::template view_1d<T>;

  struct Params {
    int    truncation   = 20;       // total wavenumber cutoff N
    int    nmin         = 1;        // smallest total wavenumber retained (>=1)
    Real   power        = 1.5;      // exponent p in sigma_n^2 ~ (n(n+1))^(-p)
    Real   stddev       = 1.0;      // target stationary std of the realized pattern
    Real   decorr_time  = 21600.0;  // tau [s]
    int    base_seed    = 0;        // master seed
    int    member_id    = 0;        // ensemble member (perturbation index)
  };

  SpectralPatternGenerator () = default;

  // Build mode tables / sigma_n, and create the (replicated) coefficient field.
  // name_prefix makes the field + its dimension name unique across schemes
  // (e.g. "sppt", "skebs") to avoid IO dimension-name collisions.
  void initialize (const std::string& name_prefix,
                   const std::shared_ptr<const AbstractGrid>& grid,
                   const Params& params,
                   const ekat::Comm& comm);

  // Cache the column coordinates used by evaluate_*(). lat/lon are the grid
  // geometry data, in DEGREES (as provided by EAMxx grids).
  void set_columns (const Field& lat_deg, const Field& lon_deg);

  // The coefficient Field; the owning process must add it to the RESTART group
  // via add_internal_field(get_coefficient_field(), {"RESTART"}).
  const Field& get_coefficient_field () const { return m_coeffs; }

  int  num_coefficients () const { return m_num_coeffs; }
  int  truncation () const { return m_params.truncation; }

  // Step-counter access (the process persists this via restart_extra_data).
  std::int64_t step_counter () const { return m_step; }
  void set_step_counter (const std::int64_t n) { m_step = n; }

  // Advance the AR(1) coefficients by one step of size dt. The tiny Gaussian
  // noise vector is drawn on the host (reproducibly), copied to the device, and
  // the coefficient update is done in a device kernel so the coefficient state
  // (the restart-backed field) is always device-authoritative -- which is the
  // state EAMxx restart leaves valid. Increments the step counter.
  void advance (const double dt);

  // Evaluate the scalar pattern at every cached column -> pattern(ncols).
  void evaluate_pattern (const view_1d<Real>& pattern) const;

  // Evaluate the rotational wind from the pattern-as-streamfunction at every
  // cached column. a_radius = sphere radius [m]; the pattern is interpreted as
  // a streamfunction with units of m^2/s when scaled by the caller.
  void evaluate_winds (const Real a_radius, const Real cphi_floor,
                       const view_1d<Real>& du, const view_1d<Real>& dv) const;

private:
  // 64-bit mix (splitmix64) used to build an independent seed per coefficient.
  static std::uint64_t hash_seed (std::uint64_t x);

  Params      m_params;
  ekat::Comm  m_comm;
  std::string m_prefix;

  int m_num_coeffs = 0;          // padded coefficient count = 2*(N+1)^2
  std::vector<Real> m_sigma;     // per-degree stddev, size N+1 (0 outside band)

  Field m_coeffs;                // coefficient state (replicated, RESTART, device-authoritative)

  view_1d<Real> m_sigma_dev;     // per-coefficient stddev (0 for unused/out-of-band entries)
  view_1d<Real> m_noise_dev;     // per-coefficient Gaussian noise (device scratch)
  typename view_1d<Real>::HostMirror m_noise_host;

  int          m_ncols = 0;
  view_1d<Real> m_sin_lat;       // sin(latitude), per column
  view_1d<Real> m_lon;           // longitude in radians, per column

  std::int64_t m_step = 0;
};

} // namespace stochastic
} // namespace scream

#endif // EAMXX_SPECTRAL_PATTERN_GENERATOR_HPP
