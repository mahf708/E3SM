#ifndef EAMXX_SKEBS_PROCESS_INTERFACE_HPP
#define EAMXX_SKEBS_PROCESS_INTERFACE_HPP

#include "share/atm_process/atmosphere_process.hpp"
#include "share/stochastic/spectral_pattern_generator.hpp"

#include <ekat_parameter_list.hpp>

#include <string>

namespace scream
{

/*
 * Stochastic Kinetic Energy Backscatter Scheme (SKEBS)
 * ====================================================
 *
 * After Shutts (2005). A fraction of the kinetic energy that the model removes
 * (numerical diffusion, gravity-wave/mountain drag, convection) is scattered
 * back to the resolved scales by adding a *rotational* wind increment derived
 * from a stochastic streamfunction:
 *
 *     psi'    = streamfunction pattern (SpectralPatternGenerator)
 *     (u',v') = curl(psi') = ( -(1/a) dpsi'/dphi , (1/(a cos phi)) dpsi'/dlambda )
 *     d(u,v)  = dt * b * mu(z) * sqrt(max(D,0)) * (u',v')
 *
 * The increment is non-divergent by construction, so it conserves mass exactly
 * and does not touch tracers. b is the (empirical) backscatter coefficient that
 * absorbs the dimensional constants and sets the amplitude (tuned as in Shutts),
 * mu(z) is the vertical taper, and D is the energy-dissipation-rate estimate.
 *
 * The dissipation rate D is pluggable (dissipation_source):
 *   - "uniform"       : D = dissipation_rate (constant); robust for bring-up.
 *   - "smoothed_shear": a cheap proxy proportional to the squared vertical wind
 *                        shear; self-contained (uses only the winds).
 * Hooking real dissipation diagnostics (a "from_field" source) is a follow-on.
 *
 * Kinetic energy is *injected* by design; enabling the per-process energy fixer
 * (enable_energy_fixer) closes the global energy budget so there is no secular
 * drift over long (climate) integrations, while retaining the spatial/spectral
 * redistribution. The pattern is restart- and decomposition-BFB.
 *
 * SKEBS only needs the post-dynamics winds, so it is naturally placed at the end
 * of the physics group, e.g. physics::atm_procs_list = ..., rrtmgp, skebs.
 */

class SKEBS : public AtmosphereProcess
{
public:
  using KT = KokkosTypes<DefaultDevice>;
  template<typename T> using view_1d = typename KT::template view_1d<T>;
  template<typename T> using view_2d = typename KT::template view_2d<T>;

  enum class DissipationSource { Uniform, SmoothedShear };

  SKEBS (const ekat::Comm& comm, const ekat::ParameterList& params);

  AtmosphereProcessType type () const override { return AtmosphereProcessType::Physics; }
  std::string name () const override { return "skebs"; }

  void create_requests () override;

#ifndef KOKKOS_ENABLE_CUDA
protected:
#endif
  void run_impl (const double dt) override;

protected:
  void initialize_impl (const RunType run_type) override;
  void finalize_impl   () override {}

  std::shared_ptr<const AbstractGrid> m_grid;
  int m_num_cols = 0;
  int m_num_levs = 0;

  // Amplitude / structure controls
  Real m_backscatter = 1.0;      // empirical backscatter coefficient b
  bool m_taper       = true;
  Real m_taper_pbl   = 0.0;      // boundary-layer ramp thickness [Pa]
  Real m_taper_top   = 0.0;      // model-top ramp pressure [Pa]
  Real m_cphi_floor  = 1.0e-3;   // pole regularization for 1/cos(phi)

  // Dissipation-rate estimate
  DissipationSource m_diss_source = DissipationSource::Uniform;
  Real m_diss_uniform = 1.0e-4;  // [m^2/s^3]
  Real m_shear_coeff  = 1.0;     // proportionality for the shear proxy

  bool m_use_energy_fixer = false;
  bool m_first_step = true;

  stochastic::SpectralPatternGenerator m_pattern;
  view_1d<Real> m_du;            // rotational unit wind (zonal) per column
  view_1d<Real> m_dv;            // rotational unit wind (meridional) per column
  view_2d<Real> m_diss;          // dissipation rate D per (col,lev)

  static constexpr auto c_extra_data_key = "skebs_nsteps";
};

} // namespace scream

#endif // EAMXX_SKEBS_PROCESS_INTERFACE_HPP
