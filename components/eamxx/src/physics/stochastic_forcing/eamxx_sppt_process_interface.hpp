#ifndef EAMXX_SPPT_PROCESS_INTERFACE_HPP
#define EAMXX_SPPT_PROCESS_INTERFACE_HPP

#include "share/atm_process/atmosphere_process.hpp"
#include "share/stochastic/spectral_pattern_generator.hpp"

#include <ekat_parameter_list.hpp>

#include <string>

namespace scream
{

/*
 * Stochastically Perturbed Parameterization Tendencies (SPPT)
 * ===========================================================
 *
 * SPPT represents model uncertainty by perturbing the *net physics tendency*
 * over the physics step:
 *
 *     X_final = X_init + (1 + r * mu(z)) * dX_phys
 *
 * where dX_phys is the change in the prognostic field X produced by the physics
 * parameterizations, r is a smooth, mean-zero, spatially/temporally correlated
 * random pattern (one value per column, see SpectralPatternGenerator), and
 * mu(z) is a vertical taper that switches the perturbation off in the boundary
 * layer and near the model top.
 *
 * Capturing dX_phys requires the state at the *start* of the physics step. EAMxx
 * processes update the state in place, so a single process placed at the end of
 * the physics group cannot see the pre-physics state. SPPT is therefore split
 * into two cooperating processes that bracket the physics group:
 *
 *   SPPTBegin : placed FIRST in the physics group. Snapshots T_mid, horiz_winds
 *               and qv into helper fields (sppt_*_beg).
 *   SPPT      : placed LAST in the physics group. Forms dX_phys = X_now - X_beg,
 *               draws the pattern, and applies r*mu*dX_phys.
 *
 * The pattern coefficients are restart- and decomposition-BFB (see
 * SpectralPatternGenerator). Global energy conservation is obtained by enabling
 * the existing per-process energy fixer (enable_energy_fixer), which restores
 * the global total energy with a uniform temperature increment after the
 * perturbation. Water non-negativity is enforced with a clamp.
 *
 * Recommended atm_procs_list (inside the physics group), e.g.:
 *     physics::atm_procs_list = sppt_begin, mac_aero_mic, rrtmgp, sppt
 */

// -------------------------------------------------------------------------- //
// Snapshot of the pre-physics state (placed first in the physics group).
// -------------------------------------------------------------------------- //
class SPPTBegin : public AtmosphereProcess
{
public:
  SPPTBegin (const ekat::Comm& comm, const ekat::ParameterList& params)
    : AtmosphereProcess(comm, params) {}

  AtmosphereProcessType type () const override { return AtmosphereProcessType::Physics; }
  std::string name () const override { return "sppt_begin"; }

  void create_requests () override;

protected:
  void initialize_impl (const RunType /* run_type */) override {}
  void run_impl        (const double dt) override;
  void finalize_impl   () override {}

  std::shared_ptr<const AbstractGrid> m_grid;
};

// -------------------------------------------------------------------------- //
// Apply the perturbation to the net physics tendency (placed last in physics).
// -------------------------------------------------------------------------- //
class SPPT : public AtmosphereProcess
{
public:
  using KT = KokkosTypes<DefaultDevice>;
  template<typename T> using view_1d = typename KT::template view_1d<T>;

  SPPT (const ekat::Comm& comm, const ekat::ParameterList& params);

  AtmosphereProcessType type () const override { return AtmosphereProcessType::Physics; }
  std::string name () const override { return "sppt"; }

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

  // Which prognostics to perturb
  bool m_perturb_T  = true;
  bool m_perturb_uv = true;
  bool m_perturb_qv = true;

  // Perturbation controls
  Real m_clip      = 2.0;        // clamp |r| <= m_clip
  bool m_taper     = true;       // apply vertical taper
  Real m_taper_pbl = 0.0;        // boundary-layer ramp thickness [Pa]
  Real m_taper_top = 0.0;        // model-top ramp pressure [Pa]
  Real m_qv_min    = 0.0;        // lower bound enforced on qv after perturbing

  bool m_use_energy_fixer = false;
  bool m_first_step = true;

  stochastic::SpectralPatternGenerator m_pattern;
  view_1d<Real> m_r;             // pattern value per column

  static constexpr auto c_extra_data_key = "sppt_nsteps";
};

} // namespace scream

#endif // EAMXX_SPPT_PROCESS_INTERFACE_HPP
