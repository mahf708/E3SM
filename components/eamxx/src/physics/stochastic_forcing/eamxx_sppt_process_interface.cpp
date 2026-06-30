#include "physics/stochastic_forcing/eamxx_sppt_process_interface.hpp"

#include "share/stochastic/stochastic_taper.hpp"
#include "share/property_checks/field_within_interval_check.hpp"

#include <ekat_units.hpp>

#include <algorithm>
#include <any>
#include <cstdint>

namespace scream
{

// ========================================================================== //
//                                SPPTBegin                                   //
// ========================================================================== //
void SPPTBegin::create_requests ()
{
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  m_grid = m_grids_manager->get_grid("physics");
  const auto& gname = m_grid->name();

  const auto scalar3d_mid = m_grid->get_3d_scalar_layout(LEV);
  const auto vector3d_mid = m_grid->get_3d_vector_layout(LEV, 2);
  constexpr int ps = SCREAM_PACK_SIZE;

  // Inputs: the pre-physics prognostic state
  add_field<Required>("T_mid",       scalar3d_mid, K,     gname, ps);
  add_field<Required>("horiz_winds", vector3d_mid, m/s,   gname, ps);
  add_tracer<Required>("qv", m_grid, kg/kg, ps);

  // Outputs: snapshots consumed by the SPPT (apply) process
  add_field<Computed>("sppt_T_mid_beg",       scalar3d_mid, K,     gname, ps);
  add_field<Computed>("sppt_horiz_winds_beg", vector3d_mid, m/s,   gname, ps);
  add_field<Computed>("sppt_qv_beg",          scalar3d_mid, kg/kg, gname, ps);
}

void SPPTBegin::run_impl (const double /* dt */)
{
  // Snapshot the state at the start of the physics group.
  get_field_out("sppt_T_mid_beg")      .deep_copy(get_field_in("T_mid"));
  get_field_out("sppt_horiz_winds_beg").deep_copy(get_field_in("horiz_winds"));
  get_field_out("sppt_qv_beg")         .deep_copy(get_field_in("qv"));
}

// ========================================================================== //
//                                   SPPT                                     //
// ========================================================================== //
SPPT::SPPT (const ekat::Comm& comm, const ekat::ParameterList& params)
  : AtmosphereProcess(comm, params)
{
  m_clip      = m_params.get<double>("clip_magnitude", 2.0);
  m_taper     = m_params.get<bool>("apply_vertical_taper", true);
  m_taper_pbl = m_params.get<double>("taper_pbl_ramp", 10000.0);     // Pa
  m_taper_top = m_params.get<double>("taper_model_top_ramp", 5000.0); // Pa
  m_qv_min    = m_params.get<double>("qv_min", 0.0);
  m_use_energy_fixer = m_params.get<bool>("enable_energy_fixer", false);

  const auto pf = m_params.get<std::vector<std::string>>(
      "perturbed_fields", {"T_mid","horiz_winds","qv"});
  auto has = [&](const std::string& s) {
    return std::find(pf.begin(), pf.end(), s) != pf.end();
  };
  m_perturb_T  = has("T_mid");
  m_perturb_uv = has("horiz_winds");
  m_perturb_qv = has("qv");
}

void SPPT::create_requests ()
{
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  m_grid     = m_grids_manager->get_grid("physics");
  m_num_cols = m_grid->get_num_local_dofs();
  m_num_levs = m_grid->get_num_vertical_levels();
  const auto& gname = m_grid->name();

  const auto scalar2d     = m_grid->get_2d_scalar_layout();
  const auto scalar3d_mid = m_grid->get_3d_scalar_layout(LEV);
  const auto vector3d_mid = m_grid->get_3d_vector_layout(LEV, 2);
  constexpr int ps = SCREAM_PACK_SIZE;

  // Perturbed prognostics + their pre-physics snapshots
  if (m_perturb_T) {
    add_field<Updated> ("T_mid",         scalar3d_mid, K, gname, ps);
    add_field<Required>("sppt_T_mid_beg", scalar3d_mid, K, gname, ps);
  }
  if (m_perturb_uv) {
    add_field<Updated> ("horiz_winds",         vector3d_mid, m/s, gname, ps);
    add_field<Required>("sppt_horiz_winds_beg", vector3d_mid, m/s, gname, ps);
  }
  if (m_perturb_qv) {
    add_tracer<Updated>("qv", m_grid, kg/kg, ps);
    add_field<Required>("sppt_qv_beg", scalar3d_mid, kg/kg, gname, ps);
  }

  // Inputs for the vertical taper
  add_field<Required>("p_mid", scalar3d_mid, Pa, gname, ps);
  add_field<Required>("ps",    scalar2d,     Pa, gname);

  // Boundary-flux fields required when this process runs the global energy
  // fixer. There are no physical surface fluxes associated with a stochastic
  // perturbation, so they are identically zero.
  if (m_use_energy_fixer) {
    add_field<Computed>("vapor_flux", scalar2d, kg/(m2*s), gname);
    add_field<Computed>("water_flux", scalar2d, m/s,       gname);
    add_field<Computed>("ice_flux",   scalar2d, m/s,       gname);
    add_field<Computed>("heat_flux",  scalar2d, W/m2,      gname);
  }
}

void SPPT::initialize_impl (const RunType /* run_type */)
{
  using namespace stochastic;

  SpectralPatternGenerator::Params p;
  p.truncation  = m_params.get<int>   ("spectral_truncation", 20);
  p.nmin        = m_params.get<int>   ("spectral_nmin", 1);
  p.power       = m_params.get<double>("power_exponent", 1.5);
  p.stddev      = m_params.get<double>("pattern_stddev", 0.5);
  p.decorr_time = m_params.get<double>("decorrelation_time", 21600.0);
  p.base_seed   = m_params.get<int>   ("random_seed", 0);
  p.member_id   = m_params.get<int>   ("perturbation_index", 0);

  m_pattern.initialize("sppt", m_grid, p, m_comm);
  m_pattern.set_columns(m_grid->get_geometry_data("lat"),
                        m_grid->get_geometry_data("lon"));

  // Register the spectral coefficients for restart (BFB across restarts).
  add_internal_field(m_pattern.get_coefficient_field(), {"RESTART"});

  // Persist the AR(1) step counter across restarts.
  get_restart_extra_data()[c_extra_data_key] =
      std::make_shared<std::any>(std::make_any<std::int64_t>(0));

  m_r = view_1d<Real>("sppt_pattern_r", m_num_cols);
  m_first_step = true;

  if (m_use_energy_fixer) {
    get_field_out("vapor_flux").deep_copy(0);
    get_field_out("water_flux").deep_copy(0);
    get_field_out("ice_flux")  .deep_copy(0);
    get_field_out("heat_flux") .deep_copy(0);
  }

  // Safety net: keep qv non-negative (repairable).
  if (m_perturb_qv) {
    using Interval = FieldWithinIntervalCheck;
    add_postcondition_check<Interval>(get_field_out("qv"), m_grid, 0.0, 1.0, true);
  }
}

void SPPT::run_impl (const double dt)
{
  // On the first step, pick up the (possibly restarted) AR(1) step count. The
  // coefficient field itself is device-authoritative and already holds the
  // restart-read values, so nothing else needs syncing.
  if (m_first_step) {
    const auto& ed = get_restart_extra_data();
    const auto it = ed.find(c_extra_data_key);
    if (it != ed.end()) {
      m_pattern.set_step_counter(std::any_cast<std::int64_t>(*it->second));
    }
    m_first_step = false;
  }

  // Evolve and evaluate the random pattern (one value per column).
  m_pattern.advance(dt);
  m_pattern.evaluate_pattern(m_r);

  const auto r     = m_r;
  const auto p_mid = get_field_in("p_mid").get_view<const Real**>();
  const auto ps    = get_field_in("ps").get_view<const Real*>();

  const Real clip = m_clip;
  const bool taper = m_taper;
  const Real tp = m_taper_pbl, tt = m_taper_top;
  const int  ncols = m_num_cols, nlevs = m_num_levs;

  using Range2 = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  const Range2 policy({0,0},{ncols,nlevs});

  // weight(i,k) = taper(z) * clamp(r). Inlined into each kernel (nested device
  // lambdas are not portable across CUDA/HIP).
  if (m_perturb_T) {
    const auto T  = get_field_out("T_mid").get_view<Real**>();
    const auto Tb = get_field_in("sppt_T_mid_beg").get_view<const Real**>();
    Kokkos::parallel_for("sppt_apply_T", policy, KOKKOS_LAMBDA (const int i, const int k) {
      Real ri = r(i); ri = ri>clip?clip:(ri<-clip?-clip:ri);
      const Real w = taper ? stochastic::vertical_taper(p_mid(i,k), ps(i), tp, tt) : static_cast<Real>(1);
      T(i,k) += w*ri * (T(i,k) - Tb(i,k));
    });
  }

  if (m_perturb_uv) {
    const auto hw = get_field_out("horiz_winds").get_view<Real***>();
    const auto hb = get_field_in("sppt_horiz_winds_beg").get_view<const Real***>();
    Kokkos::parallel_for("sppt_apply_uv", policy, KOKKOS_LAMBDA (const int i, const int k) {
      Real ri = r(i); ri = ri>clip?clip:(ri<-clip?-clip:ri);
      const Real w = taper ? stochastic::vertical_taper(p_mid(i,k), ps(i), tp, tt) : static_cast<Real>(1);
      const Real f = w*ri;
      hw(i,0,k) += f * (hw(i,0,k) - hb(i,0,k));
      hw(i,1,k) += f * (hw(i,1,k) - hb(i,1,k));
    });
  }

  if (m_perturb_qv) {
    const auto qv     = get_field_out("qv").get_view<Real**>();
    const auto qb     = get_field_in("sppt_qv_beg").get_view<const Real**>();
    const Real qv_min = m_qv_min;
    Kokkos::parallel_for("sppt_apply_qv", policy, KOKKOS_LAMBDA (const int i, const int k) {
      Real ri = r(i); ri = ri>clip?clip:(ri<-clip?-clip:ri);
      const Real w = taper ? stochastic::vertical_taper(p_mid(i,k), ps(i), tp, tt) : static_cast<Real>(1);
      const Real q = qv(i,k) + w*ri * (qv(i,k) - qb(i,k));
      qv(i,k) = q < qv_min ? qv_min : q;
    });
  }

  // Persist the updated step counter for restart.
  std::any_cast<std::int64_t&>(*get_restart_extra_data()[c_extra_data_key]) =
      m_pattern.step_counter();
}

} // namespace scream
