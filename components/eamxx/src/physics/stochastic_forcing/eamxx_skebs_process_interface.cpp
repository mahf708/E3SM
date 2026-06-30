#include "physics/stochastic_forcing/eamxx_skebs_process_interface.hpp"

#include "share/stochastic/stochastic_taper.hpp"
#include "share/property_checks/field_within_interval_check.hpp"
#include "share/physics/physics_constants.hpp"

#include <ekat_units.hpp>

#include <any>
#include <cstdint>

namespace scream
{

SKEBS::SKEBS (const ekat::Comm& comm, const ekat::ParameterList& params)
  : AtmosphereProcess(comm, params)
{
  m_backscatter = m_params.get<double>("backscatter_coefficient", 1.0);
  m_taper       = m_params.get<bool>("apply_vertical_taper", true);
  m_taper_pbl   = m_params.get<double>("taper_pbl_ramp", 10000.0);
  m_taper_top   = m_params.get<double>("taper_model_top_ramp", 5000.0);
  m_cphi_floor  = m_params.get<double>("pole_cos_floor", 1.0e-3);
  m_use_energy_fixer = m_params.get<bool>("enable_energy_fixer", false);

  m_diss_uniform = m_params.get<double>("dissipation_rate", 1.0e-4);
  m_shear_coeff  = m_params.get<double>("shear_dissipation_coeff", 1.0);

  const auto src = m_params.get<std::string>("dissipation_source", "uniform");
  if (src=="uniform") {
    m_diss_source = DissipationSource::Uniform;
  } else if (src=="smoothed_shear") {
    m_diss_source = DissipationSource::SmoothedShear;
  } else {
    EKAT_ERROR_MSG("Error! [SKEBS] unknown dissipation_source '" + src + "'. "
                   "Valid: 'uniform', 'smoothed_shear'.\n");
  }
}

void SKEBS::create_requests ()
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

  add_field<Updated> ("horiz_winds", vector3d_mid, m/s, gname, ps);
  add_field<Required>("p_mid",       scalar3d_mid, Pa,  gname, ps);
  add_field<Required>("ps",          scalar2d,     Pa,  gname);

  if (m_use_energy_fixer) {
    add_field<Computed>("vapor_flux", scalar2d, kg/(m2*s), gname);
    add_field<Computed>("water_flux", scalar2d, m/s,       gname);
    add_field<Computed>("ice_flux",   scalar2d, m/s,       gname);
    add_field<Computed>("heat_flux",  scalar2d, W/m2,      gname);
  }
}

void SKEBS::initialize_impl (const RunType /* run_type */)
{
  using namespace stochastic;

  SpectralPatternGenerator::Params p;
  p.truncation  = m_params.get<int>   ("spectral_truncation", 20);
  p.nmin        = m_params.get<int>   ("spectral_nmin", 1);
  p.power       = m_params.get<double>("power_exponent", 1.5);
  p.stddev      = m_params.get<double>("pattern_stddev", 1.0);
  p.decorr_time = m_params.get<double>("decorrelation_time", 21600.0);
  p.base_seed   = m_params.get<int>   ("random_seed", 0);
  p.member_id   = m_params.get<int>   ("perturbation_index", 0);

  m_pattern.initialize("skebs", m_grid, p, m_comm);
  m_pattern.set_columns(m_grid->get_geometry_data("lat"),
                        m_grid->get_geometry_data("lon"));

  add_internal_field(m_pattern.get_coefficient_field(), {"RESTART"});

  get_restart_extra_data()[c_extra_data_key] =
      std::make_shared<std::any>(std::make_any<std::int64_t>(0));

  m_du   = view_1d<Real>("skebs_du", m_num_cols);
  m_dv   = view_1d<Real>("skebs_dv", m_num_cols);
  m_diss = view_2d<Real>("skebs_D", m_num_cols, m_num_levs);
  m_first_step = true;

  if (m_use_energy_fixer) {
    get_field_out("vapor_flux").deep_copy(0);
    get_field_out("water_flux").deep_copy(0);
    get_field_out("ice_flux")  .deep_copy(0);
    get_field_out("heat_flux") .deep_copy(0);
  }

  using Interval = FieldWithinIntervalCheck;
  add_postcondition_check<Interval>(get_field_out("horiz_winds"), m_grid, -400.0, 400.0, false);
}

void SKEBS::run_impl (const double dt)
{
  using C = physics::Constants<Real>;

  // Pick up the (possibly restarted) AR(1) step count; the coefficient field is
  // device-authoritative and already holds the restart-read values.
  if (m_first_step) {
    const auto& ed = get_restart_extra_data();
    const auto it = ed.find(c_extra_data_key);
    if (it != ed.end()) {
      m_pattern.set_step_counter(std::any_cast<std::int64_t>(*it->second));
    }
    m_first_step = false;
  }

  // Evolve the streamfunction pattern and evaluate the implied rotational wind.
  m_pattern.advance(dt);
  const Real a_radius = C::r_earth.value;
  m_pattern.evaluate_winds(a_radius, m_cphi_floor, m_du, m_dv);

  const auto hw_in = get_field_out("horiz_winds").get_view<const Real***>();
  const int  ncols = m_num_cols, nlevs = m_num_levs;
  using Range2 = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  const Range2 policy({0,0},{ncols,nlevs});

  // 1) Compute the dissipation-rate estimate D(i,k) from the pre-increment state.
  const auto D = m_diss;
  if (m_diss_source==DissipationSource::Uniform) {
    Kokkos::deep_copy(m_diss, m_diss_uniform);
  } else { // SmoothedShear: crude proxy ~ squared vertical wind shear
    const Real cshear = m_shear_coeff;
    Kokkos::parallel_for("skebs_dissipation", policy, KOKKOS_LAMBDA (const int i, const int k) {
      const int kp = (k+1 < nlevs) ? k+1 : nlevs-1;
      const int km = (k-1 >= 0)    ? k-1 : 0;
      const Real du = hw_in(i,0,kp) - hw_in(i,0,km);
      const Real dv = hw_in(i,1,kp) - hw_in(i,1,km);
      D(i,k) = cshear * (du*du + dv*dv);
    });
  }

  // 2) Apply the rotational wind increment d(u,v) = dt*b*mu*sqrt(D)*(u',v').
  const auto hw  = get_field_out("horiz_winds").get_view<Real***>();
  const auto p_mid = get_field_in("p_mid").get_view<const Real**>();
  const auto ps    = get_field_in("ps").get_view<const Real*>();
  const auto du = m_du;
  const auto dv = m_dv;
  const Real b = m_backscatter;
  const bool taper = m_taper;
  const Real tp = m_taper_pbl, tt = m_taper_top;
  const Real dtr = static_cast<Real>(dt);

  Kokkos::parallel_for("skebs_apply", policy, KOKKOS_LAMBDA (const int i, const int k) {
    const Real w = taper ? stochastic::vertical_taper(p_mid(i,k), ps(i), tp, tt)
                         : static_cast<Real>(1);
    const Real Dik = D(i,k) > 0 ? D(i,k) : static_cast<Real>(0);
    const Real amp = dtr * b * w * Kokkos::sqrt(Dik);
    hw(i,0,k) += amp * du(i);
    hw(i,1,k) += amp * dv(i);
  });

  std::any_cast<std::int64_t&>(*get_restart_extra_data()[c_extra_data_key]) =
      m_pattern.step_counter();
}

} // namespace scream
