#ifndef SCREAM_NUDGING_HPP
#define SCREAM_NUDGING_HPP

#include "share/atm_process/atmosphere_process.hpp"

#include "share/algorithm/eamxx_time_interpolation.hpp"
#include "share/remap/abstract_remapper.hpp"

#include <ekat_parameter_list.hpp>
#include <string>
#include <map>

namespace scream
{

/*
 * The class responsible to handle the nudging of variables.
 *
 * Enhancements based on Zhang et al. (2026, GMD):
 *   - Per-variable timescales and coefficients
 *   - Unified weight function w_m(P_m, Z_m) (Eq. 4)
 *   - Horizontal/vertical Heaviside window functions
 *   - Advanced thermodynamic nudging (Tv, RH-based)
 *   - Nudging tendency diagnostic output
*/
class Nudging : public AtmosphereProcess
{
public:
  // enum to track how the source pressure levels are defined
  enum SourcePresType {
    // DEFAULT - source data should include time/spatially varying p_mid with dimensions (time, col, lev)
    TIME_DEPENDENT_3D_PROFILE,
    // source data includes p_levs which is a static set of levels in both space and time, with dimensions (lev),
    STATIC_1D_VERTICAL_PROFILE
  };

  // Constructors
  Nudging (const ekat::Comm& comm, const ekat::ParameterList& params);

  // The type of subcomponent
  AtmosphereProcessType type () const override { return AtmosphereProcessType::Physics; }

  // The name of the subcomponent
  std::string name () const override { return "Nudging"; }

  // Set the grid
  void create_requests () override;

#ifndef KOKKOS_ENABLE_CUDA
  // Cuda requires methods enclosing __device__ lambda's to be public
protected:
#endif

  void run_impl (const double dt) override;

  // Internal function to apply nudging tendency for a specific field.
  // Handles per-variable timescale/coeff, weight function, spatial windows, etc.
  void apply_tendency (const std::string& field_name,
                       Field &state, const Field &nudge, const Real dt) const;

protected:

  Field get_field_out_wrap(const std::string& field_name);

  // The two other main overrides for the subcomponent
  void initialize_impl (const RunType run_type) override;
  void finalize_impl   () override;

  // Creates an helper field, not to be shared with the AD's FieldManager
  Field create_helper_field (const std::string& name,
                            const FieldLayout& layout,
                            const std::string& grid_name,
                            const int ps = 1);

  // Retrieve a helper field
  Field get_helper_field (const std::string& name) const { return m_helper_fields.at(name); }

  // Compute precomputed weights (called from initialize_impl)
  void compute_horiz_window_weights ();
  void compute_vert_window_weights ();

  std::shared_ptr<const AbstractGrid>   m_grid;
  // Keep track of field dimensions and the iteration count
  int m_num_cols;
  int m_num_levs;
  int m_num_src_levs;

  // --- Per-variable timescales and coefficients ---
  int m_default_timescale;                          // default timescale (seconds), 0 = direct replacement
  std::map<std::string, int> m_timescales;          // per-field timescale overrides
  std::map<std::string, Real> m_nudge_coefs;        // per-field coefficient [0,1]

  bool m_use_weights;
  bool m_skip_vert_interpolation;
  std::vector<std::string> m_datafiles;
  std::string              m_static_vertical_pressure_file;
  // add nudging weights for regional nudging update
  std::string              m_weights_file;

  SourcePresType m_src_pres_type;

  std::map<std::string,Field> m_helper_fields;

  std::vector<std::string> m_fields_nudge;

  /* Nudge from coarse data */
  // if true, remap coarse data to fine grid
  bool m_refine_remap;
  // file containing coarse data mapping
  std::string m_refine_remap_file;
  // (refining) remapper object
  std::shared_ptr<scream::AbstractRemapper> m_horiz_remapper;
  // (refining) remapper vertical cutoff
  Real m_refine_remap_vert_cutoff;

  util::TimeInterpolation m_time_interp;

  // --- Unified weight function w_m (Eq. 4, Zhang et al. 2026) ---
  bool m_use_weight_function;
  Real m_wfn_p_top;                                  // Pa, top cutoff (default 100 Pa = 1 hPa)
  Real m_wfn_p0_default;                             // Pa, default upper transition pressure
  std::map<std::string, Real> m_wfn_p0;              // Pa, per-variable upper transition pressure
  Real m_wfn_z_b;                                    // m, PBL height threshold

  // --- Horizontal Heaviside window function ---
  bool m_use_horiz_window;
  Real m_hwin_lat0, m_hwin_lon0;                     // window center (degrees)
  Real m_hwin_latwidth, m_hwin_lonwidth;              // half-widths (degrees)
  Real m_hwin_latdelta, m_hwin_londelta;              // transition steepness (degrees)
  bool m_hwin_invert;
  using view_1d_host = Kokkos::View<Real*, Kokkos::HostSpace>;
  using view_1d_dev  = Kokkos::View<Real*, DefaultDevice>;
  view_1d_dev m_horiz_weight;                         // precomputed per-column weight

  // --- Vertical Heaviside window function ---
  bool m_use_vert_window;
  Real m_vwin_lindex, m_vwin_hindex;                  // transition level indices
  Real m_vwin_ldelta, m_vwin_hdelta;                  // transition steepness
  bool m_vwin_invert;
  view_1d_dev m_vert_weight;                          // precomputed per-level weight

  // --- Advanced thermodynamic nudging options ---
  int m_t_nudge_opt;   // 0=direct, 1=virtual_temp, 2=rh_adjust
  int m_q_nudge_opt;   // 0=direct, 1=rh_adjust

  // NOTE: For nudging tendency diagnostics, use the built-in compute_tendencies
  // mechanism from AtmosphereProcess base class. Configure via YAML:
  //   nudging:
  //     compute_tendencies: [T_mid, qv]
}; // class Nudging

} // namespace scream

#endif // SCREAM_NUDGING_HPP
