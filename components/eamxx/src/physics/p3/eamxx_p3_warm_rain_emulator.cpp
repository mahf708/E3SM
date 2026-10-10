#include "physics/p3/eamxx_p3_process_interface.hpp"

#ifdef EAMXX_HAS_PYTHON
#include "share/atm_process/atmosphere_process_pyhelpers.hpp"
#endif

#include <ekat_assert.hpp>
#include <ekat_units.hpp>

#include <array>
#include <string>

/*
 * Warm-rain emulator coupling for P3.
 *
 * Two backends are available (warm_rain_emulator_backend):
 *  - python: the model runs in a python module, on host copies of the state;
 *  - kokkos: the model (an MLP) is evaluated inside a device kernel
 *    (warm_rain_emulator/p3_warm_rain_mlp.hpp); no host round trip, no python.
 * Both use the same model contract, and the same merge into P3's rates.
 *
 * Python backend:
 * P3's warm-rain collision rates (autoconversion, droplet self-collection,
 * accretion, rain self-collection) are computed in their own stage of p3_main
 * (see impl/p3_warm_rain_impl.hpp). With small kernels, p3_main calls a host
 * hook right after that stage, outside of any kernel. Here, the hook
 *   1. gathers the grid-mean state P3 sees at that point (device kernel),
 *   2. calls the python module's forward() on it (host, numpy arrays),
 *   3. merges the emulated grid-mean rates into P3's in-cloud rates where the
 *      module's masks allow, keeping stock P3 elsewhere (device kernel).
 * All the fields exchanged with python are padded like P3's packed views, so
 * the device kernels use packs, while python sees (ncol,nlev) strided arrays.
 *
 * Python module contract (see warm_rain_emulator/p3_warm_rain_emulator.py):
 *   init(model_file)
 *   check_timestep(dt)
 *   forward(qc, nc, qr, nr, rho,                 # inputs, (ncol,nlev)
 *           qc2qr_autoconv_tend, qc2qr_accret_tend, ncautr, nc2nr_autoconv_tend,
 *           nc_accret_tend, nc_selfcollect_tend, nr_selfcollect_tend,
 *           use_cloud, use_rain)                 # outputs, written in place
 * Inputs are dry mixing ratios [kg/kg, #/kg] and dry density [kg/m3]. Outputs
 * are grid-mean rates in P3's sign conventions, and 0/1 masks.
 */

namespace scream
{

namespace {

// Emulator output fields, in the order of the python forward() arguments
constexpr std::array<const char*, 9> emu_out_names = {
  "warm_rain_emu_qc2qr_autoconv_tend", "warm_rain_emu_qc2qr_accret_tend",
  "warm_rain_emu_ncautr", "warm_rain_emu_nc2nr_autoconv_tend",
  "warm_rain_emu_nc_accret_tend", "warm_rain_emu_nc_selfcollect_tend",
  "warm_rain_emu_nr_selfcollect_tend", "warm_rain_emu_use_cloud", "warm_rain_emu_use_rain"
};
constexpr std::array<const char*, 5> emu_in_names = {
  "warm_rain_emu_qc", "warm_rain_emu_nc", "warm_rain_emu_qr", "warm_rain_emu_nr",
  "warm_rain_emu_rho"
};

} // anonymous namespace

// =========================================================================================
ekat::ParameterList P3Microphysics::
set_warm_rain_emulator_params (const ekat::ParameterList& params)
{
  auto p = params;
  if (p.get<bool>("use_warm_rain_emulator", false)) {
#ifndef SCREAM_P3_SMALL_KERNELS
    EKAT_ERROR_MSG ("[P3Microphysics] Error! use_warm_rain_emulator=true requires SCREAM_P3_SMALL_KERNELS=ON.\n"
                    "  The emulator replaces P3's warm-rain rates between kernels.\n");
#endif
    const auto backend = p.get<std::string>("warm_rain_emulator_backend", "python");
    EKAT_REQUIRE_MSG (backend=="python" or backend=="kokkos",
        "[P3Microphysics] Error! Unsupported warm_rain_emulator_backend '" + backend + "'.\n"
        "  Valid values: python, kokkos.\n");
    if (backend=="kokkos") {
      return p;
    }
#ifndef EAMXX_HAS_PYTHON
    EKAT_ERROR_MSG ("[P3Microphysics] Error! use_warm_rain_emulator=true requires EAMXX_ENABLE_PYTHON=ON.\n");
#endif
    // Default to the in-tree python module. Setting these before the AtmosphereProcess
    // constructor runs makes it import the module.
    if (p.get<std::string>("py_module_name","")=="") {
      p.set<std::string>("py_module_name", "p3_warm_rain_emulator");
    }
    if (not p.isParameter("py_module_path")) {
      p.set<std::string>("py_module_path", P3_WARM_RAIN_EMULATOR_DIR);
    }
  }
  return p;
}

// =========================================================================================
void P3Microphysics::create_warm_rain_emulator_fields ()
{
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  // The kokkos backend works on P3's own views, and needs no extra fields
  if (m_params.get<std::string>("warm_rain_emulator_backend", "python")=="kokkos") {
    return;
  }

  const auto& grid_name = m_grid->name();
  const FieldLayout layout { {COL,LEV}, {m_num_cols,m_num_levs} };

  // Padded as P3's packed views, so that device code can read them as packs
  auto add_emu_field = [&](const std::string& name, const ekat::units::Units& u) {
    Field f(FieldIdentifier(name, layout, u, grid_name));
    f.get_header().get_alloc_properties().request_allocation(Pack::n);
    f.allocate_view();
    f.deep_copy(0);
    add_internal_field(f);
  };
  const std::array<ekat::units::Units, 5> in_units = {kg/kg, 1/kg, kg/kg, 1/kg, kg/pow(m,3)};
  for (size_t i=0; i<emu_in_names.size(); ++i) {
    add_emu_field(emu_in_names[i], in_units[i]);
  }
  const std::array<ekat::units::Units, 9> out_units = {
    kg/kg/s, kg/kg/s, 1/(kg*s), 1/(kg*s), 1/(kg*s), 1/(kg*s), 1/(kg*s), none, none};
  for (size_t i=0; i<emu_out_names.size(); ++i) {
    add_emu_field(emu_out_names[i], out_units[i]);
  }
}

// =========================================================================================
void P3Microphysics::initialize_warm_rain_emulator ()
{
#ifdef SCREAM_P3_SMALL_KERNELS
  const auto model_file = m_params.get<std::string>("warm_rain_emulator_file", "");
  EKAT_REQUIRE_MSG (model_file!="" and model_file!="none",
      "[P3Microphysics] Error! use_warm_rain_emulator=true requires warm_rain_emulator_file.\n");

  if (m_params.get<std::string>("warm_rain_emulator_backend", "python")=="kokkos") {
    const auto mlp = p3::WarmRainMLP<Pack, DefaultDevice>::load(model_file);
    m_warm_rain_hook = [this, mlp](const P3F::P3Temporaries& t) {
      run_warm_rain_emulator_kokkos(mlp, t);
    };
    return;
  }
#endif

#if defined(EAMXX_HAS_PYTHON) && defined(SCREAM_P3_SMALL_KERNELS)
  EKAT_REQUIRE_MSG (has_py_module(),
      "[P3Microphysics] Error! use_warm_rain_emulator=true, but no python module was loaded.\n");
  py_module_call("init", model_file);

  m_warm_rain_hook = [this](const P3F::P3Temporaries& t) {
    run_warm_rain_emulator(t);
  };
#endif
}

#ifdef SCREAM_P3_SMALL_KERNELS
// =========================================================================================
void P3Microphysics::run_warm_rain_emulator (const P3F::P3Temporaries& temporaries)
{
#ifdef EAMXX_HAS_PYTHON
  auto view = [&](const char* name) {
    return get_internal_field(name).get_view<Pack**>();
  };
  auto cview = [&](const char* name) {
    return get_internal_field(name).get_view<const Pack**>();
  };

  // 1. Gather the emulator inputs: the state P3 sees at the warm-rain stage
  P3F::warm_rain_emulator_inputs_disp(
    m_num_cols, m_num_levs, prog_state, temporaries,
    view(emu_in_names[0]), view(emu_in_names[1]), view(emu_in_names[2]),
    view(emu_in_names[3]), view(emu_in_names[4]));
  Kokkos::fence();

  // 2. Run the emulator. Python works on host copies of the fields
  if (infrastructure.it==1) {
    py_module_call("check_timestep", static_cast<double>(infrastructure.dt));
  }
  for (const auto name : emu_in_names) {
    get_internal_field(name).sync_to_host();
  }
  const auto& out = emu_out_names;
  py_module_call("forward",
    get_py_field_host(emu_in_names[0]), get_py_field_host(emu_in_names[1]),
    get_py_field_host(emu_in_names[2]), get_py_field_host(emu_in_names[3]),
    get_py_field_host(emu_in_names[4]),
    get_py_field_host(out[0]), get_py_field_host(out[1]), get_py_field_host(out[2]),
    get_py_field_host(out[3]), get_py_field_host(out[4]), get_py_field_host(out[5]),
    get_py_field_host(out[6]), get_py_field_host(out[7]), get_py_field_host(out[8]));
  for (const auto name : emu_out_names) {
    get_internal_field(name).sync_to_dev();
  }

  // 3. Merge the emulated rates into P3's warm-rain rates
  const P3F::P3WarmRainRates<P3F::uview_2d<const Pack>> emu_rates {
    cview("warm_rain_emu_qc2qr_autoconv_tend"), cview("warm_rain_emu_nc2nr_autoconv_tend"),
    cview("warm_rain_emu_ncautr"), cview("warm_rain_emu_nc_selfcollect_tend"),
    cview("warm_rain_emu_qc2qr_accret_tend"), cview("warm_rain_emu_nc_accret_tend"),
    cview("warm_rain_emu_nr_selfcollect_tend")};
  P3F::warm_rain_emulator_merge_disp(
    m_num_cols, m_num_levs, infrastructure.dt, m_warm_rain_emulator_kk_factor,
    m_warm_rain_emulator_cloud_self_collection, prog_state, diag_inputs, temporaries,
    emu_rates, cview("warm_rain_emu_use_cloud"), cview("warm_rain_emu_use_rain"));

  // Keep the time stamps of the internal fields current
  const auto ts = end_of_step_ts();
  for (const auto name : emu_in_names) {
    get_internal_field(name).get_header().get_tracking().update_time_stamp(ts);
  }
  for (const auto name : emu_out_names) {
    get_internal_field(name).get_header().get_tracking().update_time_stamp(ts);
  }
#else
  (void) temporaries;
#endif
}

// =========================================================================================
void P3Microphysics::
run_warm_rain_emulator_kokkos (const p3::WarmRainMLP<Pack, DefaultDevice>& mlp,
                               const P3F::P3Temporaries& temporaries)
{
  using ExeSpace = typename KT::ExeSpace;

  const auto qc         = prog_state.qc;
  const auto nc         = prog_state.nc;
  const auto qr         = prog_state.qr;
  const auto nr         = prog_state.nr;
  const auto rho        = temporaries.rho;
  const auto cld_frac_l = diag_inputs.cld_frac_l;
  const auto cld_frac_r = diag_inputs.cld_frac_r;
  const auto warm_rain  = temporaries.warm_rain;
  const Real inv_dt     = 1 / infrastructure.dt;
  const Real kk_factor  = m_warm_rain_emulator_kk_factor;
  const bool do_sc      = m_warm_rain_emulator_cloud_self_collection;
  const int  nlev       = m_num_levs;

  // One fully parallel kernel: evaluate the MLP and merge, a pack of levels at a time
  Kokkos::parallel_for(
    "p3_warm_rain_emulator_kokkos",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<2>>({0, 0}, {m_num_cols, ekat::npack<Pack>(m_num_levs)}),
    KOKKOS_LAMBDA (const int i, const int k) {
    const auto range_mask = ekat::range<IntPack>(k*Pack::n) < nlev;

    Pack emu[7], use_cloud, use_rain;
    mlp.p3_rates(qc(i,k), nc(i,k), qr(i,k), nr(i,k), rho(i,k),
                 emu[0], emu[1], emu[2], emu[3], emu[4], emu[5], emu[6], use_cloud, use_rain);

    P3F::warm_rain_emulator_merge(
      cld_frac_l(i,k), cld_frac_r(i,k), nc(i,k), nr(i,k), inv_dt, kk_factor, do_sc,
      emu[0], emu[1], emu[2], emu[3], emu[4], emu[5], emu[6], use_cloud, use_rain,
      warm_rain.qc2qr_autoconv_tend(i,k), warm_rain.nc2nr_autoconv_tend(i,k),
      warm_rain.ncautr(i,k), warm_rain.nc_selfcollect_tend(i,k),
      warm_rain.qc2qr_accret_tend(i,k), warm_rain.nc_accret_tend(i,k),
      warm_rain.nr_selfcollect_tend(i,k), range_mask);
  });
}
#endif

} // namespace scream
