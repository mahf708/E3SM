#include <catch2/catch.hpp>

#include "share/diagnostics/register_diagnostics.hpp"
#include "share/physics/physics_constants.hpp"
#include "share/field/field_utils.hpp"
#include "share/data_managers/mesh_free_grids_manager.hpp"
#include "share/core/eamxx_setup_random_test.hpp"

#include <cmath>

namespace scream {

std::shared_ptr<GridsManager> create_gm(const ekat::Comm &comm, const int ncols,
                                        const int nlevs) {
  const int num_global_cols = ncols * comm.size();

  using vos_t = std::vector<std::string>;
  ekat::ParameterList gm_params;
  gm_params.set("grids_names", vos_t{"point_grid"});
  auto &pl = gm_params.sublist("point_grid");
  pl.set<std::string>("type", "point_grid");
  pl.set("aliases", vos_t{"physics"});
  pl.set<int>("number_of_global_columns", num_global_cols);
  pl.set<int>("number_of_vertical_levels", nlevs);

  auto gm = create_mesh_free_grids_manager(comm, gm_params);
  gm->build_grids();

  return gm;
}

TEST_CASE("unary_ops") {
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  ekat::Comm comm(MPI_COMM_WORLD);

  util::TimeStamp t0({2024, 1, 1}, {0, 0, 0});

  constexpr int nlevs = 72;
  const int ngcols    = 128 * comm.size();

  auto gm   = create_gm(comm, ngcols, nlevs);
  auto grid = gm->get_grid("physics");

  // Create input fields with positive values (needed for sqrt, log)
  FieldLayout scalar2d_layout{{COL, LEV}, {ngcols, nlevs}};
  FieldIdentifier f_fid("test_field", scalar2d_layout, Units::nondimensional(), grid->name());
  Field f(f_fid, true);

  int seed = get_random_test_seed();
  f.get_header().get_tracking().update_time_stamp(t0);
  randomize_uniform(f, seed++, 1.0, 100.0);  // positive values for sqrt/log

  // Register diagnostics
  auto &diag_factory = AtmosphereDiagnosticFactory::instance();
  register_diagnostics();

  // --- Test parameter validation ---
  SECTION("invalid_op_throws") {
    ekat::ParameterList bad_params("bad");
    bad_params.set("grid_name", grid->name());
    bad_params.set<std::string>("arg", "test_field");
    bad_params.set<std::string>("unary_op", "invalid_op");
    REQUIRE_THROWS(diag_factory.create("UnaryOpsDiag", comm, bad_params));
  }

  SECTION("missing_params_throws") {
    ekat::ParameterList bad_params("bad");
    REQUIRE_THROWS(diag_factory.create("UnaryOpsDiag", comm, bad_params));
  }

  // --- Test sqrt ---
  SECTION("sqrt_field") {
    ekat::ParameterList params("sqrt_test");
    params.set("grid_name", grid->name());
    params.set<std::string>("arg", "test_field");
    params.set<std::string>("unary_op", "sqrt");
    auto diag = diag_factory.create("UnaryOpsDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(f);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();

    const auto &res_v = result.get_view<Real**, Host>();
    const auto &f_v   = f.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        REQUIRE(res_v(icol, ilev) == Approx(std::sqrt(f_v(icol, ilev))).epsilon(1e-10));
      }
    }
  }

  // --- Test abs ---
  SECTION("abs_field") {
    // Create a field with negative values
    FieldIdentifier neg_fid("neg_field", scalar2d_layout, Units::nondimensional(), grid->name());
    Field neg_f(neg_fid, true);
    neg_f.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(neg_f, seed++, -100.0, 100.0);

    ekat::ParameterList params("abs_test");
    params.set("grid_name", grid->name());
    params.set<std::string>("arg", "neg_field");
    params.set<std::string>("unary_op", "abs");
    auto diag = diag_factory.create("UnaryOpsDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(neg_f);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();

    const auto &res_v = result.get_view<Real**, Host>();
    const auto &neg_v = neg_f.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        REQUIRE(res_v(icol, ilev) == std::fabs(neg_v(icol, ilev)));
      }
    }
  }

  // --- Test square ---
  SECTION("square_field") {
    ekat::ParameterList params("square_test");
    params.set("grid_name", grid->name());
    params.set<std::string>("arg", "test_field");
    params.set<std::string>("unary_op", "square");
    auto diag = diag_factory.create("UnaryOpsDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(f);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();

    const auto &res_v = result.get_view<Real**, Host>();
    const auto &f_v   = f.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        REQUIRE(res_v(icol, ilev) == f_v(icol, ilev) * f_v(icol, ilev));
      }
    }
  }

  // --- Test exp ---
  SECTION("exp_field") {
    // Use small nondimensional values for exp to avoid overflow
    FieldIdentifier small_fid("small_field", scalar2d_layout,
                              Units::nondimensional(), grid->name());
    Field small_f(small_fid, true);
    small_f.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(small_f, seed++, 0.0, 5.0);

    ekat::ParameterList params("exp_test");
    params.set("grid_name", grid->name());
    params.set<std::string>("arg", "small_field");
    params.set<std::string>("unary_op", "exp");
    auto diag = diag_factory.create("UnaryOpsDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(small_f);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();

    const auto &res_v = result.get_view<Real**, Host>();
    const auto &sm_v  = small_f.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        REQUIRE(res_v(icol, ilev) == Approx(std::exp(sm_v(icol, ilev))).epsilon(1e-10));
      }
    }
  }

  // --- Test log ---
  SECTION("log_field") {
    // log requires nondimensional units
    FieldIdentifier nd_fid("nd_field", scalar2d_layout,
                           Units::nondimensional(), grid->name());
    Field nd_f(nd_fid, true);
    nd_f.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(nd_f, seed++, 1.0, 100.0);

    ekat::ParameterList params("log_test");
    params.set("grid_name", grid->name());
    params.set<std::string>("arg", "nd_field");
    params.set<std::string>("unary_op", "log");
    auto diag = diag_factory.create("UnaryOpsDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(nd_f);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();

    const auto &res_v = result.get_view<Real**, Host>();
    const auto &nd_v  = nd_f.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        REQUIRE(res_v(icol, ilev) == Approx(std::log(nd_v(icol, ilev))).epsilon(1e-10));
      }
    }
  }

  // --- Test scalar constant input (nondimensional) ---
  SECTION("sqrt_scalar_constant") {
    ekat::ParameterList params("sqrt_o2mmr");
    params.set("grid_name", grid->name());
    params.set<std::string>("arg", "o2mmr");
    params.set<std::string>("unary_op", "sqrt");
    auto diag = diag_factory.create("UnaryOpsDiag", comm, params);

    diag->set_grids(gm);
    diag->initialize(t0, RunType::Initial);
    // Constant diags are pre-computed in initialize

    auto result = diag->get_diagnostic();
    result.sync_to_host();

    const auto &res_v = result.get_view<Real, Host>();
    const Real o2mmr = physics::Constants<Real>::o2mmr.value;
    REQUIRE(res_v() == Approx(std::sqrt(o2mmr)).epsilon(1e-10));
  }

  // --- Test square with dimensional units ---
  SECTION("square_dimensional") {
    FieldIdentifier dim_fid("dim_field", scalar2d_layout, m, grid->name());
    Field dim_f(dim_fid, true);
    dim_f.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(dim_f, seed++, 1.0, 50.0);

    ekat::ParameterList params("square_dim");
    params.set("grid_name", grid->name());
    params.set<std::string>("arg", "dim_field");
    params.set<std::string>("unary_op", "square");
    auto diag = diag_factory.create("UnaryOpsDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(dim_f);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();

    const auto &res_v = result.get_view<Real**, Host>();
    const auto &d_v   = dim_f.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        REQUIRE(res_v(icol, ilev) == d_v(icol, ilev) * d_v(icol, ilev));
      }
    }
    // Check that output units are m^2
    const auto& out_units = result.get_header().get_identifier().get_units();
    REQUIRE(out_units == m * m);
  }
}

}  // namespace scream
