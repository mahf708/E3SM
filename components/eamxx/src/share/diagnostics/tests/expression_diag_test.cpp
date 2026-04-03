#include <catch2/catch.hpp>

#include "share/diagnostics/register_diagnostics.hpp"
#include "share/diagnostics/expression_diag.hpp"
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

// ============================================================
// PARSER UNIT TESTS
// ============================================================

TEST_CASE("expression_parser") {
  using scream::parse_expression;

  SECTION("simple_addition") {
    auto result = parse_expression("a + b");
    REQUIRE(result.field_names.size() == 2);
    REQUIRE(result.field_names[0] == "a");
    REQUIRE(result.field_names[1] == "b");
    REQUIRE(result.program.size() == 3);  // push a, push b, add
    REQUIRE(result.program[0].op == ExprOp::PushField);
    REQUIRE(result.program[1].op == ExprOp::PushField);
    REQUIRE(result.program[2].op == ExprOp::Add);
  }

  SECTION("operator_precedence") {
    // a + b * c  should parse as  a + (b * c)
    auto result = parse_expression("a + b * c");
    REQUIRE(result.field_names.size() == 3);
    // RPN: a b c * +
    REQUIRE(result.program.size() == 5);
    REQUIRE(result.program[0].op == ExprOp::PushField);  // a
    REQUIRE(result.program[1].op == ExprOp::PushField);  // b
    REQUIRE(result.program[2].op == ExprOp::PushField);  // c
    REQUIRE(result.program[3].op == ExprOp::Mul);
    REQUIRE(result.program[4].op == ExprOp::Add);
  }

  SECTION("parentheses") {
    // (a + b) * c
    auto result = parse_expression("(a + b) * c");
    // RPN: a b + c *
    REQUIRE(result.program.size() == 5);
    REQUIRE(result.program[2].op == ExprOp::Add);
    REQUIRE(result.program[4].op == ExprOp::Mul);
  }

  SECTION("unary_minus") {
    auto result = parse_expression("-a");
    REQUIRE(result.program.size() == 2);
    REQUIRE(result.program[0].op == ExprOp::PushField);
    REQUIRE(result.program[1].op == ExprOp::Neg);
  }

  SECTION("function_call") {
    auto result = parse_expression("sqrt(a)");
    REQUIRE(result.program.size() == 2);
    REQUIRE(result.program[0].op == ExprOp::PushField);
    REQUIRE(result.program[1].op == ExprOp::Sqrt);
  }

  SECTION("nested_functions") {
    auto result = parse_expression("sqrt(abs(a))");
    REQUIRE(result.program.size() == 3);
    REQUIRE(result.program[0].op == ExprOp::PushField);
    REQUIRE(result.program[1].op == ExprOp::Abs);
    REQUIRE(result.program[2].op == ExprOp::Sqrt);
  }

  SECTION("numeric_literal") {
    auto result = parse_expression("a + 3.14");
    REQUIRE(result.field_names.size() == 1);
    REQUIRE(result.program.size() == 3);
    REQUIRE(result.program[1].op == ExprOp::PushConst);
    REQUIRE(result.program[1].const_val == Approx(3.14));
  }

  SECTION("physics_constant") {
    auto result = parse_expression("a / gravit");
    REQUIRE(result.field_names.size() == 1);
    REQUIRE(result.field_names[0] == "a");
    REQUIRE(result.program.size() == 3);
    REQUIRE(result.program[1].op == ExprOp::PushConst);
    REQUIRE(result.program[1].const_val ==
            Approx(physics::Constants<Real>::gravit.value));
  }

  SECTION("power_operator") {
    auto result = parse_expression("a ** 2");
    REQUIRE(result.program.size() == 3);
    REQUIRE(result.program[2].op == ExprOp::Pow);
  }

  SECTION("complex_expression") {
    // sqrt(u**2 + v**2) - like wind speed
    auto result = parse_expression("sqrt(u**2 + v**2)");
    REQUIRE(result.field_names.size() == 2);
    REQUIRE(result.field_names[0] == "u");
    REQUIRE(result.field_names[1] == "v");
    // RPN: u 2 ** v 2 ** + sqrt
    REQUIRE(result.program.size() == 8);
  }

  SECTION("field_reuse") {
    // Same field used twice should only appear once in field_names
    auto result = parse_expression("a * a");
    REQUIRE(result.field_names.size() == 1);
    REQUIRE(result.program[0].operand_idx == 0);
    REQUIRE(result.program[1].operand_idx == 0);
  }

  SECTION("scientific_notation") {
    auto result = parse_expression("a * 1.5e-3");
    REQUIRE(result.program[1].op == ExprOp::PushConst);
    REQUIRE(result.program[1].const_val == Approx(1.5e-3));
  }
}

// ============================================================
// FULL DIAGNOSTIC TESTS
// ============================================================

TEST_CASE("expression_diag") {
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  ekat::Comm comm(MPI_COMM_WORLD);
  util::TimeStamp t0({2024, 1, 1}, {0, 0, 0});

  constexpr int nlevs = 72;
  const int ngcols    = 64 * comm.size();

  auto gm   = create_gm(comm, ngcols, nlevs);
  auto grid = gm->get_grid("physics");

  auto &diag_factory = AtmosphereDiagnosticFactory::instance();
  register_diagnostics();

  FieldLayout scalar2d_layout{{COL, LEV}, {ngcols, nlevs}};

  int seed = get_random_test_seed();

  SECTION("simple_addition") {
    FieldIdentifier a_fid("a", scalar2d_layout, Units::nondimensional(), grid->name());
    FieldIdentifier b_fid("b", scalar2d_layout, Units::nondimensional(), grid->name());
    Field a(a_fid, true); Field b(b_fid, true);
    a.get_header().get_tracking().update_time_stamp(t0);
    b.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(a, seed++, 0.0, 100.0);
    randomize_uniform(b, seed++, 0.0, 100.0);

    ekat::ParameterList params("expr_a+b");
    params.set("grid_name", grid->name());
    params.set<std::string>("expression", "a + b");
    auto diag = diag_factory.create("ExpressionDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(a);
    diag->set_required_field(b);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();
    a.sync_to_host(); b.sync_to_host();

    const auto &r_v = result.get_view<Real**, Host>();
    const auto &a_v = a.get_view<Real**, Host>();
    const auto &b_v = b.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        REQUIRE(r_v(icol, ilev) == Approx(a_v(icol, ilev) + b_v(icol, ilev)));
      }
    }
  }

  SECTION("field_times_constant") {
    FieldIdentifier q_fid("qc", scalar2d_layout, Units::nondimensional(), grid->name());
    Field qc(q_fid, true);
    qc.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(qc, seed++, 0.0, 1.0);

    const Real g = physics::Constants<Real>::gravit.value;

    ekat::ParameterList params("expr_qc/gravit");
    params.set("grid_name", grid->name());
    params.set<std::string>("expression", "qc / gravit");
    auto diag = diag_factory.create("ExpressionDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(qc);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();
    qc.sync_to_host();

    const auto &r_v  = result.get_view<Real**, Host>();
    const auto &qc_v = qc.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        REQUIRE(r_v(icol, ilev) == Approx(qc_v(icol, ilev) / g).epsilon(1e-12));
      }
    }
  }

  SECTION("wind_speed_expression") {
    // Replicate the WindSpeed diagnostic: sqrt(u**2 + v**2)
    FieldIdentifier u_fid("u", scalar2d_layout, Units::nondimensional(), grid->name());
    FieldIdentifier v_fid("v", scalar2d_layout, Units::nondimensional(), grid->name());
    Field u(u_fid, true); Field v(v_fid, true);
    u.get_header().get_tracking().update_time_stamp(t0);
    v.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(u, seed++, -50.0, 50.0);
    randomize_uniform(v, seed++, -50.0, 50.0);

    ekat::ParameterList params("expr_sqrt(u**2+v**2)");
    params.set("grid_name", grid->name());
    params.set<std::string>("expression", "sqrt(u**2 + v**2)");
    auto diag = diag_factory.create("ExpressionDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(u);
    diag->set_required_field(v);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();
    u.sync_to_host(); v.sync_to_host();

    const auto &r_v = result.get_view<Real**, Host>();
    const auto &u_v = u.get_view<Real**, Host>();
    const auto &v_v = v.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        Real expected = std::sqrt(u_v(icol,ilev)*u_v(icol,ilev) +
                                  v_v(icol,ilev)*v_v(icol,ilev));
        REQUIRE(r_v(icol, ilev) == Approx(expected).epsilon(1e-12));
      }
    }
  }

  SECTION("multi_field_expression") {
    // a * b - c / 2.0
    FieldIdentifier a_fid("a", scalar2d_layout, Units::nondimensional(), grid->name());
    FieldIdentifier b_fid("b", scalar2d_layout, Units::nondimensional(), grid->name());
    FieldIdentifier c_fid("c", scalar2d_layout, Units::nondimensional(), grid->name());
    Field a(a_fid, true); Field b(b_fid, true); Field c(c_fid, true);
    a.get_header().get_tracking().update_time_stamp(t0);
    b.get_header().get_tracking().update_time_stamp(t0);
    c.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(a, seed++, 1.0, 10.0);
    randomize_uniform(b, seed++, 1.0, 10.0);
    randomize_uniform(c, seed++, 1.0, 10.0);

    ekat::ParameterList params("expr_a*b-c/2");
    params.set("grid_name", grid->name());
    params.set<std::string>("expression", "a * b - c / 2.0");
    auto diag = diag_factory.create("ExpressionDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(a);
    diag->set_required_field(b);
    diag->set_required_field(c);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();
    a.sync_to_host(); b.sync_to_host(); c.sync_to_host();

    const auto &r_v = result.get_view<Real**, Host>();
    const auto &a_v = a.get_view<Real**, Host>();
    const auto &b_v = b.get_view<Real**, Host>();
    const auto &c_v = c.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        Real expected = a_v(icol,ilev) * b_v(icol,ilev) -
                        c_v(icol,ilev) / 2.0;
        REQUIRE(r_v(icol, ilev) == Approx(expected).epsilon(1e-12));
      }
    }
  }

  SECTION("unary_minus_expression") {
    FieldIdentifier a_fid("a", scalar2d_layout, Units::nondimensional(), grid->name());
    Field a(a_fid, true);
    a.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(a, seed++, 1.0, 100.0);

    ekat::ParameterList params("expr_neg");
    params.set("grid_name", grid->name());
    params.set<std::string>("expression", "-a");
    auto diag = diag_factory.create("ExpressionDiag", comm, params);

    diag->set_grids(gm);
    diag->set_required_field(a);
    diag->initialize(t0, RunType::Initial);
    diag->compute_diagnostic();

    auto result = diag->get_diagnostic();
    result.sync_to_host();
    a.sync_to_host();

    const auto &r_v = result.get_view<Real**, Host>();
    const auto &a_v = a.get_view<Real**, Host>();
    for (int icol = 0; icol < ngcols; ++icol) {
      for (int ilev = 0; ilev < nlevs; ++ilev) {
        REQUIRE(r_v(icol, ilev) == -a_v(icol, ilev));
      }
    }
  }

  SECTION("invalid_expression_throws") {
    ekat::ParameterList params("bad");
    params.set("grid_name", grid->name());
    params.set<std::string>("expression", "a + + b");
    REQUIRE_THROWS(diag_factory.create("ExpressionDiag", comm, params));
  }

  SECTION("empty_expression_throws") {
    ekat::ParameterList params("bad");
    params.set("grid_name", grid->name());
    params.set<std::string>("expression", "");
    REQUIRE_THROWS(diag_factory.create("ExpressionDiag", comm, params));
  }
}

}  // namespace scream
