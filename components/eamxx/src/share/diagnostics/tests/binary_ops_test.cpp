#include <catch2/catch.hpp>

#include "share/diagnostics/register_diagnostics.hpp"
#include "share/physics/physics_constants.hpp"
#include "share/field/field_utils.hpp"
#include "share/grid/point_grid.hpp"
#include "share/core/eamxx_setup_random_test.hpp"
#include "share/util/eamxx_universal_constants.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace scream {


TEST_CASE("binary_ops") {
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  // A world comm
  ekat::Comm comm(MPI_COMM_WORLD);

  // A time stamp
  util::TimeStamp t0({2024, 1, 1}, {0, 0, 0});

  // Create a grids manager - single column for these tests
  const int nlevs = 201;
  const int ncols = 260;

  auto grid = create_point_grid("physics",ncols,nlevs,comm);

  // Input (randomized) qc, qv
  FieldLayout scalar2d_layout = grid->get_3d_scalar_layout(LEV);
  FieldIdentifier qc_fid("qc", scalar2d_layout, kg / kg, grid->name());
  FieldIdentifier qv_fid("qv", scalar2d_layout, kg / kg, grid->name());
  FieldIdentifier m_fid("m", scalar2d_layout, kg, grid->name());

  Field qc(qc_fid, true);
  Field qv(qv_fid, true);
  Field m(m_fid, true);

  // Random number generator seed
  int seed = get_random_test_seed();

  // Construct the Diagnostics
  auto &diag_factory = DiagnosticFactory::instance();
  register_diagnostics();

  ekat::ParameterList params_plus, params_times, params_scl_scl;
  REQUIRE_THROWS(diag_factory.create("BinaryOp", comm,
                                     params_plus, grid));  // No 'arg1', 'arg2', or 'binary_op'

  // Set time for qc and randomize its values
  qc.get_header().get_tracking().update_time_stamp(t0);
  qv.get_header().get_tracking().update_time_stamp(t0);
  m.get_header().get_tracking().update_time_stamp(t0);
  randomize_uniform(qc, seed++, 0, 200);
  randomize_uniform(qv, seed++, 0, 200);
  randomize_uniform(m,  seed++, 0, 200);

  // Create random mask fields for qc/qv
  qc.create_valid_mask();
  qv.create_valid_mask();
  randomize_uniform(qc.get_valid_mask(),seed++);
  randomize_uniform(qv.get_valid_mask(),seed++);

  // Create and set up the diagnostic
  params_plus.set("grid_name", grid->name());
  params_plus.set<std::string>("arg1", "qc");
  params_plus.set<std::string>("arg2", "qv");
  params_plus.set<std::string>("binary_op", "starcraft");
  REQUIRE_THROWS(diag_factory.create("BinaryOp", comm, params_plus, grid));  // Unknown binary_op
  params_plus.set<std::string>("binary_op", "plus");
  auto plus_diag = diag_factory.create("BinaryOp", comm, params_plus, grid);

  params_times.set("grid_name", grid->name());
  params_times.set<std::string>("arg1", "m");
  params_times.set<std::string>("arg2", "gravit");
  params_times.set<std::string>("binary_op", "times");
  auto prod_diag = diag_factory.create("BinaryOp", comm, params_times, grid);

  params_scl_scl.set("grid_name", grid->name());
  params_scl_scl.set<std::string>("arg1", "Avogad");
  params_scl_scl.set<std::string>("arg2", "boltz");
  params_scl_scl.set<std::string>("binary_op", "times");
  auto prod_scl_scl = diag_factory.create("BinaryOp", comm, params_scl_scl, grid);

  plus_diag->set_input_field(qc);
  plus_diag->set_input_field(qv);
  plus_diag->initialize();

  prod_diag->set_input_field(m);
  prod_diag->initialize();

  prod_scl_scl->initialize();

  // Run diag
  plus_diag->compute(t0);
  prod_diag->compute(t0);
  prod_scl_scl->compute(t0);

  auto plus_diag_f = plus_diag->get(); plus_diag_f.sync_to_host();
  auto prod_diag_f = prod_diag->get(); prod_diag_f.sync_to_host();
  auto rgas_diag_f = prod_scl_scl->get(); rgas_diag_f.sync_to_host();

  // Check that the output fields have the right values
  const auto g = physics::Constants<Real>::gravit.value;

  // field*constant diag
  m.scale(g);
  REQUIRE (not prod_diag_f.has_valid_mask());
  REQUIRE (views_are_equal(prod_diag_f,m));

  // field+field diag
  REQUIRE (plus_diag_f.has_valid_mask());
  auto tgt_mask = qc.get_valid_mask().clone(CloneFlags::CopyData);
  tgt_mask.scale(qv.get_valid_mask());
  REQUIRE (views_are_equal(tgt_mask,plus_diag_f.get_valid_mask()));

  qv.update(qc,1,1);
  auto tol = std::numeric_limits<Real>::epsilon()*1e2*0;
  REQUIRE (views_are_equal(plus_diag_f,qv,tol));

  // constant*constant diag
  REQUIRE (rgas_diag_f.get_view<Real,Host>()()==physics::Constants<Real>::dictionary().at("Rgas").value);
}

TEST_CASE("binary_ops_pow") {
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  ekat::Comm comm(MPI_COMM_WORLD);
  util::TimeStamp t0({2024, 1, 1}, {0, 0, 0});

  const int nlevs = 20;
  const int ncols = 30;
  auto grid = create_point_grid("physics",ncols,nlevs,comm);

  int seed = get_random_test_seed();

  auto& diag_factory = DiagnosticFactory::instance();
  register_diagnostics();

  auto make_diag = [&](const std::string& arg1, const std::string& arg2) {
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("arg1", arg1);
    params.set<std::string>("arg2", arg2);
    params.set<std::string>("binary_op", "pow");
    return diag_factory.create("BinaryOp", comm, params, grid);
  };

  FieldLayout layout = grid->get_3d_scalar_layout(LEV);
  Field T(FieldIdentifier("T", layout, K, grid->name()), true);
  T.get_header().get_tracking().update_time_stamp(t0);
  randomize_uniform(T, seed++, 200, 300);

  SECTION ("exponent_must_be_a_number") {
    REQUIRE_THROWS (make_diag("T","T"));   // field
    REQUIRE_THROWS (make_diag("T","P0"));  // named constant
    REQUIRE_THROWS (make_diag("2","T"));   // the base can't be a number with a field exponent
  }

  SECTION ("field_pow_number") {
    const std::vector<std::pair<std::string,Units>> cases = {
      {"2",   K*K},
      {"0.5", sqrt(K)},
      {"-1",  1/K},
      {"1.5", pow(K,ekat::RationalConstant(3,2))},
    };
    for (const auto& [p,units] : cases) {
      auto diag = make_diag("T",p);
      diag->set_input_field(T);
      diag->initialize();
      diag->compute(t0);
      auto d = diag->get();
      REQUIRE (d.get_header().get_identifier().get_units()==units);

      auto ref = T.clone("ref",CloneFlags::CopyData);
      apply_pow(ref,std::stod(p));
      REQUIRE (views_are_equal(d,ref));
    }

    // Units only take rational exponents
    auto bad = make_diag("T","0.70710678");
    bad->set_input_field(T);
    REQUIRE_THROWS (bad->initialize());
  }

  SECTION ("masked") {
    // Mask out the entries below 250K: they must come back as fill_value
    auto mask = compute_mask(T,250,Comparison::GE);
    T.set_valid_mask(mask);

    auto diag = make_diag("T","2");
    diag->set_input_field(T);
    diag->initialize();
    diag->compute(t0);
    auto d = diag->get();
    REQUIRE (d.has_valid_mask());
    REQUIRE (views_are_equal(d.get_valid_mask(),mask));

    auto ref = T.clone("ref",CloneFlags::CopyData);
    ref.scale(T);
    ref.deep_copy(constants::fill_value<Real>,mask,true);
    REQUIRE (views_are_equal(d,ref));
  }

  SECTION ("number_pow_number") {
    auto diag = make_diag("P0","2");
    diag->initialize();
    diag->compute(t0);
    auto d = diag->get();
    d.sync_to_host();
    const auto P0 = physics::Constants<Real>::P0.value;
    REQUIRE (d.get_view<const Real,Host>()()==std::pow(P0,Real(2)));
    REQUIRE (d.get_header().get_identifier().get_units()==Pa*Pa);
  }
}

}  // namespace scream
