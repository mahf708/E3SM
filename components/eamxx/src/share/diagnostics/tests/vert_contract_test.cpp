#include <catch2/catch.hpp>

#include "share/diagnostics/register_diagnostics.hpp"
#include "share/physics/physics_constants.hpp"
#include "share/field/field_utils.hpp"
#include "share/grid/point_grid.hpp"
#include "share/core/eamxx_setup_random_test.hpp"
#include "share/util/eamxx_universal_constants.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scream {


TEST_CASE("vert_contract") {
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  // A world comm
  ekat::Comm comm(MPI_COMM_WORLD);

  // A time stamp
  util::TimeStamp t0({2024, 1, 1}, {0, 0, 0});

  // Create a grids manager
  const int nlevs = 30;
  const int dim5  = 5;
  const int ncols = 95;

  auto grid = create_point_grid("physics",ncols*comm.size(),nlevs,comm);

  auto &diag_factory = DiagnosticFactory::instance();
  register_diagnostics();

  int seed = get_random_test_seed();

  // Shared field layouts
  FieldLayout scalar2d_layout = grid->get_3d_scalar_layout(LEV);
  FieldLayout scalar3d_layout = grid->get_3d_vector_layout(LEV,dim5,"dim5");

  SECTION("bad_input") {
    ekat::ParameterList params;
    // No field_name -> throws in constructor
    REQUIRE_THROWS(diag_factory.create("VertContract", comm, params, grid));

    params.set<std::string>("field_name", "qc");
    params.set("grid_name", grid->name());
    // No contract_method -> throws in constructor
    REQUIRE_THROWS(diag_factory.create("VertContract", comm, params, grid));

    params.set<std::string>("contract_method", "xyz");
    // Invalid contract_method -> throws in constructor
    REQUIRE_THROWS(diag_factory.create("VertContract", comm, params, grid));

    params.set<std::string>("contract_method", "max");
    params.set<std::string>("weighting_method", "dp");
    // Weighted max/min -> throws in constructor
    REQUIRE_THROWS(diag_factory.create("VertContract", comm, params, grid));

    // ...while weighted var/std are fine
    params.set<std::string>("contract_method", "var");
    REQUIRE_NOTHROW(diag_factory.create("VertContract", comm, params, grid));
  }

  SECTION("dp_weighted_sum") {
    // Create and randomize a 3D (COL x CMP x LEV) input field
    Field fin(FieldIdentifier("qc", scalar3d_layout, kg/kg, grid->name()));
    fin.allocate_view();
    fin.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(fin, seed++, 1, 200);

    Field dp(FieldIdentifier("pseudo_density", scalar2d_layout, Pa, grid->name()));
    dp.allocate_view();
    dp.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(dp, seed++, 1, 200);

    // Create and set up the diagnostic
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("field_name", "qc");
    params.set<std::string>("contract_method", "sum");
    params.set<std::string>("weighting_method", "dp");
    auto diag = diag_factory.create("VertContract", comm, params, grid);
    diag->set_input_field(fin);
    diag->set_input_field(dp);
    diag->initialize();
    diag->compute(t0);
    auto diag_f = diag->get();

    // Manual reference: ref(col,cmp) = sum_lev( (dp/g)(col,lev) * f(col,cmp,lev) )
    auto dp_scaled = dp.clone("dp_scaled", CloneFlags::CopyData);
    dp_scaled.scale(1 / physics::Constants<Real>::gravit.value);
    auto ref = diag_f.clone("ref");
    vert_contraction(ref, fin, dp_scaled);

    REQUIRE(views_are_equal(diag_f, ref));
  }

  SECTION("dp_weighted_avg") {
    // Create and randomize a 2D (COL x LEV) input field
    Field fin(FieldIdentifier("qc", scalar2d_layout, kg/kg, grid->name()));
    fin.allocate_view();
    fin.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(fin, seed++, 1, 200);

    Field dp(FieldIdentifier("pseudo_density", scalar2d_layout, Pa, grid->name()));
    dp.allocate_view();
    dp.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(dp, seed++, 1, 200);

    // Create and set up the diagnostic
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("field_name", "qc");
    params.set<std::string>("contract_method", "avg");
    params.set<std::string>("weighting_method", "dp");
    auto diag = diag_factory.create("VertContract", comm, params, grid);
    diag->set_input_field(fin);
    diag->set_input_field(dp);
    diag->initialize();
    diag->compute(t0);
    auto diag_f = diag->get();

    // Manual reference:
    //   num(col) = sum_lev( (dp/g)(col,lev) * f(col,lev) )
    //   den(col) = sum_lev( (dp/g)(col,lev) )
    //   ref(col) = num(col) / den(col)
    auto dp_scaled = dp.clone("dp_scaled", CloneFlags::CopyData);
    dp_scaled.scale(1 / physics::Constants<Real>::gravit.value);
    auto dp_ones = dp_scaled.clone("ones");
    dp_ones.deep_copy(1);
    auto ref = diag_f.clone("ref");
    auto den = diag_f.clone("den");
    vert_contraction(ref, fin, dp_scaled);
    vert_contraction(den, dp_scaled, dp_ones);
    ref.scale_inv(den);

    REQUIRE(views_are_equal(diag_f, ref));
  }

  SECTION("unweighted_sum") {
    // Create and randomize a 2D (COL x LEV) input field
    Field fin(FieldIdentifier("qc", scalar2d_layout, kg/kg, grid->name()));
    fin.allocate_view();
    fin.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(fin, seed++, 1, 200);

    // Create and set up the diagnostic
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("field_name", "qc");
    params.set<std::string>("contract_method", "sum");
    params.set<std::string>("weighting_method", "none");
    auto diag = diag_factory.create("VertContract", comm, params, grid);
    diag->set_input_field(fin);
    diag->initialize();
    diag->compute(t0);
    auto diag_f = diag->get();

    // Manual reference: ref(col) = sum_lev( f(col,lev) )
    // Use a 1D all-ones weight (same as the diagnostic)
    FieldLayout ones_lev_layout{{LEV}, {nlevs}};
    Field ones_lev(FieldIdentifier("ones_lev", ones_lev_layout, none, grid->name()), true);
    ones_lev.deep_copy(1);
    auto ref = diag_f.clone("ref");
    vert_contraction(ref, fin, ones_lev);

    REQUIRE(views_are_equal(diag_f, ref));
  }

  SECTION("unweighted_avg") {
    // Create and randomize a 3D (COL x CMP x LEV) input field
    Field fin(FieldIdentifier("qc", scalar3d_layout, kg/kg, grid->name()));
    fin.allocate_view();
    fin.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(fin, seed++, 1, 200);

    // Create and set up the diagnostic
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("field_name", "qc");
    params.set<std::string>("contract_method", "avg");
    params.set<std::string>("weighting_method", "none");
    auto diag = diag_factory.create("VertContract", comm, params, grid);
    diag->set_input_field(fin);
    diag->initialize();
    diag->compute(t0);
    auto diag_f = diag->get();

    // Manual reference:
    //   num(col,cmp) = sum_lev( f(col,cmp,lev) )
    //   den(col,cmp) = sum_lev( 1 ) = nlevs
    //   ref(col,cmp) = num(col,cmp) / den(col,cmp)
    // Use a 1D all-ones weight (same as the diagnostic)
    FieldLayout ones_lev_layout{{LEV}, {nlevs}};
    Field ones_lev(FieldIdentifier("ones_lev", ones_lev_layout,
                                   none, grid->name()), true);
    ones_lev.deep_copy(1);
    auto fin_ones = fin.clone("fin_ones");
    fin_ones.deep_copy(1);
    auto ref = diag_f.clone("ref");
    auto den = diag_f.clone("den");
    vert_contraction(ref, fin, ones_lev);
    vert_contraction(den, fin_ones, ones_lev);
    ref.scale_inv(den);

    REQUIRE(views_are_equal(diag_f, ref));
  }

  SECTION("max_min") {
    // Create and randomize a 3D (COL x CMP x LEV) input field
    Field fin(FieldIdentifier("qc", scalar3d_layout, kg/kg, grid->name()));
    fin.allocate_view();
    fin.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(fin, seed++, -100, 100);

    for (std::string method : {"max", "min"}) {
      ekat::ParameterList params;
      params.set("grid_name", grid->name());
      params.set<std::string>("field_name", "qc");
      params.set<std::string>("contract_method", method);
      auto diag = diag_factory.create("VertContract", comm, params, grid);
      diag->set_input_field(fin);
      diag->initialize();
      diag->compute(t0);
      auto diag_f = diag->get();

      REQUIRE (diag_f.name()=="qc_vert_"+method);
      REQUIRE (diag_f.get_header().get_identifier().get_units()==kg/kg);
      REQUIRE (not diag_f.has_valid_mask());

      FieldLayout ones_lev_layout{{LEV}, {nlevs}};
      Field ones_lev(FieldIdentifier("ones_lev", ones_lev_layout, none, grid->name()), true);
      ones_lev.deep_copy(1);
      auto ref = diag_f.clone("ref");
      vert_contraction(ref, fin, ones_lev, method=="max" ? ContractionOp::Max : ContractionOp::Min);
      REQUIRE(views_are_equal(diag_f, ref));

      // Sanity check against a manual host computation
      fin.sync_to_host();
      diag_f.sync_to_host();
      auto v_in  = fin.get_view<const Real***,Host>();
      auto v_out = diag_f.get_view<const Real**,Host>();
      for (int i=0; i<v_out.extent_int(0); ++i) {
        for (int j=0; j<dim5; ++j) {
          Real tgt = v_in(i,j,0);
          for (int k=1; k<nlevs; ++k) {
            tgt = method=="max" ? std::max(tgt,v_in(i,j,k)) : std::min(tgt,v_in(i,j,k));
          }
          REQUIRE (v_out(i,j)==tgt);
        }
      }
    }
  }

  SECTION("max_min_masked") {
    // Create and randomize a 2D (COL x LEV) masked input field,
    // with column 0 fully masked
    Field fin(FieldIdentifier("qc", scalar2d_layout, kg/kg, grid->name()));
    fin.allocate_view();
    fin.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(fin, seed++, -100, 100);
    auto& mask = fin.create_valid_mask();
    randomize_uniform(mask, seed++);
    mask.sync_to_host();
    auto v_mask = mask.get_view<int**,Host>();
    for (int k=0; k<nlevs; ++k) {
      v_mask(0,k) = 0;
    }
    mask.sync_to_dev();

    for (std::string method : {"max", "min"}) {
      ekat::ParameterList params;
      params.set("grid_name", grid->name());
      params.set<std::string>("field_name", "qc");
      params.set<std::string>("contract_method", method);
      auto diag = diag_factory.create("VertContract", comm, params, grid);
      diag->set_input_field(fin);
      diag->initialize();
      diag->compute(t0);
      auto diag_f = diag->get();
      REQUIRE (diag_f.has_valid_mask());

      fin.sync_to_host();
      diag_f.sync_to_host();
      diag_f.get_valid_mask().sync_to_host();
      auto v_in    = fin.get_view<const Real**,Host>();
      auto v_out   = diag_f.get_view<const Real*,Host>();
      auto v_omask = diag_f.get_valid_mask().get_view<const int*,Host>();
      for (int i=0; i<v_out.extent_int(0); ++i) {
        bool any_valid = false;
        Real tgt = 0;
        for (int k=0; k<nlevs; ++k) {
          if (not v_mask(i,k)) continue;
          tgt = not any_valid ? v_in(i,k)
              : (method=="max" ? std::max(tgt,v_in(i,k)) : std::min(tgt,v_in(i,k)));
          any_valid = true;
        }
        REQUIRE (v_omask(i)==(any_valid ? 1 : 0));
        REQUIRE (v_out(i)==(any_valid ? tgt : constants::fill_value<Real>));
      }
      // Column 0 is fully masked
      REQUIRE (v_omask(0)==0);
    }
  }

  SECTION("var_std_dp_weighted") {
    // Create and randomize a 3D (COL x CMP x LEV) input field
    Field fin(FieldIdentifier("qc", scalar3d_layout, kg/kg, grid->name()));
    fin.allocate_view();
    fin.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(fin, seed++, 1, 200);

    Field dp(FieldIdentifier("pseudo_density", scalar2d_layout, Pa, grid->name()));
    dp.allocate_view();
    dp.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(dp, seed++, 1, 200);

    fin.sync_to_host();
    dp.sync_to_host();
    auto v_in = fin.get_view<const Real***,Host>();
    auto v_dp = dp.get_view<const Real**,Host>();

    for (std::string method : {"var", "std"}) {
      ekat::ParameterList params;
      params.set("grid_name", grid->name());
      params.set<std::string>("field_name", "qc");
      params.set<std::string>("contract_method", method);
      params.set<std::string>("weighting_method", "dp");
      auto diag = diag_factory.create("VertContract", comm, params, grid);
      diag->set_input_field(fin);
      diag->set_input_field(dp);
      diag->initialize();
      diag->compute(t0);
      auto diag_f = diag->get();

      REQUIRE (diag_f.name()=="qc_vert_"+method+"_dp_weighted");
      const auto tgt_units = method=="var" ? (kg/kg)*(kg/kg) : kg/kg;
      REQUIRE (diag_f.get_header().get_identifier().get_units()==tgt_units);

      // Manual two-pass reference (the 1/g in the weights cancels out)
      diag_f.sync_to_host();
      auto v_out = diag_f.get_view<const Real**,Host>();
      const auto tol = std::numeric_limits<Real>::epsilon()*1000;
      for (int i=0; i<v_out.extent_int(0); ++i) {
        for (int j=0; j<dim5; ++j) {
          Real wsum = 0, mean = 0, var = 0;
          for (int k=0; k<nlevs; ++k) {
            wsum += v_dp(i,k);
            mean += v_dp(i,k)*v_in(i,j,k);
          }
          mean /= wsum;
          for (int k=0; k<nlevs; ++k) {
            var += v_dp(i,k)*(v_in(i,j,k)-mean)*(v_in(i,j,k)-mean);
          }
          var /= wsum;
          const Real tgt = method=="var" ? var : std::sqrt(var);
          REQUIRE_THAT (v_out(i,j), Catch::Matchers::WithinRel(tgt,tol));
        }
      }
    }
  }

  SECTION("var_std_masked") {
    // Create and randomize a 2D (COL x LEV) masked input field,
    // with column 0 fully masked
    Field fin(FieldIdentifier("qc", scalar2d_layout, kg/kg, grid->name()));
    fin.allocate_view();
    fin.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(fin, seed++, 1, 200);
    auto& mask = fin.create_valid_mask();
    randomize_uniform(mask, seed++);
    mask.sync_to_host();
    auto v_mask = mask.get_view<int**,Host>();
    for (int k=0; k<nlevs; ++k) {
      v_mask(0,k) = 0;
    }
    mask.sync_to_dev();

    fin.sync_to_host();
    auto v_in = fin.get_view<const Real**,Host>();

    for (std::string method : {"var", "std"}) {
      ekat::ParameterList params;
      params.set("grid_name", grid->name());
      params.set<std::string>("field_name", "qc");
      params.set<std::string>("contract_method", method);
      auto diag = diag_factory.create("VertContract", comm, params, grid);
      diag->set_input_field(fin);
      diag->initialize();
      diag->compute(t0);
      auto diag_f = diag->get();
      REQUIRE (diag_f.has_valid_mask());

      diag_f.sync_to_host();
      diag_f.get_valid_mask().sync_to_host();
      auto v_out   = diag_f.get_view<const Real*,Host>();
      auto v_omask = diag_f.get_valid_mask().get_view<const int*,Host>();
      const auto tol = std::numeric_limits<Real>::epsilon()*1000;
      for (int i=0; i<v_out.extent_int(0); ++i) {
        int n = 0;
        Real mean = 0, var = 0;
        for (int k=0; k<nlevs; ++k) {
          if (v_mask(i,k)) {
            ++n;
            mean += v_in(i,k);
          }
        }
        REQUIRE (v_omask(i)==(n>0 ? 1 : 0));
        if (n==0) {
          REQUIRE (v_out(i)==constants::fill_value<Real>);
          continue;
        }
        mean /= n;
        for (int k=0; k<nlevs; ++k) {
          if (v_mask(i,k)) {
            var += (v_in(i,k)-mean)*(v_in(i,k)-mean);
          }
        }
        var /= n;
        const Real tgt = method=="var" ? var : std::sqrt(var);
        // A column with a single valid level has zero variance
        REQUIRE_THAT (v_out(i), Catch::Matchers::WithinAbs(tgt,tol*std::max(Real(1),tgt)));
      }
      REQUIRE (v_omask(0)==0);
    }
  }
}

}  // namespace scream
