#include <catch2/catch.hpp>

#include "share/diagnostics/register_diagnostics.hpp"
#include "share/physics/physics_constants.hpp"
#include "share/field/field_utils.hpp"
#include "share/grid/point_grid.hpp"
#include "share/core/eamxx_setup_random_test.hpp"
#include "share/util/eamxx_universal_constants.hpp"

namespace scream {


TEST_CASE("horiz_avg") {
  using namespace ShortFieldTagsNames;
  using PC = physics::Constants<Real>;

  // A numerical tolerance
  auto tol = std::numeric_limits<Real>::epsilon() * 100;

  // Random number generator seed
  int seed = get_random_test_seed();

  // A world comm
  ekat::Comm comm(MPI_COMM_WORLD);

  // A time stamp
  util::TimeStamp t0({2024, 1, 1}, {0, 0, 0});

  // Create a grids manager - single column for these tests
  const int nlevs = 3;
  const int ncols = 6;

  auto grid = create_point_grid("physics",ncols,nlevs,comm);
  auto area = grid->create_geometry_data("area",grid->get_2d_scalar_layout());
  area.deep_copy(4.0*PC::Pi/grid->get_num_global_dofs());

  auto &diag_factory = DiagnosticFactory::instance();
  register_diagnostics();

  // No 'field_name' parameter
  REQUIRE_THROWS(diag_factory.create("HorizAvg", comm, ekat::ParameterList(), grid));

  // Get a hold to area, for easy cloning later
  auto area_sum = field_sum(area,&comm).as<Real>();

  SECTION ("scalar_x") {
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("field_name", "s_x");

    FieldLayout layout = grid->get_2d_scalar_layout();
    FieldIdentifier scl_x_fid ("s_x" , layout, ekat::units::none, grid->name());
    Field scl_x (scl_x_fid , true);
    scl_x.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(scl_x,seed++);

    auto diag = diag_factory.create("HorizAvg", comm, params, grid);
    diag->set_input_field(scl_x);
    diag->initialize();
    diag->compute(t0);
    auto diag_f = diag->get();

    auto manual = diag_f.clone("manual");
    auto area_scaled = area.clone(CloneFlags::CopyData);
    area_scaled.scale(1/area_sum);

    horiz_contraction(manual,scl_x,area_scaled,comm);

    // Compare
    auto diff = diag_f.clone(CloneFlags::CopyData);
    diff.update(manual,-1,1);
    auto diff_norm = inf_norm(diff,&comm).as<Real>();
    REQUIRE_THAT (diff_norm, Catch::WithinAbs(0,tol));
  }

  SECTION ("vector_xz_masked") {
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("field_name", "vec_xz");

    FieldLayout layout = grid->get_3d_vector_layout(LEV,2,"dim2");
    FieldIdentifier vec_xz_fid ("vec_xz" , layout, ekat::units::none, grid->name());
    Field vec_xz (vec_xz_fid , true);
    randomize_uniform(vec_xz,seed++);
    vec_xz.get_header().get_tracking().update_time_stamp(t0);
    auto& mask = vec_xz.create_valid_mask();
    randomize_uniform(mask,seed++);

    auto diag = diag_factory.create("HorizAvg", comm, params, grid);
    diag->set_input_field(vec_xz);
    diag->initialize();
    diag->compute(t0);
    auto diag_f = diag->get();
    REQUIRE (diag_f.has_valid_mask());

    auto& d_mask = diag_f.get_valid_mask();

    // Manual calculation: numerator = sum_col(area*f*mask), denominator = sum_col(area*mask)
    auto manual = diag_f.clone("manual");
    horiz_contraction(manual,vec_xz,area,comm);

    auto ones_xz = vec_xz.clone("ones");
    ones_xz.deep_copy(1);
    ones_xz.set_valid_mask(mask);
    auto manual_den = diag_f.clone("manual_den");
    horiz_contraction(manual_den,ones_xz,area,comm);

    // Expected valid_mask: 1 where denominator > 0
    auto tgt_mask = d_mask.clone("tgt_mask");
    compute_mask(manual_den,0,Comparison::NE,tgt_mask);
    REQUIRE (views_are_equal(tgt_mask,d_mask));

    // Compute manual average: numerator / denominator (where valid)
    manual.scale_inv(manual_den,tgt_mask);

    // Compare where valid; zero out fill_value entries before norm
    auto diff = diag_f.clone("diff", CloneFlags::CopyData);
    diff.update(manual,-1,1,d_mask);
    diff.deep_copy(0,d_mask,true);
    auto diff_norm = inf_norm(diff,&comm).as<Real>();
    REQUIRE_THAT (diff_norm, Catch::WithinAbs(0,tol));
  }

  SECTION ("vector_xz_fully_masked") {
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("field_name", "vec_xz");

    FieldLayout layout = grid->get_3d_vector_layout(LEV,2,"dim2");
    FieldIdentifier vec_xz_fid ("vec_xz" , layout, ekat::units::none, grid->name());
    Field vec_xz (vec_xz_fid , true);
    randomize_uniform(vec_xz,seed++);
    vec_xz.get_header().get_tracking().update_time_stamp(t0);
    auto& mask = vec_xz.create_valid_mask();
    mask.deep_copy(0);

    auto diag = diag_factory.create("HorizAvg", comm, params, grid);
    diag->set_input_field(vec_xz);
    diag->initialize();
    diag->compute(t0);
    auto diag_f = diag->get();
    REQUIRE (diag_f.has_valid_mask());

    auto& d_mask = diag_f.get_valid_mask();

    auto tgt_mask = d_mask.clone("tgt_mask");
    mask.deep_copy(0);
    diag->compute(t0);
    tgt_mask.deep_copy(0);
    REQUIRE(views_are_equal(d_mask,tgt_mask));
  }

  SECTION ("constant_x") {
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("field_name", "s_xz");

    FieldLayout layout = grid->get_3d_scalar_layout(LEV);
    FieldIdentifier scl_xz_fid ("s_xz" , layout, ekat::units::none, grid->name());
    Field scl_xz (scl_xz_fid , true);
    scl_xz.get_header().get_tracking().update_time_stamp(t0);

    Real c = 1.2345;
    scl_xz.deep_copy(c);

    auto diag = diag_factory.create("HorizAvg", comm, params, grid);
    diag->set_input_field(scl_xz);
    diag->initialize();
    diag->compute(t0);
    auto diag_f = diag->get();

    // Constant fields should have an avg equal to the constant(within rounding errors)
    auto manual = diag_f.clone("manual");
    manual.deep_copy(c);

    // Compare
    auto diff = diag_f.clone(CloneFlags::CopyData);
    diff.update(manual,-1,1);
    auto diff_norm = inf_norm(diff,&comm).as<Real>();
    REQUIRE_THAT (diff_norm, Catch::WithinAbs(0,tol));
  }

  SECTION ("bad_contract_method") {
    ekat::ParameterList params;
    params.set("grid_name", grid->name());
    params.set<std::string>("field_name", "s_x");
    params.set<std::string>("contract_method", "xyz");
    REQUIRE_THROWS(diag_factory.create("HorizAvg", comm, params, grid));
  }

  SECTION ("sum_max_min_scalar_x") {
    FieldLayout layout = grid->get_2d_scalar_layout();
    FieldIdentifier scl_x_fid ("s_x" , layout, ekat::units::K, grid->name());
    Field scl_x (scl_x_fid , true);
    scl_x.get_header().get_tracking().update_time_stamp(t0);
    randomize_uniform(scl_x,seed++,-1,1);

    for (std::string method : {"sum", "max", "min"}) {
      ekat::ParameterList params;
      params.set("grid_name", grid->name());
      params.set<std::string>("field_name", "s_x");
      params.set<std::string>("contract_method", method);

      auto diag = diag_factory.create("HorizAvg", comm, params, grid);
      diag->set_input_field(scl_x);
      diag->initialize();
      diag->compute(t0);
      auto diag_f = diag->get();

      REQUIRE (diag_f.name()=="s_x_horiz_"+method);
      // All three are unweighted, so units are unchanged
      REQUIRE (diag_f.get_header().get_identifier().get_units()==ekat::units::K);
      REQUIRE (not diag_f.has_valid_mask());

      diag_f.sync_to_host();
      const auto d = diag_f.get_view<const Real,Host>()();
      if (method=="sum") {
        REQUIRE_THAT (d, Catch::WithinAbs(field_sum(scl_x,&comm).as<Real>(),tol));
      } else if (method=="max") {
        REQUIRE (d==field_max(scl_x,&comm).as<Real>());
      } else {
        REQUIRE (d==field_min(scl_x,&comm).as<Real>());
      }
    }
  }

  SECTION ("max_min_vector_xz_masked") {
    FieldLayout layout = grid->get_3d_vector_layout(LEV,2,"dim2");
    FieldIdentifier vec_xz_fid ("vec_xz" , layout, ekat::units::none, grid->name());
    Field vec_xz (vec_xz_fid , true);
    randomize_uniform(vec_xz,seed++,-1,1);
    vec_xz.get_header().get_tracking().update_time_stamp(t0);
    auto& mask = vec_xz.create_valid_mask();
    randomize_uniform(mask,seed++);
    // Fully mask out the first component at the first level, on all ranks
    mask.sync_to_host();
    auto v_mask = mask.get_view<int***,Host>();
    for (int i=0; i<v_mask.extent_int(0); ++i) {
      v_mask(i,0,0) = 0;
    }
    mask.sync_to_dev();

    for (std::string method : {"max", "min"}) {
      ekat::ParameterList params;
      params.set("grid_name", grid->name());
      params.set<std::string>("field_name", "vec_xz");
      params.set<std::string>("contract_method", method);

      auto diag = diag_factory.create("HorizAvg", comm, params, grid);
      diag->set_input_field(vec_xz);
      diag->initialize();
      diag->compute(t0);
      auto diag_f = diag->get();
      REQUIRE (diag_f.has_valid_mask());
      auto& d_mask = diag_f.get_valid_mask();

      // Expected valid_mask: 1 where at least one column is valid
      auto ones = vec_xz.clone("ones");
      ones.deep_copy(1);
      ones.set_valid_mask(mask);
      auto count = diag_f.clone("count");
      auto w = grid->get_geometry_data("area").clone("w");
      w.deep_copy(1);
      horiz_contraction(count,ones,w,comm);
      auto tgt_mask = d_mask.clone("tgt_mask");
      compute_mask(count,0,Comparison::NE,tgt_mask);
      REQUIRE (views_are_equal(tgt_mask,d_mask));

      // Expected values where valid, fill_value elsewhere
      auto manual = diag_f.clone("manual");
      horiz_contraction(manual,vec_xz,w,comm,method=="max" ? ContractionOp::Max : ContractionOp::Min);
      manual.deep_copy(constants::fill_value<Real>,tgt_mask,true);
      REQUIRE (views_are_equal(diag_f,manual));

      diag_f.sync_to_host();
      d_mask.sync_to_host();
      REQUIRE (d_mask.get_view<const int**,Host>()(0,0)==0);
      REQUIRE (diag_f.get_view<const Real**,Host>()(0,0)==constants::fill_value<Real>);
    }
  }
}

}  // namespace scream
