#include <catch2/catch.hpp>

#include "share/data_managers/field_manager.hpp"
#include "share/data_managers/mesh_free_grids_manager.hpp"
#include "share/diagnostics/register_diagnostics.hpp"
#include "share/field/field.hpp"
#include "share/field/field_reader.hpp"
#include "share/io/eamxx_output_manager.hpp"
#include "share/scorpio_interface/eamxx_scorpio_interface.hpp"
#include "share/util/eamxx_time_stamp.hpp"
#include "share/util/eamxx_universal_constants.hpp"

#include <ekat_comm.hpp>
#include <ekat_parameter_list.hpp>
#include <ekat_units.hpp>

#include <fstream>
#include <string>
#include <vector>

/*
 * Averaged conditional sampling, end to end, against hand-computed values.
 *
 * The three things checked here used to be wrong together, and they are easy
 * to get wrong again independently:
 *
 *  - a masked field must be flagged may_be_filled, or the temporal average
 *    sums fill values as though they were data;
 *  - it must be enrolled in valid-sample counting, or it divides by the number
 *    of STEPS rather than the number of VALID steps;
 *  - it must get a count of its OWN. An earlier version keyed the count by
 *    layout, so two masked fields of the same shape shared one denominator,
 *    which is right only when their masks agree.
 *
 * Chained sampling is checked in the same stream, because X.where(A).where(B)
 * is the documented way to spell a conjunction and it must be the
 * intersection, not "whatever B says".
 */

namespace scream {

namespace {

constexpr int ncols = 4;
constexpr int nlevs = 3;
constexpr int nsteps = 4;

constexpr Real FILL = constants::fill_value<Real>;

// X is just the step number, so the mean over any subset of steps is the mean
// of those step numbers and can be written down by hand.
//
// The two conditions are deliberately DIFFERENT and both level-dependent, so
// that a1 and a5 end up with different masks and a shared denominator would
// give a visibly wrong answer:
//   C1>0 on step n iff (n+k) is even   -> half the steps, alternating by level
//   C2>0 on step n iff n >= 2          -> the last three steps, all levels
// except level 0, which is never valid for C2, to exercise the all-invalid
// case with a zero count.
void calc_fields (Field& X, Field& C1, Field& C2, const int n)
{
  auto X_h  = X .get_view<Real**,Host>();
  auto C1_h = C1.get_view<Real**,Host>();
  auto C2_h = C2.get_view<Real**,Host>();
  const int nl = X.get_header().get_identifier().get_layout().dim(0);
  for (int i=0; i<nl; ++i) {
    for (int k=0; k<nlevs; ++k) {
      X_h (i,k) = n;
      C1_h(i,k) = ((n+k)%2==0) ? 1.0 : -1.0;
      C2_h(i,k) = (k>0 and n>=2) ? 1.0 : -1.0;
    }
  }
  X.sync_to_dev();
  C1.sync_to_dev();
  C2.sync_to_dev();
}

// The same predicates, on the host, for the expected values.
bool c1_holds (int n, int k) { return (n+k)%2==0; }
bool c2_holds (int n, int k) { return k>0 and n>=2; }

std::shared_ptr<FieldManager>
create_test_fm (const std::shared_ptr<const GridsManager>& gm,
                const std::shared_ptr<const AbstractGrid>& grid,
                const util::TimeStamp& t0)
{
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  const auto layout3d = grid->get_3d_scalar_layout(LEV);
  auto fm = std::make_shared<FieldManager>(gm);

  Field X(FieldIdentifier("X" ,layout3d,K,grid->name()));
  Field C1(FieldIdentifier("C1",layout3d,K,grid->name()));
  Field C2(FieldIdentifier("C2",layout3d,K,grid->name()));
  for (auto* f : {&X,&C1,&C2}) {
    f->allocate_view();
    f->get_header().get_tracking().update_time_stamp(t0);
    fm->add_field(*f);
  }
  calc_fields(X,C1,C2,0);
  return fm;
}

} // anonymous namespace

TEST_CASE ("averaged_conditional_sampling")
{
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  register_diagnostics();

  ekat::Comm comm(MPI_COMM_WORLD);
  scorpio::init_subsystem(comm);

  auto gm = create_mesh_free_grids_manager(comm,0,0,nlevs,ncols);
  gm->build_grids();
  auto grid = gm->get_grid("point_grid");
  const auto gname = grid->name();

  util::TimeStamp t0({2023,1,1},{0,0,0});
  auto fm = create_test_fm(gm,grid,t0);

  const std::string prefix = "io_dexpr_masks";
  const int dt = 1;

  ekat::ParameterList params;
  params.set<std::string>("filename_prefix",prefix);
  params.set<std::string>("averaging_type","AVERAGE");
  params.set<std::string>("floating_point_precision","real");
  auto& f_pl = params.sublist("fields").sublist(gname);
  f_pl.set<std::vector<std::string>>("field_names",{
      "a1 := X.where(C1>0)",
      "a5 := X.where(C1>0).where(C2>0)",
  });
  auto& ctrl_pl = params.sublist("output_control");
  ctrl_pl.set<std::string>("frequency_units","nsteps");
  ctrl_pl.set<int>("frequency",nsteps);
  ctrl_pl.set<bool>("save_grid_data",false);

  auto t = t0;
  {
    OutputManager om;
    om.initialize(comm,params,t0,false);
    om.setup(fm,gm->get_grid_names());

    for (int n=1; n<=nsteps; ++n) {
      om.init_timestep(t,dt);
      t += dt;
      auto X  = fm->get_field("X");
      auto C1 = fm->get_field("C1");
      auto C2 = fm->get_field("C2");
      calc_fields(X,C1,C2,n);
      for (auto f : {X,C1,C2}) {
        f.get_header().get_tracking().update_time_stamp(t);
      }
      om.run(t);
    }
    om.finalize();
  }

  const auto filename = prefix + ".AVERAGE.nsteps_x" + std::to_string(nsteps) +
                        ".np" + std::to_string(comm.size()) + "." +
                        t0.to_string() + ".nc";
  std::ifstream file_check(filename);
  REQUIRE (file_check.good());
  file_check.close();

  // Each masked field must get a count of its OWN. A single layout-keyed
  // avg_count shared between a1 and a5 is the bug this guards: their masks
  // differ, so one denominator cannot serve both.
  scorpio::register_file(filename,scorpio::Read);
  REQUIRE (scorpio::has_var(filename,"a1"));
  REQUIRE (scorpio::has_var(filename,"a5"));
  REQUIRE (scorpio::has_var(filename,"avg_count_a1_ncol_lev"));
  REQUIRE (scorpio::has_var(filename,"avg_count_a5_ncol_lev"));
  // The counts are keyed on the name the stream WRITES, which for 'a1 := ..'
  // is the alias and not the diagnostic's internal name. Getting that wrong
  // does not fail loudly: the lookup misses, an empty suffix is substituted,
  // and both fields quietly share one avg_count_<layout>. So assert the
  // shared one is absent, not just that the per-field ones are present.
  REQUIRE_FALSE (scorpio::has_var(filename,"avg_count_ncol_lev"));
  scorpio::release_file(filename);

  {
    const auto layout3d = grid->get_3d_scalar_layout(LEV);
    Field a1(FieldIdentifier("a1",layout3d,K,gname));
    Field a5(FieldIdentifier("a5",layout3d,K,gname));
    for (auto* f : {&a1,&a5}) {
      f->allocate_view();
      f->get_header().get_tracking().update_time_stamp(t0);
    }

    FieldReader reader;
    reader.set_file_specs(filename);
    reader.set_dim_decomp(grid->get_partitioned_dim_gids(),comm);
    reader.set_fields({a1,a5});
    reader.read(0);
    a1.sync_to_host();
    a5.sync_to_host();
    auto a1_h = a1.get_view<const Real**,Host>();
    auto a5_h = a5.get_view<const Real**,Host>();

    const int nlocal = grid->get_num_local_dofs();
    for (int i=0; i<nlocal; ++i) {
      for (int k=0; k<nlevs; ++k) {
        // a1: mean of the step numbers where C1 held. Never divided by nsteps.
        Real sum1 = 0; int cnt1 = 0;
        // a5: the INTERSECTION. An element that C1 rejected holds a fill
        // value, and C2 alone must not resurrect it.
        Real sum5 = 0; int cnt5 = 0;
        for (int n=1; n<=nsteps; ++n) {
          if (c1_holds(n,k))                 { sum1 += n; ++cnt1; }
          if (c1_holds(n,k) and c2_holds(n,k)){ sum5 += n; ++cnt5; }
        }

        if (cnt1==0) {
          REQUIRE (a1_h(i,k)==FILL);
        } else {
          REQUIRE (a1_h(i,k)==Approx(sum1/cnt1));
        }
        // Level 0 is never valid for C2, so a5 there is an all-invalid window:
        // it must be fill, not zero and not the unconditional mean.
        if (cnt5==0) {
          REQUIRE (a5_h(i,k)==FILL);
        } else {
          REQUIRE (a5_h(i,k)==Approx(sum5/cnt5));
        }
        // The point of keeping the counts separate. a1 and a5 have the same
        // layout and different masks, so wherever their valid-sample counts
        // differ, one shared denominator cannot serve both: a5 would come
        // back as sum5/cnt1. Assert we are not looking at that number.
        if (cnt5>0 and cnt1>0 and cnt1!=cnt5) {
          REQUIRE (a5_h(i,k)!=Approx(sum5/cnt1));
        }
      }
    }
  }

  scorpio::finalize_subsystem();
}

} // namespace scream
