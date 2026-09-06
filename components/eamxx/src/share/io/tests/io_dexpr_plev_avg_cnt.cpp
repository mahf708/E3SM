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
 * Two fields interpolated to ONE pressure level must not share one valid-sample
 * count.
 *
 * scorpio_output.cpp used to exempt FieldAtPressureLevel, FieldAtHeight and
 * AerosolOpticalDepth550nm from per-field enrollment and key their count on the
 * level instead, on the stated grounds that fields at one level are invalid in
 * exactly the same places. Only the geometric half of that is true.
 * field_at_pressure_level.cpp marks a column valid where the level is in range
 * AND the two bracketing SOURCE values are valid, so two fields at one level
 * differ wherever their inputs do.
 *
 * The consequence is not a biased average. Here the condition is nowhere true,
 * so the masked field accumulates nothing; with a shared denominator it is
 * divided by the other field's count and published as 0 K in every column with
 * no fill flag, where the correct output is missing everywhere. A check that
 * only asked whether output appeared would pass.
 *
 * TWO THINGS ABOUT HOW THIS IS SPELLED, both of which decide whether the test
 * can fail at all:
 *
 *  - the outputs are requested by their CANONICAL names, with no ':=' on the
 *    output side. An expression bound with ':=' is written under an alias, and
 *    aliases were already enrolled per field, so the ':=' spelling of this same
 *    request never reaches the exempted branch. The first test written for this
 *    used that spelling and passed while the defect was live.
 *  - the unmasked field is asserted to hold its hand-computed value, not merely
 *    to exist. Without that, a run in which 500 hPa was out of range everywhere
 *    would give fill for both fields and pass for the wrong reason.
 */

namespace scream {

namespace {

constexpr int ncols  = 4;
constexpr int nlevs  = 4;
constexpr int nsteps = 4;

constexpr Real FILL = constants::fill_value<Real>;

// 200/400/600/800 hPa, so the requested 500 hPa is strictly interior and is
// bracketed by levels 1 and 2 in every column.
Real p_mid_at (const int k) { return 20000.0 + 20000.0*k; }
Real p_int_at (const int k) { return 10000.0 + 20000.0*k; }

// X is the step number and is constant in the vertical, so interpolating it to
// any interior pressure returns the step number exactly, and the average over
// the window is the mean of 1..nsteps with no interpolation error to allow for.
//
// C is never positive, so X.where(C>0) is fill at every level, in every column,
// at every step: the masked interpolation has NOTHING to average.
void calc_fields (Field& X, Field& C, const int n)
{
  auto X_h = X.get_view<Real**,Host>();
  auto C_h = C.get_view<Real**,Host>();
  const int nl = X.get_header().get_identifier().get_layout().dim(0);
  for (int i=0; i<nl; ++i) {
    for (int k=0; k<nlevs; ++k) {
      X_h(i,k) = n;
      C_h(i,k) = -1.0;
    }
  }
  X.sync_to_dev();
  C.sync_to_dev();
}

std::shared_ptr<FieldManager>
create_test_fm (const std::shared_ptr<const GridsManager>& gm,
                const std::shared_ptr<const AbstractGrid>& grid,
                const util::TimeStamp& t0)
{
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  const auto layout_mid = grid->get_3d_scalar_layout(LEV);
  const auto layout_int = grid->get_3d_scalar_layout(ILEV);
  auto fm = std::make_shared<FieldManager>(gm);

  Field X(FieldIdentifier("X",layout_mid,K,grid->name()));
  Field C(FieldIdentifier("C",layout_mid,K,grid->name()));
  // FieldAtPressureLevel lists both p_mid and p_int as inputs, since it does
  // not know which it needs until it sees the field's layout. Provide both.
  Field pm(FieldIdentifier("p_mid",layout_mid,Pa,grid->name()));
  Field pi(FieldIdentifier("p_int",layout_int,Pa,grid->name()));
  for (auto* f : {&X,&C,&pm,&pi}) {
    f->allocate_view();
    f->get_header().get_tracking().update_time_stamp(t0);
    fm->add_field(*f);
  }

  auto pm_h = pm.get_view<Real**,Host>();
  auto pi_h = pi.get_view<Real**,Host>();
  const int nl = grid->get_num_local_dofs();
  for (int i=0; i<nl; ++i) {
    for (int k=0; k<nlevs; ++k)   pm_h(i,k) = p_mid_at(k);
    for (int k=0; k<nlevs+1; ++k) pi_h(i,k) = p_int_at(k);
  }
  pm.sync_to_dev();
  pi.sync_to_dev();

  calc_fields(X,C,0);
  return fm;
}

} // anonymous namespace

TEST_CASE ("plev_diags_do_not_share_an_avg_count")
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

  const std::string prefix = "io_dexpr_plev_avg_cnt";
  const int dt = 1;

  ekat::ParameterList params;
  params.set<std::string>("filename_prefix",prefix);
  params.set<std::string>("averaging_type","AVERAGE");
  params.set<std::string>("floating_point_precision","real");
  auto& f_pl = params.sublist("fields").sublist(gname);
  // The masked source is an INTERMEDIATE: it is not itself written, so it
  // carries no output alias of its own and cannot mask the defect.
  f_pl.set<std::vector<std::string>>("aliases",{
      "Xmasked := X.where(C>0)",
  });
  // Canonical names, no ':='. This is the ordinary configuration spelling and
  // the only one that reaches the branch under test.
  f_pl.set<std::vector<std::string>>("field_names",{
      "X_at_500hPa",
      "Xmasked_at_500hPa",
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
      auto X = fm->get_field("X");
      auto C = fm->get_field("C");
      calc_fields(X,C,n);
      for (auto f : {X,C,fm->get_field("p_mid"),fm->get_field("p_int")}) {
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

  scorpio::register_file(filename,scorpio::Read);
  REQUIRE (scorpio::has_var(filename,"X_at_500hPa"));
  REQUIRE (scorpio::has_var(filename,"Xmasked_at_500hPa"));
  // Each interpolated output gets a count of its own ...
  //
  // CHECK rather than REQUIRE, deliberately. If the class-based exemption
  // comes back, the counts and the values are both wrong, and a fatal
  // assertion here would stop the test before it could report the value --
  // leaving whoever reads the failure to guess how bad it is. With CHECK the
  // run continues and prints the fabricated 0 K alongside the missing count.
  CHECK (scorpio::has_var(filename,"avg_count_X_at_500hPa_ncol"));
  CHECK (scorpio::has_var(filename,"avg_count_Xmasked_at_500hPa_ncol"));
  // ... and the count keyed on the LEVEL, which both used to share, is gone.
  CHECK_FALSE (scorpio::has_var(filename,"avg_count_500hPa_ncol"));
  scorpio::release_file(filename);

  {
    const auto layout2d = grid->get_2d_scalar_layout();
    Field xu(FieldIdentifier("X_at_500hPa",layout2d,K,gname));
    Field xm(FieldIdentifier("Xmasked_at_500hPa",layout2d,K,gname));
    for (auto* f : {&xu,&xm}) {
      f->allocate_view();
      f->get_header().get_tracking().update_time_stamp(t0);
    }

    FieldReader reader;
    reader.set_file_specs(filename);
    reader.set_dim_decomp(grid->get_partitioned_dim_gids(),comm);
    reader.set_fields({xu,xm});
    reader.read(0);
    xu.sync_to_host();
    xm.sync_to_host();
    auto xu_h = xu.get_view<const Real*,Host>();
    auto xm_h = xm.get_view<const Real*,Host>();

    // Mean of the step numbers 1..nsteps. X is constant in the vertical, so
    // the interpolation to 500 hPa is exact and this is the whole answer.
    Real expected = 0;
    for (int n=1; n<=nsteps; ++n) expected += n;
    expected /= nsteps;

    const int nlocal = grid->get_num_local_dofs();
    for (int i=0; i<nlocal; ++i) {
      // HAS TEETH. 500 hPa is reachable and the unmasked interpolation is
      // produced, so the fill asserted below is the mask's doing and not an
      // out-of-range level that would make both fields fill.
      REQUIRE (xu_h(i)!=FILL);
      REQUIRE (xu_h(i)==Approx(expected));

      // The condition never held, so this window has no valid samples and the
      // correct output is missing. With a shared denominator it came back as
      // 0/nsteps == 0 K, unflagged -- a fabricated field, not a biased one.
      REQUIRE (xm_h(i)==FILL);
    }
  }

  scorpio::finalize_subsystem();
}

} // namespace scream
