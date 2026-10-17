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
 * Fields interpolated to the same pressure level share a valid-sample count,
 * also when aliased, except for a field whose input carries a mask of its own.
 * Midpoint and interface fields do not share, since their ranges differ.
 * An alias of a filled model field uses the count of that field.
 */

namespace scream {

namespace {

constexpr int ncols  = 4;
constexpr int nlevs  = 4;
constexpr int nsteps = 4;

constexpr Real FILL = constants::fill_value<Real>;

// 200/400/600/800 hPa, so the requested 500 hPa is strictly interior and is
// bracketed by levels 1 and 2 in every column. The interfaces reach 900 hPa,
// so 850 hPa is in range for an interface field, but not for a midpoint one.
Real p_mid_at (const int k) { return 20000.0 + 20000.0*k; }
Real p_int_at (const int k) { return 10000.0 + 20000.0*k; }

// X is the step number, constant in the vertical, so interpolation is exact.
// C is never positive, so X.where(C>0) is fill_value everywhere.
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

  // A field on interfaces (set to the step number in the time loop)
  Field Xi(FieldIdentifier("Xi",layout_int,K,grid->name()));
  Xi.allocate_view();
  Xi.get_header().get_tracking().update_time_stamp(t0);
  Xi.deep_copy(0);
  fm->add_field(Xi);

  // A model field that is filled in the first local column
  Field M(FieldIdentifier("m2d",grid->get_2d_scalar_layout(),K,grid->name()));
  M.allocate_view();
  M.get_header().set_may_be_filled(true);
  M.get_header().get_tracking().update_time_stamp(t0);
  M.deep_copy(7.0);
  M.sync_to_host();
  M.get_view<Real*,Host>()(0) = FILL;
  M.sync_to_dev();
  fm->add_field(M);

  calc_fields(X,C,0);
  return fm;
}

} // anonymous namespace

TEST_CASE ("plev_diags_share_an_avg_count_unless_input_is_masked")
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
  // The masked source is an intermediate (not written)
  f_pl.set<std::vector<std::string>>("aliases",{
      "Xmasked := X.where(C>0)",
  });
  f_pl.set<std::vector<std::string>>("field_names",{
      "X_at_500hPa",
      "C500 := C_at_500hPa",
      "Xmasked_at_500hPa",
      // NOTE: the alias sorts before its target, and is processed first
      "AM := m2d",
      "X_at_850hPa",
      "Xi_at_850hPa",
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
      auto Xi = fm->get_field("Xi");
      Xi.deep_copy(n);
      for (auto f : {X,C,Xi,fm->get_field("p_mid"),fm->get_field("p_int"),fm->get_field("m2d")}) {
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
  REQUIRE (scorpio::has_var(filename,"C500"));
  REQUIRE (scorpio::has_var(filename,"Xmasked_at_500hPa"));
  // The unmasked fields share the count keyed on the level, alias included.
  // CHECK, so a failure still reports the values below
  CHECK (scorpio::has_var(filename,"avg_count_500hPa_ncol"));
  CHECK_FALSE (scorpio::has_var(filename,"avg_count_X_at_500hPa_ncol"));
  CHECK_FALSE (scorpio::has_var(filename,"avg_count_C500_ncol"));
  CHECK_FALSE (scorpio::has_var(filename,"avg_count_ncol"));
  // ... while the field with a masked input has its own
  CHECK (scorpio::has_var(filename,"avg_count_Xmasked_at_500hPa_ncol"));
  // ... and so does the alias of the filled model field
  CHECK (scorpio::has_var(filename,"avg_count_AM_ncol"));
  // Midpoint and interface fields at the same level do not share
  CHECK (scorpio::has_var(filename,"avg_count_850hPa_ncol"));
  CHECK (scorpio::has_var(filename,"avg_count_850hPa_int_ncol"));
  scorpio::release_file(filename);

  {
    const auto layout2d = grid->get_2d_scalar_layout();
    Field xu(FieldIdentifier("X_at_500hPa",layout2d,K,gname));
    Field xm(FieldIdentifier("Xmasked_at_500hPa",layout2d,K,gname));
    Field c5(FieldIdentifier("C500",layout2d,K,gname));
    Field am(FieldIdentifier("AM",layout2d,K,gname));
    Field x8(FieldIdentifier("X_at_850hPa",layout2d,K,gname));
    Field xi8(FieldIdentifier("Xi_at_850hPa",layout2d,K,gname));
    for (auto* f : {&xu,&xm,&c5,&am,&x8,&xi8}) {
      f->allocate_view();
      f->get_header().get_tracking().update_time_stamp(t0);
    }

    FieldReader reader;
    reader.set_file_specs(filename);
    reader.set_dim_decomp(grid->get_partitioned_dim_gids(),comm);
    reader.set_fields({xu,xm,c5,am,x8,xi8});
    reader.read(0);
    xu.sync_to_host();
    xm.sync_to_host();
    c5.sync_to_host();
    am.sync_to_host();
    x8.sync_to_host();
    xi8.sync_to_host();
    auto xu_h = xu.get_view<const Real*,Host>();
    auto xm_h = xm.get_view<const Real*,Host>();
    auto c5_h = c5.get_view<const Real*,Host>();
    auto am_h = am.get_view<const Real*,Host>();
    auto x8_h = x8.get_view<const Real*,Host>();
    auto xi8_h = xi8.get_view<const Real*,Host>();

    // Mean of the step numbers 1..nsteps
    Real expected = 0;
    for (int n=1; n<=nsteps; ++n) expected += n;
    expected /= nsteps;

    const int nlocal = grid->get_num_local_dofs();
    for (int i=0; i<nlocal; ++i) {
      // The unmasked field is valid, so 500hPa is in range
      REQUIRE (xu_h(i)!=FILL);
      REQUIRE (xu_h(i)==Approx(expected));
      REQUIRE (c5_h(i)==Approx(-1.0));
      // The filled column has no valid samples (it was 0 with the layout count)
      REQUIRE (am_h(i)==(i==0 ? FILL : Real(7.0)));

      // 850 hPa is below the lowest midpoint, but above the lowest interface
      REQUIRE (x8_h(i)==FILL);
      REQUIRE (xi8_h(i)==Approx(expected));

      // No valid samples: must be fill_value (it was 0 with a shared count)
      REQUIRE (xm_h(i)==FILL);
    }
  }

  scorpio::finalize_subsystem();
}

} // namespace scream
