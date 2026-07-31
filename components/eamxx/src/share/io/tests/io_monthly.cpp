#include <catch2/catch.hpp>

#include "share/io/eamxx_output_manager.hpp"

#include "share/data_managers/mesh_free_grids_manager.hpp"

#include "share/field/field_reader.hpp"
#include "share/field/field_utils.hpp"
#include "share/field/field.hpp"
#include "share/data_managers/field_manager.hpp"

#include "share/util/eamxx_universal_constants.hpp"
#include "share/core/eamxx_setup_random_test.hpp"
#include "share/util/eamxx_time_stamp.hpp"
#include "share/core/eamxx_types.hpp"

#include <ekat_units.hpp>
#include <ekat_parameter_list.hpp>
#include <ekat_assert.hpp>
#include <ekat_comm.hpp>

#include <iomanip>
#include <memory>

namespace scream {

void add (const Field& f, const double v) {
  auto data = f.get_internal_view_data<Real,Host>();
  auto nscalars = f.get_header().get_alloc_properties().get_num_scalars();
  for (int i=0; i<nscalars; ++i) {
    data[i] += v;
  }
  f.sync_to_dev();
}

util::TimeStamp get_t0 () {
  return util::TimeStamp({2000,1,15},{0,0,0});
}

// For the monthly-average test, start at the beginning of a month, which
// is the typical setup of a run producing monthly averages
util::TimeStamp get_t0_avg () {
  return util::TimeStamp({2000,1,1},{0,0,0});
}

// Number of monthly averages written by the monthly-average test
constexpr int num_months = 3;

std::shared_ptr<const GridsManager>
get_gm (const ekat::Comm& comm)
{
  // For 2+ ranks tests, this will check IO works correctly
  // even if one rank owns 0 dofs
  const int ngcols = std::max(comm.size()-1,1);
  const int nlevs = 4;
  auto gm = create_mesh_free_grids_manager(comm,0,0,nlevs,ngcols);
  gm->build_grids();
  return gm;
}

std::shared_ptr<FieldManager>
get_fm (const std::shared_ptr<const AbstractGrid>& grid,
        const util::TimeStamp& t0, int seed)
{
  using FL  = FieldLayout;
  using FID = FieldIdentifier;
  using namespace ShortFieldTagsNames;

  // Note: we use a discrete set of random values, so we can
  // check answers without risk of non-bfb diffs due to ops order
  std::vector<Real> values;
  for (int i=0; i<=100; ++i)
    values.push_back(static_cast<Real>(i));

  const int nlcols = grid->get_num_local_dofs();
  const int nlevs  = grid->get_num_vertical_levels();

  std::vector<FL> layouts =
  {
    FL({COL         }, {nlcols        }),
    FL({COL,     LEV}, {nlcols,  nlevs}),
    FL({COL,CMP,ILEV}, {nlcols,2,nlevs+1})
  };

  auto fm = std::make_shared<FieldManager>(grid);

  int count=0;
  for (const auto& fl : layouts) {
    FID fid("f_"+std::to_string(count),fl,ekat::units::none,grid->name());
    Field f(fid);
    f.allocate_view();
    randomize_discrete (f,seed++,values);
    f.get_header().get_tracking().update_time_stamp(t0);
    fm->add_field(f);
    ++count;
  }

  return fm;
}

// Returns fields after initialization
void write (const int seed, const ekat::Comm& comm)
{
  // Create grid
  auto gm = get_gm(comm);
  auto grid = gm->get_grid("point_grid");

  // Time advance parameters
  auto t0 = get_t0();
  const int dt = 86400*30; // 30 days

  // Create some fields
  auto fm = get_fm(grid,t0,seed);
  std::vector<std::string> fnames;
  for (auto it : fm->get_repo()) {
    fnames.push_back(it.second->name());
  }

  // Create output params
  ekat::ParameterList om_pl;
  om_pl.set("filename_prefix",std::string("io_monthly"));
  om_pl.set("field_names",fnames);
  om_pl.set("averaging_type", std::string("instant"));
  om_pl.set("file_max_storage_type",std::string("one_month"));
  om_pl.set("floating_point_precision",std::string("single"));
  auto& ctrl_pl = om_pl.sublist("output_control");
  ctrl_pl.set("frequency_units",std::string("nsteps"));
  ctrl_pl.set("frequency",1);
  ctrl_pl.set("save_grid_data",false);

  // Create Output manager
  OutputManager om;
  om.initialize(comm,om_pl,t0,false);
  om.setup(fm,gm->get_grid_names());

  // Time loop: do 11 steps, since we already did Jan output at t0
  const int nsteps = 11;
  auto t = t0;
  for (int n=0; n<nsteps; ++n) {
    // Update time
    t += dt;

    om.init_timestep(t,dt);

    // Add 1 to all fields entries
    for (const auto& name : fnames) {
      auto f = fm->get_field(name);
      add(f,1);
    }

    // Run output manager
    om.run (t);
  }

  // Close file and cleanup
  om.finalize();
}

void read (const int seed, const ekat::Comm& comm)
{
  // Time quantities
  auto t0 = get_t0();
  int dt = 86400*30;

  // Get gm
  auto gm = get_gm (comm);
  auto grid = gm->get_grid("point_grid");
  auto gids = grid->get_partitioned_dim_gids();

  // Get initial fields. Use wrong seed for fm, so fields are not
  // inited with right data (avoid getting right answer without reading).
  auto fm0 = get_fm(grid,t0,seed);
  std::vector<Field> fields;
  for (auto it : fm0->get_repo()) {
    fields.push_back(it.second->clone());
  }

  // Get filename from timestamp
  std::string casename = "io_monthly";
  auto get_filename = [&](const util::TimeStamp& t) {
    auto t_str = t.to_string().substr(0,7);
    std::string fname = casename
                      + ".INSTANT.nsteps_x1"
                      + ".np" + std::to_string(comm.size())
                      + "." + t_str
                      + ".nc";
    return fname;
  };

  for (int n=0; n<12; ++n) {
    auto t = t0 + n*dt;
    auto filename = get_filename(t);

    // There should be just one time snapshot per file
    REQUIRE(scorpio::get_dimlen(filename,"time")==1);

    read_fields (filename,fields,gids,comm);

    for (const auto& f : fields) {
      auto f0 = fm0->get_field(f.name()).clone(CloneFlags::CopyData);
      add(f0,n);
      REQUIRE (views_are_equal(f,f0));
    }
  }
}

// Same as write, but with monthly *averages*, one per file. The avg window
// ends on the 1st of the next month, so this checks that the file the snapshot
// goes in is determined by the *start* of the avg window, and not by its end.
void write_avg (const int seed, const ekat::Comm& comm)
{
  // Create grid
  auto gm = get_gm(comm);
  auto grid = gm->get_grid("point_grid");

  // Time advance parameters
  auto t0 = get_t0_avg();
  const int dt = 86400; // 1 day

  // Create some fields
  auto fm = get_fm(grid,t0,seed);
  std::vector<std::string> fnames;
  for (auto it : fm->get_repo()) {
    fnames.push_back(it.second->name());
  }

  // Create output params
  ekat::ParameterList om_pl;
  om_pl.set("filename_prefix",std::string("io_monthly_avg"));
  om_pl.set("field_names",fnames);
  om_pl.set("averaging_type", std::string("average"));
  om_pl.set("file_max_storage_type",std::string("one_month"));
  om_pl.set("floating_point_precision",std::string("single"));
  auto& ctrl_pl = om_pl.sublist("output_control");
  ctrl_pl.set("frequency_units",std::string("nmonths"));
  ctrl_pl.set("frequency",1);
  ctrl_pl.set("save_grid_data",false);

  // Create Output manager
  OutputManager om;
  om.initialize(comm,om_pl,t0,false);
  om.setup(fm,gm->get_grid_names());

  // Time loop: one step per day, for num_months months
  auto t = t0;
  for (int m=0; m<num_months; ++m) {
    const int ndays = t.days_in_curr_month();
    for (int n=0; n<ndays; ++n) {
      om.init_timestep(t,dt);

      // Update time
      t += dt;

      // Add 1 to all fields entries
      for (const auto& name : fnames) {
        auto f = fm->get_field(name);
        add(f,1);
      }

      // Run output manager
      om.run (t);
    }

    // Since each file stores one month, and each month stores one avg,
    // the file must be closed as soon as this month's avg was written
    REQUIRE (not om.output_file_specs().is_open);
  }

  // Close file and cleanup
  om.finalize();
}

void read_avg (const ekat::Comm& comm)
{
  // Get filename from timestamp
  std::string casename = "io_monthly_avg";
  auto get_filename = [&](const util::TimeStamp& t) {
    auto t_str = t.to_string().substr(0,7);
    std::string fname = casename
                      + ".AVERAGE.nmonths_x1"
                      + ".np" + std::to_string(comm.size())
                      + "." + t_str
                      + ".nc";
    return fname;
  };

  auto t = get_t0_avg();
  int days = 0;
  for (int m=0; m<num_months; ++m) {
    const int ndays = t.days_in_curr_month();
    days += ndays;

    // The avg of month m must be the ONLY snapshot in the file of month m,
    // and its timestamp must be the end of the avg window
    auto times = scorpio::get_all_times(get_filename(t));
    REQUIRE (times.size()==1);
    REQUIRE (times[0]==days);

    t += ndays*86400;
  }
}

TEST_CASE ("io_monthly") {
  ekat::Comm comm(MPI_COMM_WORLD);
  scorpio::init_subsystem(comm);

  auto seed = get_random_test_seed(&comm);

  if (comm.am_i_root()) {
    std::cout << "   -> Testing output with one file per month ...\n";
  }
  write(seed,comm);
  read (seed,comm);
  if (comm.am_i_root()) {
    std::cout << "   -> Testing output with one file per month ... PASS\n";
  }

  if (comm.am_i_root()) {
    std::cout << "   -> Testing monthly averages with one file per month ...\n";
  }
  write_avg(seed,comm);
  read_avg (comm);
  if (comm.am_i_root()) {
    std::cout << "   -> Testing monthly averages with one file per month ... PASS\n";
  }
  scorpio::finalize_subsystem();
}

} // anonymous namespace
