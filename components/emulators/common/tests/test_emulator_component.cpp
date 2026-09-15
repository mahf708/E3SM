// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "emulator_component.hpp"
#include "emulator_test_support.hpp"
#include "grid_field_reader.hpp"
#include "scrip_reader.hpp"

#include <numeric>

namespace emulator {
namespace test {

namespace {
int fcomm() { return MPI_Comm_c2f(MPI_COMM_WORLD); }
} // namespace

TEST_CASE("Without an input file a component is unconfigured; a missing or "
          "misspelt one is refused", "[component]") {
  EmulatorComponent none(EmulatorType::ATM_COMP, "emulatoratm");
  none.create_instance(fcomm(), 1, "", "", 0, 20000101, 0);
  REQUIRE_FALSE(none.configured());
  REQUIRE_FALSE(none.has_domain());
  REQUIRE(none.get_num_local_cols() == 0);

  EmulatorComponent missing(EmulatorType::ATM_COMP, "emulatoratm");
  REQUIRE_THROWS_WITH(
      missing.create_instance(fcomm(), 1, "/no/such/atm_in", "", 0, 20000101, 0),
      Catch::Contains("/no/such/atm_in") && Catch::Contains("does not exist"));

  const TempFile typo("atm_in", "spec: " + spec_path("ace2-eamv3.yaml") +
                                    "\ncoupler_dt: 1800\ngrid: {domain: full}\n"
                                    "initial_conditon: x.nc\n");
  EmulatorComponent misspelt(EmulatorType::ATM_COMP, "emulatoratm");
  REQUIRE_THROWS_WITH(
      misspelt.create_instance(fcomm(), 1, typo.path, "", 0, 20000101, 0),
      Catch::Contains("initial_conditon") &&
          Catch::Contains("initial_condition"));
}

TEST_CASE("A component that takes another's domain refuses to run without it",
          "[component]") {
  const TempFile spec("shared.yaml", "name: shared-surface\ncoupler: {exports: []}\n");
  const TempFile ice_in("ice_in", "spec: " + spec.path +
                                      "\ncoupler_dt: 1800\n"
                                      "grid: {domain: shared, shared_from: ocn}\n");
  coupling::Exchange empty;
  EmulatorComponent ice(EmulatorType::ICE_COMP, "emulatorice", empty);
  REQUIRE_THROWS_WITH(
      ice.create_instance(fcomm(), 5, ice_in.path, "", 0, 20000101, 0),
      Catch::Contains("no domain from 'ocn'") && Catch::Contains("NTASKS"));
  REQUIRE_FALSE(ice.has_domain());
}

TEST_CASE("A full-domain component reads its grid and splits it over the "
          "ranks", "[component][grid][real]") {
  if (!grid::have_scrip_reader() || !std::filesystem::exists(kGaussianGrid)) {
    WARN("skipped: needs netCDF and " << kGaussianGrid);
    return;
  }
  const TempFile atm_in("atm_in", "spec: " + spec_path("ace2-eamv3.yaml") +
                                      "\ncoupler_dt: 1800\ngrid: {file: " +
                                      kGaussianGrid + ", domain: full}\n");
  EmulatorComponent atm(EmulatorType::ATM_COMP, "emulatoratm");
  atm.create_instance(fcomm(), 1, atm_in.path, "", 0, 20000101, 0);
  REQUIRE(atm.has_domain());
  REQUIRE(atm.get_nx() == 360);
  REQUIRE(atm.get_ny() == 180);
  REQUIRE(atm.get_num_global_cols() == 64800);

  const int n = atm.get_num_local_cols();
  std::vector<int> gids(n);
  std::vector<double> lat(n), lon(n), mask(n), frac(n);
  atm.get_local_col_gids(gids.data());
  atm.get_cols_latlon(lat.data(), lon.data());
  atm.get_cols_mask_frac(mask.data(), frac.data());

  // Every rank's coordinates are the file's, cell for cell: this is the
  // comparison the coupler's grid check makes, at 1e-12.
  const auto g = grid::read_scrip(kGaussianGrid);
  for (int i = 0; i < n; ++i) {
    REQUIRE(lat[i] == g.lat[gids[i] - 1]);
    REQUIRE(lon[i] == g.lon[gids[i] - 1]);
  }
  REQUIRE(mask == std::vector<double>(n, 1.0));
  REQUIRE(frac == mask);

  // Together the ranks own every cell once.
  REQUIRE(global_sum(n) == 64800.0);
  const double local_sum = std::accumulate(gids.begin(), gids.end(), 0.0);
  REQUIRE(global_sum(local_sum) == 64800.0 * 64801.0 / 2.0);
}

} // namespace test
} // namespace emulator

EMULATOR_TEST_MPI_MAIN
