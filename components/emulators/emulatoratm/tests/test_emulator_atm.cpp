// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "atm.hpp"
#include "scrip_reader.hpp"

#include <mpi.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <unistd.h>

namespace emulator {
namespace test {

namespace {

const std::string kGaussianGrid = "/global/cfs/cdirs/e3sm/inputdata/share/"
                                  "meshes/gaussian_180x360_latlon.scrip."
                                  "20260127.nc";

struct AtmIn {
  std::string path;
  explicit AtmIn(const std::string &contents) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    path = (std::filesystem::temp_directory_path() /
            ("emulator_atm_in_" + std::to_string(::getpid()) + "_" +
             std::to_string(rank)))
               .string();
    std::ofstream(path) << contents;
  }
  ~AtmIn() { std::remove(path.c_str()); }
};

int world_fortran_comm() { return MPI_Comm_c2f(MPI_COMM_WORLD); }

} // namespace

TEST_CASE("The atmosphere refuses grid dimensions without coordinates",
          "[atm][grid]") {
  AtmIn atm_in("nx: 360\nny: 180\n");
  EmulatorAtm atm;
  REQUIRE_THROWS_WITH(atm.create_instance(world_fortran_comm(), 1, atm_in.path,
                                          "", 0, 20000101, 0),
                      Catch::Contains("no grid_file") &&
                          Catch::Contains("latitude 0"));
  REQUIRE_FALSE(atm.has_domain());
}

TEST_CASE("Without a grid file the atmosphere waits for set_grid_data",
          "[atm][grid]") {
  AtmIn atm_in("# nothing about the grid\n");
  EmulatorAtm atm;
  atm.create_instance(world_fortran_comm(), 1, atm_in.path, "", 0, 20000101, 0);
  REQUIRE_FALSE(atm.has_domain());
  REQUIRE(atm.get_num_local_cols() == 0);
}

TEST_CASE("The atmosphere reads its grid and splits it over the ranks",
          "[atm][grid][real]") {
  if (!grid::have_scrip_reader() || !std::filesystem::exists(kGaussianGrid)) {
    WARN("skipped: needs netCDF and " << kGaussianGrid);
    return;
  }
  AtmIn atm_in("grid_file: " + kGaussianGrid + "\n");
  EmulatorAtm atm;
  atm.create_instance(world_fortran_comm(), 1, atm_in.path, "", 0, 20000101, 0);
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
  int total = 0;
  MPI_Allreduce(&n, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(total == 64800);
  long long id_sum = 0;
  const long long local_sum = std::accumulate(gids.begin(), gids.end(), 0LL);
  MPI_Allreduce(&local_sum, &id_sum, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  REQUIRE(id_sum == 64800LL * 64801LL / 2);
}

} // namespace test
} // namespace emulator

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Catch::Session session;
  if (rank != 0) {
    // One report, from rank 0; the others still run every assertion.
    session.configData().outputFilename = "%debug";
  }
  int status = session.run(argc, argv);
  int worst = 0;
  MPI_Allreduce(&status, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return worst;
}
