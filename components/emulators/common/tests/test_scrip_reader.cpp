// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "grid_field_reader.hpp"
#include "scrip_reader.hpp"

#include <netcdf.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

#include <unistd.h>

namespace emulator {
namespace grid {
namespace test {

namespace {

struct ScripOptions {
  const char *coord_units = "degrees";
  const char *area_units = "steradian";
  bool with_imask = true;
  bool radians = false;
  int nrows = 2; ///< 2 covers the sphere; 1 drops the southern band
};

void ok(int status) {
  if (status != NC_NOERR) {
    FAIL("netCDF: " << nc_strerror(status));
  }
}

/// Write the 4 x 2 global grid from test_grid.cpp as a SCRIP file.
std::string write_scrip(const std::string &name, const ScripOptions &opt) {
  const auto path =
      (std::filesystem::temp_directory_path() /
       ("emulator_scrip_" + name + "_" + std::to_string(::getpid()) + ".nc"))
          .string();

  const int nx = 4;
  const int ny = opt.nrows;
  const int n = nx * ny;
  std::vector<double> lat, lon, area;
  std::vector<int> imask;
  const double to_units = opt.radians ? std::numbers::pi / 180.0 : 1.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      lat.push_back((j == 0 && ny == 2 ? -45.0 : 45.0) * to_units);
      lon.push_back((45.0 + 90.0 * i) * to_units);
      area.push_back(std::numbers::pi / 2.0);
      imask.push_back(i == 0 ? 0 : 1);
    }
  }
  const std::vector<int> dims{nx, ny};

  int ncid = 0;
  ok(nc_create(path.c_str(), NC_CLOBBER, &ncid));
  int d_size = 0, d_rank = 0;
  ok(nc_def_dim(ncid, "grid_size", static_cast<size_t>(n), &d_size));
  ok(nc_def_dim(ncid, "grid_rank", 2, &d_rank));
  int v_lat = 0, v_lon = 0, v_area = 0, v_mask = 0, v_dims = 0;
  ok(nc_def_var(ncid, "grid_center_lat", NC_DOUBLE, 1, &d_size, &v_lat));
  ok(nc_def_var(ncid, "grid_center_lon", NC_DOUBLE, 1, &d_size, &v_lon));
  ok(nc_def_var(ncid, "grid_area", NC_DOUBLE, 1, &d_size, &v_area));
  ok(nc_def_var(ncid, "grid_dims", NC_INT, 1, &d_rank, &v_dims));
  if (opt.with_imask) {
    ok(nc_def_var(ncid, "grid_imask", NC_INT, 1, &d_size, &v_mask));
  }
  if (opt.coord_units) {
    ok(nc_put_att_text(ncid, v_lat, "units", std::strlen(opt.coord_units),
                       opt.coord_units));
    ok(nc_put_att_text(ncid, v_lon, "units", std::strlen(opt.coord_units),
                       opt.coord_units));
  }
  ok(nc_put_att_text(ncid, v_area, "units", std::strlen(opt.area_units),
                     opt.area_units));
  ok(nc_enddef(ncid));
  ok(nc_put_var_double(ncid, v_lat, lat.data()));
  ok(nc_put_var_double(ncid, v_lon, lon.data()));
  ok(nc_put_var_double(ncid, v_area, area.data()));
  ok(nc_put_var_int(ncid, v_dims, dims.data()));
  if (opt.with_imask) {
    ok(nc_put_var_int(ncid, v_mask, imask.data()));
  }
  ok(nc_close(ncid));
  return path;
}

struct Cleanup {
  std::string path;
  ~Cleanup() { std::remove(path.c_str()); }
};

} // namespace

TEST_CASE("This build reads SCRIP", "[scrip]") {
  REQUIRE(have_scrip_reader());
}

TEST_CASE("A SCRIP file in degrees reads as written", "[scrip]") {
  Cleanup file{write_scrip("deg", {})};
  const auto g = read_scrip(file.path);
  REQUIRE(g.nx == 4);
  REQUIRE(g.ny == 2);
  REQUIRE(g.size() == 8);
  REQUIRE(g.lat[0] == -45.0);
  REQUIRE(g.lat[7] == 45.0);
  REQUIRE(g.lon[3] == 315.0);
  REQUIRE(g.imask[0] == 0);
  REQUIRE(g.imask[1] == 1);
  REQUIRE(g.total_area() == Approx(4.0 * std::numbers::pi));
}

TEST_CASE("A SCRIP file in radians is converted to degrees", "[scrip]") {
  ScripOptions opt;
  opt.coord_units = "radians";
  opt.radians = true;
  Cleanup file{write_scrip("rad", opt)};
  const auto g = read_scrip(file.path);
  REQUIRE(g.lat[7] == Approx(45.0).epsilon(1e-14));
  REQUIRE(g.lon[3] == Approx(315.0).epsilon(1e-14));
}

TEST_CASE("A SCRIP file without grid_imask is all active", "[scrip]") {
  ScripOptions opt;
  opt.with_imask = false;
  Cleanup file{write_scrip("nomask", opt)};
  const auto g = read_scrip(file.path);
  REQUIRE(g.imask == std::vector<int>(8, 1));
}

TEST_CASE("A SCRIP reader refuses to guess", "[scrip]") {
  SECTION("coordinates with no units") {
    ScripOptions opt;
    opt.coord_units = nullptr;
    Cleanup file{write_scrip("nounits", opt)};
    REQUIRE_THROWS_WITH(read_scrip(file.path),
                        Catch::Contains("refusing to guess") &&
                            Catch::Contains("grid_center_lat"));
  }
  SECTION("areas in square degrees") {
    ScripOptions opt;
    opt.area_units = "degrees^2";
    Cleanup file{write_scrip("sqdeg", opt)};
    REQUIRE_THROWS_WITH(read_scrip(file.path),
                        Catch::Contains("radians^2"));
  }
  SECTION("a grid that does not cover the sphere, when one should") {
    ScripOptions opt;
    opt.nrows = 1;
    Cleanup file{write_scrip("half", opt)};
    REQUIRE_THROWS_WITH(read_scrip(file.path), Catch::Contains("4 pi"));
    REQUIRE_NOTHROW(read_scrip(file.path, false));
  }
  SECTION("a file that is not there") {
    REQUIRE_THROWS_WITH(read_scrip("/nonexistent/grid.nc"),
                        Catch::Contains("/nonexistent/grid.nc") &&
                            Catch::Contains("cannot open"));
  }
}

TEST_CASE("The Gaussian grid ACE runs on", "[scrip][real]") {
  // A Gaussian grid: an analytic regular 1-degree grid would not match it to
  // the tolerance the coupler checks atmosphere grids with.
  const std::string path = "/global/cfs/cdirs/e3sm/inputdata/share/meshes/"
                           "gaussian_180x360_latlon.scrip.20260127.nc";
  if (!std::filesystem::exists(path)) {
    WARN("skipped: " << path << " is not on this machine");
    return;
  }
  const auto g = read_scrip(path);
  REQUIRE(g.nx == 360);
  REQUIRE(g.ny == 180);
  REQUIRE(g.size() == 64800);
  REQUIRE(g.total_area() == Approx(4.0 * std::numbers::pi).epsilon(1e-10));

  // Gaussian latitudes are symmetric and do not reach the pole: the first
  // row sits near -89.24.
  REQUIRE(g.lat.front() == Approx(-g.lat.back()).margin(1e-10));
  REQUIRE(std::abs(g.lat.front()) > 89.0);
  REQUIRE(std::abs(g.lat.front()) < 89.5);
  // Row-major: the first row is one latitude.
  REQUIRE(g.lat[359] == g.lat[0]);
  REQUIRE(g.lat[360] != g.lat[0]);

  const auto d = Decomposition::contiguous_blocks(g.size(), 4, 3);
  const auto dom = Domain::full(g, d);
  REQUIRE(dom.size() == 16200);
  REQUIRE(dom.lat.back() == g.lat.back());
}

TEST_CASE("Grid fields are read as written, with unusable values counted",
          "[grid_fields]") {
  const auto path = (std::filesystem::temp_directory_path() /
                     ("emulator_fields_" + std::to_string(::getpid()) + ".nc"))
                        .string();
  Cleanup cleanup{path};
  {
    int ncid = 0, dlat = 0, dlon = 0, dt = 0;
    ok(nc_create(path.c_str(), NC_CLOBBER, &ncid));
    ok(nc_def_dim(ncid, "time", 1, &dt));
    ok(nc_def_dim(ncid, "lat", 2, &dlat));
    ok(nc_def_dim(ncid, "lon", 3, &dlon));
    int dims2[2] = {dlat, dlon};
    int dims3[3] = {dt, dlat, dlon};
    int v_ts = 0, v_ice = 0, v_ps = 0;
    ok(nc_def_var(ncid, "TS", NC_FLOAT, 2, dims2, &v_ts));
    ok(nc_def_var(ncid, "ICEFRAC", NC_FLOAT, 2, dims2, &v_ice));
    ok(nc_def_var(ncid, "PS", NC_DOUBLE, 3, dims3, &v_ps));
    const float fill = 9.96921e36f;
    ok(nc_put_att_float(ncid, v_ice, "_FillValue", NC_FLOAT, 1, &fill));
    ok(nc_enddef(ncid));
    const float ts[6] = {270, 271, 272, 280, 281, 282};
    const float ice[6] = {1, NAN, 0.5f, fill, 0, 0};
    const double ps[6] = {1e5, 1e5, 1e5, 9.9e4, 9.9e4, 9.9e4};
    ok(nc_put_var_float(ncid, v_ts, ts));
    ok(nc_put_var_float(ncid, v_ice, ice));
    ok(nc_put_var_double(ncid, v_ps, ps));
    ok(nc_close(ncid));
  }

  const auto fields = read_grid_fields(path, {"TS", "ICEFRAC", "PS"}, 2, 3);
  REQUIRE(fields[0].values[4] == 281.0); // row 1, column 1
  REQUIRE(fields[0].unusable() == 0);
  REQUIRE(fields[1].non_finite == 1);
  REQUIRE(fields[1].fill_like == 1);
  REQUIRE(fields[2].values[5] == 9.9e4); // a leading time of 1 is accepted

  REQUIRE_THROWS_WITH(read_grid_fields(path, {"PHIS"}, 2, 3),
                      Catch::Contains("no variable 'PHIS'"));
  REQUIRE_THROWS_WITH(read_grid_fields(path, {"TS"}, 3, 2),
                      Catch::Contains("'TS' is 2 x 3; expected 3 x 2"));
}

} // namespace test
} // namespace grid
} // namespace emulator
