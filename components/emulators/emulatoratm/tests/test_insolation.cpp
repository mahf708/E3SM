// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "physics/insolation.hpp"

#include "emulator_test_support.hpp"
#include "grid_field_reader.hpp"
#include "scrip_reader.hpp"

#include <cmath>
#include <filesystem>
#include <numbers>
#include <tuple>
#include <vector>

namespace emulator {
namespace physics {
namespace test {

namespace {

/// A regular 2-degree grid, row-major, with its cell areas.
struct Grid {
  std::vector<double> lat, lon, area;
  Grid() {
    for (int j = 0; j < 90; ++j) {
      const double lo = -90.0 + 2.0 * j;
      const double s = std::sin((lo + 2.0) * std::numbers::pi / 180) -
                       std::sin(lo * std::numbers::pi / 180);
      for (int i = 0; i < 180; ++i) {
        lat.push_back(lo + 1.0);
        lon.push_back(1.0 + 2.0 * i);
        area.push_back(s);
      }
    }
  }
  double mean(const std::vector<double> &f) const {
    double num = 0, den = 0;
    for (std::size_t k = 0; k < f.size(); ++k) {
      num += f[k] * area[k];
      den += area[k];
    }
    return num / den;
  }
};

/// A circular orbit with the equinox at calday 80.5: no eccentricity, so
/// eccf is 1 and the declination is set by obliquity alone.
Orbit circular() {
  Orbit o;
  o.obliqr = 23.44 * std::numbers::pi / 180.0;
  return o;
}

double rms_difference(const std::vector<double> &a,
                      const std::vector<double> &b, const Grid &g) {
  std::vector<double> sq(a.size());
  for (std::size_t k = 0; k < a.size(); ++k) {
    sq[k] = (a[k] - b[k]) * (a[k] - b[k]);
  }
  return std::sqrt(g.mean(sq));
}

} // namespace

TEST_CASE("Julian day on the NO_LEAP calendar", "[insolation]") {
  REQUIRE(julian_day_noleap(20000101, 0) == 1.0);
  REQUIRE(julian_day_noleap(19710301, 43200) == 60.5);
  REQUIRE(julian_day_noleap(20001231, 86400) == 366.0);
  REQUIRE_THROWS_AS(julian_day_noleap(20000229, 0), std::invalid_argument);
  REQUIRE_THROWS_AS(julian_day_noleap(20001301, 0), std::invalid_argument);
}

TEST_CASE("Declination follows the orbit", "[insolation]") {
  double delta = 1, eccf = 0;
  solar_declination(80.5, circular(), delta, eccf);
  REQUIRE(delta == Approx(0.0).margin(1e-15));
  REQUIRE(eccf == 1.0);
  solar_declination(80.5 + 365.0 / 4, circular(), delta, eccf);
  REQUIRE(delta == Approx(23.44 * std::numbers::pi / 180.0));

  Orbit eccentric = circular();
  eccentric.eccen = 0.0167;
  solar_declination(3.0, eccentric, delta, eccf);
  REQUIRE(eccf > 0.96);
  REQUIRE(eccf < 1.04);
}

TEST_CASE("The present-day orbit puts perihelion in early January",
          "[insolation]") {
  // 1990 elements, as shr_orb_params reports them for iyear_AD = 1990.
  const auto orbit = Orbit::from_elements(0.016715, 23.4441, 102.7);
  double best_day = 0, best = 0;
  for (double day = 1; day <= 365; day += 0.5) {
    double delta = 0, eccf = 0;
    solar_declination(day, orbit, delta, eccf);
    if (eccf > best) {
      best = eccf;
      best_day = day;
    }
  }
  REQUIRE(best_day >= 1.0);
  REQUIRE(best_day <= 8.0);
  REQUIRE(best == Approx(1.0 / ((1 - 0.016715) * (1 - 0.016715))).epsilon(1e-3));
}

TEST_CASE("At an equinox noon the sun is overhead on the equator at 0 E",
          "[insolation]") {
  const std::vector<double> lat{0.0, 60.0}, lon{0.0, 0.0};
  Insolation sun(circular(), lat, lon);
  std::vector<double> s(2);
  // calday 80.5 is 21 March 12:00 on NO_LEAP.
  sun.instantaneous(20000321, 43200, s);
  REQUIRE(s[0] == Approx(solar_constant).epsilon(1e-6));
  REQUIRE(s[1] == Approx(solar_constant * 0.5).epsilon(1e-3));
}

TEST_CASE("The window mean and the instantaneous field share a global mean "
          "and nothing else", "[insolation]") {
  const Grid g;
  Insolation sun(circular(), g.lat, g.lon);
  std::vector<double> now(g.lat.size()), window(g.lat.size());
  sun.instantaneous(20000321, 21600, now);
  sun.window_mean(20000321, 0, 21600, window);

  // S0 / 4 either way: a global budget cannot tell them apart.
  REQUIRE(g.mean(now) == Approx(solar_constant / 4).epsilon(0.01));
  REQUIRE(g.mean(window) == Approx(solar_constant / 4).epsilon(0.01));
  // Point by point they are different fields.
  REQUIRE(rms_difference(now, window, g) > 200.0);
}

TEST_CASE("Forty-eight sub-steps converge the six-hour mean",
          "[insolation]") {
  const Grid g;
  Insolation sun(circular(), g.lat, g.lon);
  std::vector<double> coarse(g.lat.size()), fine(g.lat.size());
  sun.window_mean(20000615, 3600, 21600, coarse, 48);
  sun.window_mean(20000615, 3600, 21600, fine, 2400);
  REQUIRE(rms_difference(coarse, fine, g) < 0.1);
}

TEST_CASE("Polar night has no sun in the window", "[insolation]") {
  const std::vector<double> lat{-89.0, 89.0}, lon{10.0, 10.0};
  Insolation sun(circular(), lat, lon);
  std::vector<double> s(2);
  sun.window_mean(20000621, 0, 21600, s); // near the June solstice
  REQUIRE(s[0] == 0.0);
  REQUIRE(s[1] > 400.0);
}

TEST_CASE("E3SMv3's SOLIN is the six-hour mean half an hour late, at "
          "1360.53 W/m2", "[insolation][real]") {
  const std::string dir =
      "/pscratch/sd/m/mahf708/SamudrACE-E3SMv3/forcing_data/";
  if (!grid::have_scrip_reader() ||
      !std::filesystem::exists(dir + "solin-0000.nc")) {
    WARN("skipped: needs netCDF and SamudrACE's forcing SOLIN");
    return;
  }
  const auto g = grid::read_scrip(emulator::test::kGaussianGrid);
  std::vector<double> w(g.size());
  for (std::size_t k = 0; k < w.size(); ++k) {
    w[k] = std::cos(g.lat[k] * std::numbers::pi / 180.0);
  }
  auto rms = [&](const std::vector<double> &a, const std::vector<double> &b) {
    double num = 0, den = 0;
    for (std::size_t k = 0; k < a.size(); ++k) {
      num += w[k] * (a[k] - b[k]) * (a[k] - b[k]);
      den += w[k];
    }
    return std::sqrt(num / den);
  };
  const auto orbit = Orbit::from_elements(0.016715, 23.4441, 102.7);
  const Insolation fitted(orbit, g.lat, g.lon, 1360.53);
  const Insolation eatm(orbit, g.lat, g.lon);
  // The stamps' steps start six hours earlier: 3 January 06:00 and
  // 27 June 12:00 of year 425.
  for (const auto &[file, ymd, tod] :
       {std::tuple{"solin-0000.nc", 4250103, 21600},
        std::tuple{"solin-0701.nc", 4250627, 43200}}) {
    const auto ref = grid::read_grid_fields(dir + file, {"SOLIN"}, g.ny, g.nx);
    std::vector<double> late(g.size()), plain(g.size());
    fitted.window_mean(ymd, tod, 21600, late, 48, 1800);
    eatm.window_mean(ymd, tod, 21600, plain);
    INFO(file);
    CHECK(rms(late, ref[0].values) < 2.2);
    CHECK(rms(plain, ref[0].values) > 40.0);
  }
}

} // namespace test
} // namespace physics
} // namespace emulator
