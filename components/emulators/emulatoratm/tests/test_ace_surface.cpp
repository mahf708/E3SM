// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "ace_surface.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace emulator {
namespace atm {
namespace test {

namespace {

/// Owns one cell's worth of every export, and hands out spans to it.
struct OneCell {
  std::array<double, 20> v{};
  SurfaceExports spans() {
    SurfaceExports e;
    std::size_t k = 0;
    for (auto *s : {&e.z, &e.u, &e.v, &e.tbot, &e.ptem, &e.shum, &e.pbot,
                    &e.pslv, &e.dens, &e.topo, &e.lwdn, &e.rainc, &e.rainl,
                    &e.snowc, &e.snowl, &e.swndr, &e.swvdr, &e.swndf, &e.swvdf,
                    &e.swnet}) {
      *s = std::span<double>(&v[k++], 1);
    }
    return e;
  }
};

std::span<const double> one(const double &x) { return {&x, 1}; }

} // namespace

TEST_CASE("Saturation vapour pressure is datm's polynomial", "[ace][surface]") {
  // At freezing, over water: the a0 coefficient, in Pa.
  REQUIRE(saturation_vapor_pressure(273.15, 273.15) == Approx(610.7799961));
  // Just below freezing the ice branch applies, and is lower than water.
  REQUIRE(saturation_vapor_pressure(263.15, 263.15) <
          saturation_vapor_pressure(263.15, 273.15));
  REQUIRE(saturation_vapor_pressure(263.15, 263.15) == Approx(259.9).margin(1));
  // Clamped at +-50 C.
  REQUIRE(saturation_vapor_pressure(400.0, 300.0) ==
          saturation_vapor_pressure(323.15, 300.0));
}

TEST_CASE("Near-surface exports sit at 10 m with pbot = PS", "[ace][surface]") {
  const double ps = 101000, phis = 9.80616 * 50, t2 = 290, q2 = 0.008,
               u10 = 5, v10 = -2, flds = 350, fsds = 200, precip = 2e-5,
               frozen = 5e-6;
  SurfaceInputs in;
  in.ps = one(ps);
  in.phis = one(phis);
  in.t_2m = one(t2);
  in.q_2m = one(q2);
  in.u_10m = one(u10);
  in.v_10m = one(v10);
  in.flds = one(flds);
  in.fsds = one(fsds);
  in.precip = one(precip);
  in.frozen_precip = one(frozen);
  SurfaceOptions opt;
  opt.diurnal_shortwave = false;
  OneCell cell;
  auto out = cell.spans();

  const auto counts = compute_surface_exports(in, opt, out);
  REQUIRE(out.z[0] == 10.0);
  REQUIRE(out.pbot[0] == ps);
  REQUIRE(out.pslv[0] == ps);
  REQUIRE(out.tbot[0] == t2);
  REQUIRE(out.ptem[0] == t2); // (PS/pbot)^k == 1
  REQUIRE(out.u[0] == u10);
  REQUIRE(out.topo[0] == Approx(50.0));
  REQUIRE(out.dens[0] ==
          Approx(ps / (constants::rdair * t2 * (1 + 0.608 * q2))));
  REQUIRE(out.lwdn[0] == flds);
  REQUIRE(out.snowl[0] == frozen);
  REQUIRE(out.rainl[0] == Approx(precip - frozen));
  REQUIRE(out.rainc[0] == 0.0);
  REQUIRE(out.swvdr[0] + out.swndr[0] + out.swvdf[0] + out.swndf[0] ==
          Approx(fsds));
  REQUIRE(out.swnet[0] == fsds); // no FSUS channel
  REQUIRE(counts.capped_humidity == 0);
}

TEST_CASE("The lowest level sits at its midpoint, about 450 m up",
          "[ace][surface]") {
  const double ps = 100000, phis = 0, t7 = 285, q7 = 0.005, u7 = 7, v7 = 1,
               flds = 300, fsds = 0, precip = 1e-5;
  SurfaceInputs in;
  in.ps = one(ps);
  in.phis = one(phis);
  in.t_lowest = one(t7);
  in.q_lowest = one(q7);
  in.u_lowest = one(u7);
  in.v_lowest = one(v7);
  in.flds = one(flds);
  in.fsds = one(fsds);
  in.precip = one(precip);
  SurfaceOptions opt;
  opt.layer = SurfaceLayer::LowestLevel;
  opt.diurnal_shortwave = false;
  OneCell cell;
  auto out = cell.spans();
  compute_surface_exports(in, opt, out);

  const double p_int = constants::ak_bot + constants::bk_bot * ps;
  REQUIRE(out.pbot[0] == Approx(0.5 * (ps + p_int)));
  REQUIRE(out.z[0] > 400.0);
  REQUIRE(out.z[0] < 500.0);
  REQUIRE(out.ptem[0] > t7);
  // No frozen channel: all precipitation is rain above freezing.
  REQUIRE(out.rainl[0] == precip);
  REQUIRE(out.snowl[0] == 0.0);
}

TEST_CASE("Humidity is capped at saturation, and the cap is counted",
          "[ace][surface]") {
  const double ps = 100000, phis = 0, t = 280, q = 0.05, u = 0, v = 0,
               flds = 0, fsds = 0, precip = 0;
  SurfaceInputs in;
  in.ps = one(ps);
  in.phis = one(phis);
  in.t_lowest = one(t);
  in.q_lowest = one(q);
  in.u_lowest = one(u);
  in.v_lowest = one(v);
  in.flds = one(flds);
  in.fsds = one(fsds);
  in.precip = one(precip);
  SurfaceOptions opt;
  opt.layer = SurfaceLayer::LowestLevel;
  opt.diurnal_shortwave = false;
  OneCell cell;
  auto out = cell.spans();
  const auto counts = compute_surface_exports(in, opt, out);

  const double e = saturation_vapor_pressure(t, t);
  const double qsat = 0.622 * e / (out.pbot[0] - 0.378 * e);
  REQUIRE(out.shum[0] == Approx(qsat));
  REQUIRE(counts.capped_humidity == 1);
  REQUIRE(counts.max_relative_humidity == Approx(q / qsat));
}

TEST_CASE("Precipitation below freezing, and frozen precipitation in m/s",
          "[ace][surface]") {
  const double ps = 100000, phis = 0, t = 260, q = 0.001, w = 0, flds = 0,
               fsds = 0, precip = 1e-5, frozen_m_per_s = 4e-9;
  SurfaceInputs in;
  in.ps = one(ps);
  in.phis = one(phis);
  in.t_lowest = one(t);
  in.q_lowest = one(q);
  in.u_lowest = one(w);
  in.v_lowest = one(w);
  in.flds = one(flds);
  in.fsds = one(fsds);
  in.precip = one(precip);
  SurfaceOptions opt;
  opt.layer = SurfaceLayer::LowestLevel;
  opt.diurnal_shortwave = false;
  OneCell cell;
  auto out = cell.spans();

  compute_surface_exports(in, opt, out);
  REQUIRE(out.snowl[0] == precip); // temperature split
  REQUIRE(out.rainl[0] == 0.0);

  in.frozen_precip = one(frozen_m_per_s);
  opt.frozen_precip_in_m_per_s = true;
  compute_surface_exports(in, opt, out);
  REQUIRE(out.snowl[0] == Approx(4e-6)); // x1000 kg/m3
  REQUIRE(out.rainl[0] == Approx(6e-6));

  // Frozen never exceeds total.
  const double huge = 1.0;
  in.frozen_precip = one(huge);
  compute_surface_exports(in, opt, out);
  REQUIRE(out.snowl[0] == precip);
  REQUIRE(out.rainl[0] == 0.0);
}

TEST_CASE("Window-mean shortwave follows the sun within the window",
          "[ace][surface]") {
  const double ps = 100000, phis = 0, t = 290, q = 0.01, w = 0, flds = 0,
               fsds = 300, fsus = 60, precip = 0;
  double solin_now = 900, solin_window = 450;
  SurfaceInputs in;
  in.ps = one(ps);
  in.phis = one(phis);
  in.t_2m = one(t);
  in.q_2m = one(q);
  in.u_10m = one(w);
  in.v_10m = one(w);
  in.flds = one(flds);
  in.fsds = one(fsds);
  in.fsus = one(fsus);
  in.precip = one(precip);
  in.solin_now = one(solin_now);
  in.solin_window = one(solin_window);
  SurfaceOptions opt;
  OneCell cell;
  auto out = cell.spans();

  compute_surface_exports(in, opt, out);
  const double down = 300.0 * 900.0 / 450.0;
  REQUIRE(out.swvdr[0] == Approx(0.28 * down));
  REQUIRE(out.swnet[0] == Approx(down - 60.0 * 2.0));

  solin_now = 0; // the sun has set, inside a window that had some
  compute_surface_exports(in, opt, out);
  REQUIRE(out.swvdr[0] == 0.0);
  REQUIRE(out.swnet[0] == 0.0);

  solin_window = 0.5; // polar night
  solin_now = 0.4;
  compute_surface_exports(in, opt, out);
  REQUIRE(out.swndf[0] == 0.0);
}

TEST_CASE("Options that need missing channels are refused, not guessed",
          "[ace][surface]") {
  const double x = 1.0;
  SurfaceInputs in;
  in.ps = one(x);
  in.phis = one(x);
  in.t_lowest = one(x);
  in.q_lowest = one(x);
  in.u_lowest = one(x);
  in.v_lowest = one(x);
  in.flds = one(x);
  in.fsds = one(x);
  in.precip = one(x);
  OneCell cell;
  auto out = cell.spans();

  SurfaceOptions near; // ACE2 has no 2 m / 10 m channels
  near.diurnal_shortwave = false;
  REQUIRE_THROWS_WITH(compute_surface_exports(in, near, out),
                      Catch::Contains("not chosen silently"));

  SurfaceOptions diurnal;
  diurnal.layer = SurfaceLayer::LowestLevel;
  REQUIRE_THROWS_WITH(compute_surface_exports(in, diurnal, out),
                      Catch::Contains("insolation"));
}

} // namespace test
} // namespace atm
} // namespace emulator
