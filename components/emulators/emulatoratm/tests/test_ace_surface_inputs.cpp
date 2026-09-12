// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "ace_surface_inputs.hpp"

#include <vector>

namespace emulator {
namespace atm {
namespace test {

namespace {

struct Cells {
  std::vector<double> landfrac, ocnfrac, icefrac, ts;
  explicit Cells(std::size_t n) : landfrac(n), ocnfrac(n), icefrac(n), ts(n) {}
  SurfaceChannels spans() { return {landfrac, ocnfrac, icefrac, ts}; }
};

} // namespace

TEST_CASE("With a stub land model the land arrives as the deficit",
          "[ace][surface_inputs]") {
  // A coastal cell: 70% ocean at 290 K, and a land model that reports
  // nothing.  The coupler's merged Sx_t is 0.7 * 290 = 203 K.
  const std::vector<double> lfrac{0.0}, ofrac{0.7}, ifrac{0.0}, sx_t{203.0},
                            ts_emul{280.0};
  Cells out(1);
  auto spans = out.spans();
  compute_surface_inputs({lfrac, ofrac, ifrac, sx_t, ts_emul}, spans);

  // Not LANDFRAC 0 with a 203 K surface: the network was trained on neither.
  REQUIRE(out.landfrac[0] == Approx(0.3));
  REQUIRE(out.ocnfrac[0] == 0.7);
  REQUIRE(out.ts[0] == Approx(203.0 + 0.3 * 280.0));
}

TEST_CASE("Sx_t is added to, not reweighted", "[ace][surface_inputs]") {
  // Fully covered: no deficit, so TS is Sx_t exactly, whatever the
  // emulator predicted.
  const std::vector<double> lfrac{0.4}, ofrac{0.5}, ifrac{0.1}, sx_t{285.0},
                            ts_emul{250.0};
  Cells out(1);
  auto spans = out.spans();
  compute_surface_inputs({lfrac, ofrac, ifrac, sx_t, ts_emul}, spans);
  REQUIRE(out.ts[0] == 285.0);
  REQUIRE(out.landfrac[0] == Approx(0.4));
}

TEST_CASE("Round-off in the fractions is repaired and counted",
          "[ace][surface_inputs]") {
  const std::vector<double> lfrac{0.5, -1e-12}, ofrac{0.5 + 1e-9, 1.0},
                            ifrac{0.0, 0.0}, sx_t{280, 290}, ts_emul{270, 270};
  Cells out(2);
  auto spans = out.spans();
  const auto counts =
      compute_surface_inputs({lfrac, ofrac, ifrac, sx_t, ts_emul}, spans);
  REQUIRE(counts.renormalized == 1);
  REQUIRE(counts.clipped == 1);
  REQUIRE(out.landfrac[0] + out.ocnfrac[0] + out.icefrac[0] == Approx(1.0));
  REQUIRE(out.landfrac[1] == 0.0);
}

TEST_CASE("Fractions wrong by more than round-off stop the run",
          "[ace][surface_inputs]") {
  const std::vector<double> lfrac{0.3}, ofrac{0.9}, ifrac{0.0}, sx_t{280},
                            ts_emul{270};
  Cells out(1);
  auto spans = out.spans();
  REQUIRE_THROWS_WITH(
      compute_surface_inputs({lfrac, ofrac, ifrac, sx_t, ts_emul}, spans),
      Catch::Contains("worst excess 0.2") && Catch::Contains("Sx_t"));
}

TEST_CASE("An emulated ocean's ice and SST re-split the non-land part",
          "[ace][surface_inputs]") {
  // Stub land again, so L = 0.3 from the deficit; the ocean emulator says
  // half of the rest is ice, and its open water is at 275 K.
  const std::vector<double> lfrac{0.0}, ofrac{0.7}, ifrac{0.0}, sx_t{193.0},
                            ts_emul{250.0}, sif{0.5}, sst{275.0};
  Cells out(1);
  auto spans = out.spans();
  compute_surface_inputs({lfrac, ofrac, ifrac, sx_t, ts_emul, sif, sst},
                         spans);
  REQUIRE(out.landfrac[0] == Approx(0.3));
  REQUIRE(out.icefrac[0] == Approx(0.35));
  REQUIRE(out.ocnfrac[0] == Approx(0.35));
  const double open = 0.7 * 0.5;
  REQUIRE(out.ts[0] == Approx(open * 275.0 + (1 - open) * 250.0));

  const std::vector<double> none;
  REQUIRE_THROWS_WITH(
      compute_surface_inputs({lfrac, ofrac, ifrac, sx_t, ts_emul, sif, none},
                             spans),
      Catch::Contains("together"));
}

} // namespace test
} // namespace atm
} // namespace emulator
