// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "ocean_forcing.hpp"
#include "samudra_channels.hpp"

namespace emulator {
namespace ocn {
namespace test {

using fields::InputSource;

TEST_CASE("Samudra has the checkpoint's 102 inputs and 80 outputs",
          "[samudra][channels]") {
  const auto l = samudra_e3smv3();
  REQUIRE(l.model_dt == 432000);
  REQUIRE(l.inputs.size() == 102);
  REQUIRE(l.outputs.size() == 80);
  REQUIRE(l.inputs[0] == "LANDFRAC");
  REQUIRE(l.inputs[2] == "TAUX");
  REQUIRE(l.inputs[11] == "SHFLX");
  REQUIRE(l.inputs[12] == "sst");          // state block starts at 13 (1-based)
  REQUIRE(l.inputs[91] == "iceVolumeTotal");
  REQUIRE(l.inputs[92] == "TAUX:next");
  REQUIRE(l.outputs[2] == "salinityCoarsened_0");
  REQUIRE(l.outputs[21] == "temperatureCoarsened_0");
  REQUIRE(l.outputs[78] == "ocean_sea_ice_fraction");
  REQUIRE(l.source("sea_surface_fraction") == InputSource::Boundary);
  REQUIRE(l.source("FSDS") == InputSource::Forcing);
  REQUIRE(l.source("FSDS:next") == InputSource::Forcing);
  REQUIRE(l.source("velocityZonalCoarsened_18") == InputSource::Prognostic);
}

namespace {

struct Coupler {
  fields::FieldSet in{1};
  fields::FieldSet out{1};
  Coupler() {
    for (const auto &n : coupler_forcing_imports()) {
      in.add(n);
    }
    for (const auto &n : samudra_forcing_names()) {
      out.add(n);
    }
  }
  void set(const char *n, double v) { in.get(n)[0] = v; }
  double get(const char *n) { return out.get(n)[0]; }
};

} // namespace

TEST_CASE("Coupler forcing is unweighted by the ice fraction, with the "
          "checkpoint's signs", "[samudra][forcing]") {
  Coupler c;
  c.set("Si_ifrac", 0.5); // the coupler's fluxes cover half the cell
  c.set("Foxx_taux", 0.1);
  c.set("Foxx_tauy", -0.2);
  c.set("Faxa_rain", 2e-5);
  c.set("Faxa_snow", 1e-5);
  c.set("Foxx_lwup", -150.0); // coupler: positive down
  c.set("Faxa_lwdn", 160.0);
  c.set("Foxx_swnet", 94.0);
  c.set("Foxx_lat", -40.0);
  c.set("Foxx_sen", -10.0);
  coupler_forcing_sample(c.in, {}, c.out);

  REQUIRE(c.get("TAUX") == -0.1); // stress not unweighted by default
  REQUIRE(c.get("TAUY") == 0.2);
  REQUIRE(c.get("surface_precipitation_rate") == Approx(6e-5));
  REQUIRE(c.get("frozen_precipitation_rate") == Approx(2e-5));
  REQUIRE(c.get("FLUS") == 300.0);  // positive up, whole cell
  REQUIRE(c.get("FLDS") == 320.0);
  REQUIRE(c.get("FSDS") == Approx(200.0)); // 2 * 94 / 0.94
  REQUIRE(c.get("FSUS") == Approx(12.0));  // FSDS - 2 * 94
  REQUIRE(c.get("LHFLX") == 80.0);
  REQUIRE(c.get("SHFLX") == 20.0);

  SECTION("the unweighting is floored at 1%") {
    c.set("Si_ifrac", 1.0);
    coupler_forcing_sample(c.in, {}, c.out);
    REQUIRE(c.get("FLDS") == Approx(16000.0));
  }
  SECTION("and can be switched off, stress on separately") {
    CouplerForcingOptions opt;
    opt.unweight_by_ice_fraction = false;
    coupler_forcing_sample(c.in, opt, c.out);
    REQUIRE(c.get("FLDS") == 160.0);
    opt.unweight_by_ice_fraction = true;
    opt.unweight_stress = true;
    coupler_forcing_sample(c.in, opt, c.out);
    REQUIRE(c.get("TAUX") == Approx(-0.2));
  }
}

TEST_CASE("Precipitation is clipped after the window mean, not before",
          "[samudra][forcing]") {
  fields::FieldSet mean(2);
  auto p = mean.add("surface_precipitation_rate");
  auto s = mean.add("frozen_precipitation_rate");
  p[0] = -1e-7;
  p[1] = 3e-5;
  s[0] = -2e-7;
  s[1] = 0.0;
  clip_after_mean(mean);
  REQUIRE(p[0] == 0.0);
  REQUIRE(p[1] == 3e-5);
  REQUIRE(s[0] == 0.0);
}

} // namespace test
} // namespace ocn
} // namespace emulator
