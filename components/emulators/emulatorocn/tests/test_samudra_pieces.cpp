// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "ocean_forcing.hpp"
#include "channel_layout_yaml.hpp"
#include "emulator_test_support.hpp"

#include <vector>

namespace emulator {
namespace ocn {
namespace test {

namespace {
fields::ChannelLayout samudra_e3smv3() {
  return fields::read_channel_layout(
      config::Section::load_spec(
          emulator::test::spec_path("samudra-e3smv3-ocean.yaml"))
          .section("network"));
}
/// The channels coupler_forcing_sample writes.
const std::vector<std::string> &samudra_forcing_names() {
  static const std::vector<std::string> names{
      "TAUX", "TAUY", "surface_precipitation_rate", "frozen_precipitation_rate",
      "FLUS", "FSUS", "FLDS", "FSDS", "LHFLX", "SHFLX"};
  return names;
}
} // namespace

using fields::InputSource;

TEST_CASE("The Samudra spec has the checkpoint's 102 inputs and 80 outputs",
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

TEST_CASE("Cell-mean forcing completes the open-water fluxes with the ice "
          "surface's own", "[samudra][forcing]") {
  Coupler c;
  fields::FieldSet ice(1);
  for (const auto &n : ice_surface_names()) {
    ice.add(n);
  }
  const auto set_ice = [&](const char *n, double v) { ice.get(n)[0] = v; };
  // A cell 3/4 ice-covered.  The coupler's ocean fluxes carry the open
  // quarter only, plus the ice's ocean-side stress and penetrating shortwave.
  const double f = 0.75;
  set_ice("Si_ifrac", f);
  c.set("Foxx_lwup", -0.25 * 310.0);
  c.set("Foxx_lat", -0.25 * 40.0);
  c.set("Foxx_sen", -0.25 * 20.0);
  c.set("Foxx_swnet", 0.25 * 94.0 + f * 2.0);
  c.set("Foxx_taux", 0.25 * 0.1 + f * 0.05);
  c.set("Foxx_tauy", 0.25 * -0.2 + f * 0.01);
  set_ice("Fioi_swpen", 2.0);
  set_ice("Fioi_taux", 0.05);
  set_ice("Fioi_tauy", 0.01);
  set_ice("Faii_lwup", -250.0);
  set_ice("Faii_lat", -4.0);
  set_ice("Faii_sen", 8.0);
  set_ice("Faii_swnet", 30.0);
  set_ice("Faii_taux", 0.08);
  set_ice("Faii_tauy", -0.04);
  set_ice("Faxa_lwdn", 230.0);
  set_ice("Faxa_rain", 1e-6);
  set_ice("Faxa_snow", 3e-6);
  set_ice("Faxa_swvdr", 40.0);
  set_ice("Faxa_swndr", 30.0);
  set_ice("Faxa_swvdf", 20.0);
  set_ice("Faxa_swndf", 10.0);
  // The coupler's open-fraction-weighted downward fields must not be read.
  c.set("Faxa_lwdn", 0.25 * 230.0);
  c.set("Faxa_rain", 0.25 * 1e-6);
  c.set("Si_ifrac", 0.0);
  cell_mean_forcing_sample(c.in, ice, c.out);

  REQUIRE(c.get("FLUS") == Approx(0.25 * 310.0 + f * 250.0));
  REQUIRE(c.get("LHFLX") == Approx(0.25 * 40.0 + f * 4.0));
  REQUIRE(c.get("SHFLX") == Approx(0.25 * 20.0 - f * 8.0)); // into the ice
  REQUIRE(c.get("FLDS") == 230.0);
  REQUIRE(c.get("FSDS") == 100.0);
  REQUIRE(c.get("FSUS") == Approx(100.0 - (0.25 * 94.0 + f * 30.0)));
  REQUIRE(c.get("TAUX") == Approx(-(0.25 * 0.1 + f * 0.08)));
  REQUIRE(c.get("TAUY") == Approx(-(0.25 * -0.2 + f * -0.04)));
  REQUIRE(c.get("surface_precipitation_rate") == Approx(4e-6));
  REQUIRE(c.get("frozen_precipitation_rate") == Approx(3e-6));

  SECTION("under full ice the forcing is the ice surface's, not zero") {
    set_ice("Si_ifrac", 1.0);
    for (const char *n : {"Foxx_lwup", "Foxx_lat", "Foxx_sen"}) {
      c.set(n, 0.0);
    }
    c.set("Foxx_swnet", 2.0);
    cell_mean_forcing_sample(c.in, ice, c.out);
    REQUIRE(c.get("FLUS") == 250.0);
    REQUIRE(c.get("LHFLX") == 4.0);
    REQUIRE(c.get("FSUS") == Approx(70.0));
  }
  SECTION("over open water it is the coupler's") {
    set_ice("Si_ifrac", 0.0);
    c.set("Foxx_swnet", 94.0);
    c.set("Foxx_lwup", -310.0);
    cell_mean_forcing_sample(c.in, ice, c.out);
    REQUIRE(c.get("FLUS") == 310.0);
    REQUIRE(c.get("FSUS") == Approx(6.0));
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

TEST_CASE("SSH slope: centred, periodic in longitude, one-sided at the poles, "
          "zero on land", "[samudra][exports]") {
  // 4 x 3 grid; ssh rises 1 m per column and 2 m per row.
  const int nx = 4, ny = 3;
  std::vector<double> lat, ssh, mask(12, 1.0), dhdx(12), dhdy(12);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      lat.push_back(-30.0 + 30.0 * j);
      ssh.push_back(1.0 * i + 2.0 * j);
    }
  }
  mask[5] = 0.0;
  ssh_gradients(ssh, lat, mask, nx, ny, dhdx, dhdy);

  const double re = 6.37122e6, d2r = 3.14159265358979323846 / 180.0;
  const double dx_eq = 90.0 * 2.0 * d2r * re; // row 1 is the equator
  REQUIRE(dhdx[6] == Approx(2.0 / dx_eq));             // (3 - 1) / dx
  REQUIRE(dhdx[4] == Approx((3.0 - 5.0) / dx_eq));     // i=0 wraps to i=3
  REQUIRE(dhdy[6] == Approx(4.0 / (60.0 * d2r * re)));  // centred over two rows
  REQUIRE(dhdy[2] == Approx(2.0 / (30.0 * d2r * re)));  // one-sided at row 0
  REQUIRE(dhdx[5] == 0.0);                              // land
  REQUIRE(dhdy[5] == 0.0);
}

} // namespace test
} // namespace ocn
} // namespace emulator

