#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "sea_ice_surface.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace emulator {
namespace test {

using ice::AtmIceFluxes;
using ice::AtmosphereAtIce;

namespace {

// Six cells through EICE's own eice_flux_atmice (eice_flux_atmice_mod.F90 on
// mahf708/eocn/add-samudra), compiled -O0 with gfortran 14 against
// share/util/shr_const_mod.F90.  Cell 4 was masked there and is not repeated.
struct Case {
  const char *what;
  AtmosphereAtIce atm;
  double ts;
  AtmIceFluxes want;
};

const std::vector<Case> kFortran = {
    {"stable: warm air over cold ice",
     {10.0, 5.0, 2.0, 265.3, 1.5e-3, 1.33, 265.0},
     250.0,
     {2.0655199184313588E+001, 3.3012078182785944E+000,
      -2.2149900074218749E+002, 1.1645704371815692E-006,
      6.7154821767603937E-003, 2.6861928707041572E-003,
      2.5624215157852421E+002, 1.1391708492200234E-003}},
    {"unstable: cold air over warmer ice",
     {25.0, -8.0, 3.0, 240.5, 2.0e-4, 1.45, 240.0},
     255.0,
     {-3.7267590814846278E+002, -4.8306265597750873E+001,
      -2.3975764181120761E+002, -1.7041050410184807E-005,
      -1.8362519421340578E-001, 6.8859447830027162E-002,
      2.4220776042905484E+002, 7.6846067358788718E-004}},
    {"calm: wind below the 1 m/s floor",
     {5.0, 0.3, -0.2, 250.1, 5.0e-4, 1.40, 250.0},
     250.0,
     {2.2326784329951613E-001, -6.3769999007127010E-001,
      -2.2149900074218749E+002, -2.2496207361317604E-007,
      6.6638879182912404E-004, -4.4425919455274942E-004,
      2.5000000000000000E+002, 5.9119674712239552E-004}},
    {"strong wind, high level",
     {60.0, 15.0, -12.0, 256.0, 8.0e-4, 1.37, 255.0},
     262.0,
     {-2.2324678607389279E+002, -8.0341463097742675E+001,
      -2.6718797094974735E+002, -2.8342139590694846E-005,
      5.2637324437048716E-001, -4.2109859549638973E-001,
      2.5761038398133616E+002, 1.2804880158351259E-003}},
    {"no temperature difference",
     {10.0, 3.0, 4.0, 258.1, 1.0e-3, 1.36, 258.0},
     258.1,
     {0.0, -5.5361736389383607E+000, -2.5163077477003148E+002,
      -1.9530016012058984E-006, 3.3389815862620506E-002,
      4.4519754483494010E-002, 2.5801604159752674E+002,
      1.1470876891112316E-003}},
};

void require_close(double got, double want) {
  // Relative 1e-12: the same arithmetic up to the compiler's contraction.
  REQUIRE(got == Approx(want).epsilon(1e-12).margin(1e-15));
}

fields::FieldSet imports_for(const std::vector<AtmosphereAtIce> &cells) {
  fields::FieldSet f(cells.size());
  for (const auto &name : ice::sea_ice_import_names()) {
    f.add(name);
  }
  for (std::size_t i = 0; i < cells.size(); ++i) {
    f.get("Sa_z")[i] = cells[i].z;
    f.get("Sa_u")[i] = cells[i].u;
    f.get("Sa_v")[i] = cells[i].v;
    f.get("Sa_ptem")[i] = cells[i].ptem;
    f.get("Sa_shum")[i] = cells[i].shum;
    f.get("Sa_dens")[i] = cells[i].dens;
    f.get("Sa_tbot")[i] = cells[i].tbot;
  }
  return f;
}

fields::FieldSet all_exports(std::size_t n) {
  fields::FieldSet f(n);
  for (const auto &name : ice::sea_ice_export_names()) {
    f.add(name, -999.0);
  }
  return f;
}

const AtmosphereAtIce kArctic{10.0, 5.0, 2.0, 265.3, 1.5e-3, 1.33, 265.0};

} // namespace

TEST_CASE("The bulk fluxes are EICE's, on the same inputs", "[ice]") {
  for (const auto &c : kFortran) {
    INFO(c.what);
    REQUIRE(ice::bulk_fluxes_defined(c.atm));
    const auto f = ice::atm_ice_fluxes(c.atm, c.ts);
    require_close(f.sen, c.want.sen);
    require_close(f.lat, c.want.lat);
    require_close(f.lwup, c.want.lwup);
    require_close(f.evap, c.want.evap);
    require_close(f.taux, c.want.taux);
    require_close(f.tauy, c.want.tauy);
    require_close(f.tref, c.want.tref);
    require_close(f.qref, c.want.qref);
  }
}

TEST_CASE("The bulk fluxes have the coupler's signs", "[ice]") {
  // Warm air over cold ice heats the ice: positive downward.
  const auto stable = ice::atm_ice_fluxes(kFortran[0].atm, kFortran[0].ts);
  REQUIRE(stable.sen > 0.0);
  REQUIRE(stable.lwup < 0.0);
  REQUIRE(stable.evap == Approx(stable.lat / (2.501e6 + 3.337e5)));
  // The 2 m temperature lies between the surface and the lowest level.
  REQUIRE(stable.tref > kFortran[0].ts);
  REQUIRE(stable.tref < kFortran[0].atm.tbot);
  // Stress follows the wind.
  REQUIRE(stable.taux / stable.tauy == Approx(5.0 / 2.0));
}

TEST_CASE("The coupler's zeros at initialization give zero fluxes, not NaN",
          "[ice]") {
  REQUIRE_FALSE(ice::bulk_fluxes_defined(AtmosphereAtIce{}));
  AtmosphereAtIce no_density = kArctic;
  no_density.dens = 0.0;
  REQUIRE_FALSE(ice::bulk_fluxes_defined(no_density));
  AtmosphereAtIce nan_height = kArctic;
  nan_height.z = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_FALSE(ice::bulk_fluxes_defined(nan_height));

  const std::vector<double> lat{80.0, -70.0}, mask{1.0, 1.0}, frac{0.9, 0.4};
  const auto imports = imports_for({AtmosphereAtIce{}, AtmosphereAtIce{}});
  auto exports = all_exports(2);
  const auto counts = ice::compute_sea_ice_exports(
      {20000101, 0}, {lat, mask, frac}, imports, exports);
  REQUIRE(counts.domain == 2);
  REQUIRE(counts.fluxes == 0);
  REQUIRE(counts.no_atmosphere == 2);
  for (const char *name : {"Faii_sen", "Faii_lat", "Faii_lwup", "Faii_evap",
                           "Faii_taux", "Faii_tauy", "Si_qref", "Fioi_taux"}) {
    INFO(name);
    REQUIRE(exports.get(name)[0] == 0.0);
    REQUIRE(exports.get(name)[1] == 0.0);
  }
  // Not spval: the reference temperature is the skin.
  REQUIRE(exports.get("Si_tref")[0] == exports.get("Si_t")[0]);
  REQUIRE(exports.get("Si_ifrac")[0] == 0.9);
}

TEST_CASE("The skin temperature is not blended by the ice fraction", "[ice]") {
  // Blending here as well as in the coupler's merge would scale the anomaly
  // by ifrac squared.
  const std::vector<double> lat{75.0, 75.0}, mask{1.0, 1.0},
                            frac{0.01, 0.95};
  const auto imports = imports_for({kArctic, kArctic});
  auto exports = all_exports(2);
  ice::compute_sea_ice_exports({20000115, 0}, {lat, mask, frac}, imports,
                               exports);
  const auto t = exports.get("Si_t");
  REQUIRE(t[0] == t[1]);
  REQUIRE(t[0] == Approx(253.036).margin(0.001)); // 15 January, north
  REQUIRE(exports.get("Faii_sen")[0] == exports.get("Faii_sen")[1]);
  REQUIRE(exports.get("Si_snowh")[0] == Approx(0.20 * 0.01));
}

TEST_CASE("The skin temperature follows dice's season in each hemisphere",
          "[ice]") {
  // Warmest in the north on 1 September, coldest half a year on.
  REQUIRE(ice::prescribed_skin_temperature(80.0, 20000901, 0) == 270.0);
  REQUIRE(ice::prescribed_skin_temperature(-80.0, 20000901, 0) == 250.0);
  REQUIRE(ice::prescribed_skin_temperature(80.0, 20010302, 43200) ==
          Approx(250.0).margin(0.01));
  REQUIRE(ice::prescribed_skin_temperature(-80.0, 20010302, 43200) ==
          Approx(270.0).margin(0.01));
  REQUIRE_THROWS_AS(ice::prescribed_skin_temperature(80.0, 20000229, 0),
                    std::invalid_argument);
}

TEST_CASE("Outside the domain the ice reports nothing but the freezing point",
          "[ice]") {
  const std::vector<double> lat{10.0, 70.0}, mask{0.0, 1.0}, frac{0.7, 1.4};
  auto imports = imports_for({kArctic, kArctic});
  for (const char *band : {"Faxa_swvdr", "Faxa_swndr", "Faxa_swvdf",
                           "Faxa_swndf"}) {
    imports.get(band)[0] = imports.get(band)[1] = 100.0;
  }
  auto exports = all_exports(2);
  const auto counts = ice::compute_sea_ice_exports(
      {20000601, 3600}, {lat, mask, frac}, imports, exports);
  REQUIRE(counts.domain == 1);
  REQUIRE(counts.with_ice == 1);
  REQUIRE(counts.fluxes == 1);
  for (const auto &name : ice::sea_ice_export_names()) {
    INFO(name);
    if (name == "Si_t" || name == "Si_tref") {
      REQUIRE(exports.get(name)[0] == Approx(271.35));
    } else {
      REQUIRE(exports.get(name)[0] == 0.0);
    }
  }
  // Inside: the fraction clamped, and net shortwave from dice's albedos.
  REQUIRE(exports.get("Si_ifrac")[1] == 1.0);
  const double want = 100.0 * ((1 - ice::albedo::vsdr) + (1 - ice::albedo::nidr) +
                               (1 - ice::albedo::vsdf) + (1 - ice::albedo::nidf));
  REQUIRE(exports.get("Faii_swnet")[1] == Approx(want));
  REQUIRE(want == Approx(100.0 * (4 - 0.5858 - 0.77436 - 0.5572 - 0.7715)));
}

TEST_CASE("The ocean feels the atmosphere's stress and none of the melt",
          "[ice]") {
  const std::vector<double> lat{70.0}, mask{1.0}, frac{0.5};
  const auto imports = imports_for({kArctic});
  auto exports = all_exports(1);
  ice::compute_sea_ice_exports({20000301, 0}, {lat, mask, frac}, imports,
                               exports);
  REQUIRE(exports.get("Faii_taux")[0] != 0.0);
  REQUIRE(exports.get("Fioi_taux")[0] == exports.get("Faii_taux")[0]);
  REQUIRE(exports.get("Fioi_tauy")[0] == exports.get("Faii_tauy")[0]);
  for (const char *name : {"Fioi_melth", "Fioi_meltw", "Fioi_salt",
                           "Fioi_swpen"}) {
    INFO(name);
    REQUIRE(exports.get(name)[0] == 0.0);
  }
}

TEST_CASE("A field the coupler does not carry is simply not written",
          "[ice]") {
  const std::vector<double> lat{70.0}, mask{1.0}, frac{0.5};
  const auto imports = imports_for({kArctic});
  fields::FieldSet exports(1);
  for (const auto &name : ice::sea_ice_export_names()) {
    if (name != "Si_snowh") {
      exports.add(name);
    }
  }
  REQUIRE_NOTHROW(ice::compute_sea_ice_exports({20000301, 0},
                                               {lat, mask, frac}, imports,
                                               exports));
  REQUIRE(exports.get("Si_ifrac")[0] == 0.5);
}

TEST_CASE("A non-finite export stops the run and names the field", "[ice]") {
  const std::vector<double> lat{70.0}, mask{1.0}, frac{0.5};
  auto imports = imports_for({kArctic});
  imports.get("Sa_u")[0] = std::numeric_limits<double>::quiet_NaN();
  auto exports = all_exports(1);
  REQUIRE_THROWS_WITH(
      ice::compute_sea_ice_exports({20000301, 0}, {lat, mask, frac}, imports,
                                   exports),
      Catch::Contains("Faii_") && Catch::Contains("non-finite"));
}

TEST_CASE("The balanced skin closes the surface energy budget", "[ice]") {
  const ice::SkinOptions o{ice::SkinOptions::Mode::EnergyBalance};
  // Arctic winter night: 265 K air, 200 W/m2 down, no sun.
  const double ts = ice::balanced_skin_temperature(kArctic, 0.0, 200.0, 80.0, o);
  const auto f = ice::atm_ice_fluxes(kArctic, ts);
  const double conductance = 1.0 / (2.0 / 2.03 + 0.2 / 0.31);
  const double residual = 200.0 + f.lwup + f.sen + f.lat +
                          conductance * (ice::constants::tkfrzsw - ts);
  REQUIRE(std::abs(residual) < 1e-2);
  // Colder than the air it radiates to space under, warmer than the
  // prescribed 253 K of mid-January: a few W/m2 of sensible heat, not 150.
  REQUIRE(ts < kArctic.tbot);
  REQUIRE(ts > 240.0);
  REQUIRE(std::abs(f.sen) < 60.0);
}

TEST_CASE("The balanced skin warms with the longwave and stops at melting",
          "[ice]") {
  const ice::SkinOptions o{ice::SkinOptions::Mode::EnergyBalance};
  const double cold = ice::balanced_skin_temperature(kArctic, 0.0, 150.0, 80.0, o);
  const double warm = ice::balanced_skin_temperature(kArctic, 0.0, 250.0, 80.0, o);
  REQUIRE(warm > cold);
  // Thinner southern ice conducts more of the ocean's heat to the skin.
  REQUIRE(ice::balanced_skin_temperature(kArctic, 0.0, 150.0, -70.0, o) > cold);
  // Summer: warm air and sun melt the surface.
  const AtmosphereAtIce summer{10.0, 4.0, 1.0, 276.0, 4.0e-3, 1.27, 275.5};
  REQUIRE(ice::balanced_skin_temperature(summer, 150.0, 300.0, 80.0, o) ==
          ice::tmelt);
}

TEST_CASE("The energy-balance skin is used only where there is ice",
          "[ice]") {
  const std::vector<double> lat{75.0, 75.0}, mask{1.0, 1.0}, frac{0.0, 0.8};
  auto imports = imports_for({kArctic, kArctic});
  imports.add("Faxa_lwdn");
  imports.get("Faxa_lwdn")[0] = imports.get("Faxa_lwdn")[1] = 200.0;
  auto exports = all_exports(2);
  ice::SkinOptions o;
  o.mode = ice::SkinOptions::Mode::EnergyBalance;
  ice::compute_sea_ice_exports({20000115, 0}, {lat, mask, frac}, imports,
                               exports, o);
  const auto t = exports.get("Si_t");
  REQUIRE(t[0] == Approx(253.036).margin(0.001)); // no ice: prescribed
  REQUIRE(t[1] == ice::balanced_skin_temperature(kArctic, 0.0, 200.0, 75.0, o));
  REQUIRE(exports.get("Faii_lwup")[1] ==
          Approx(-ice::constants::stebol * std::pow(t[1], 4)));
  // Without the longwave it cannot balance, and says so.
  const auto no_lwdn = imports_for({kArctic, kArctic});
  REQUIRE_THROWS_WITH(ice::compute_sea_ice_exports({20000115, 0},
                                                   {lat, mask, frac}, no_lwdn,
                                                   exports, o),
                      Catch::Contains("Faxa_lwdn"));
}

} // namespace test
} // namespace emulator
