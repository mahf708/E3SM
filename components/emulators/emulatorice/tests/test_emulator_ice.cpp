// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "emulator_component.hpp"
#include "emulator_test_support.hpp"
#include "sea_ice_operators.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace emulator {
namespace test {

namespace {

// Slices of the coupler's lists, with fields the ice does not use.
const std::string kX2i = "So_t:So_s:Sa_z:Sa_u:Sa_v:Sa_ptem:Sa_tbot:Sa_shum:"
                         "Sa_dens:Faxa_swndr:Faxa_swvdr:Faxa_swndf:"
                         "Faxa_swvdf:Faxa_lwdn:Faxa_rain:Faxa_snow";
const std::string kI2x =
    "Si_avsdr:Si_anidr:Si_avsdf:Si_anidf:Si_tref:Si_qref:Si_t:Si_snowh:"
    "Si_u10:Si_ifrac:Faii_taux:Fioi_taux:Faii_tauy:Fioi_tauy:Faii_lat:"
    "Faii_sen:Faii_lwup:Faii_evap:Faii_swnet:Fioi_swpen:Fioi_melth:"
    "Fioi_meltw:Fioi_salt:Fioi_bcphi";

std::string ice_in_text() {
  return "spec: " + spec_path("samudrace-e3smv3-sea-ice.yaml") +
         "\ncoupler_dt: 1800\ngrid: {domain: shared, shared_from: ocn}\n";
}

int fcomm() { return MPI_Comm_c2f(MPI_COMM_WORLD); }

/// A 6 x 3 ocean domain on this rank's contiguous block, as EmulatorOcn
/// would publish it: the middle row's first two cells are land.
coupling::SharedDomain six_by_three(int rank, int size) {
  const int nx = 6, ny = 3;
  const auto decomp = grid::Decomposition::contiguous_blocks(nx * ny, size, rank);
  coupling::SharedDomain s;
  s.nx = nx;
  s.ny = ny;
  s.num_global = nx * ny;
  s.domain.global_ids = decomp.global_ids();
  for (int gid : s.domain.global_ids) {
    const int c = gid - 1;
    const double lat = -60.0 + 60.0 * (c / nx);
    const bool land = c == 6 || c == 7;
    s.domain.lat.push_back(lat);
    s.domain.lon.push_back(30.0 + 60.0 * (c % nx));
    s.domain.area.push_back(0.1);
    s.domain.mask.push_back(land ? 0.0 : 1.0);
    s.domain.frac.push_back(land ? 0.0 : 1.0);
  }
  return s;
}

} // namespace

TEST_CASE("The sea ice refuses to run without the emulated ocean's domain",
          "[ice]") {
  coupling::Exchange empty;
  const TempFile ice_in("ice_in", ice_in_text());
  EmulatorComponent ice(EmulatorType::ICE_COMP, "emulatorice", empty);
  REQUIRE_THROWS_WITH(
      ice.create_instance(fcomm(), 5, ice_in.path, "", 0, 20000101, 0),
      Catch::Contains("no domain from 'ocn'") && Catch::Contains("NTASKS"));
  REQUIRE_FALSE(ice.has_domain());
}

TEST_CASE("The sea ice refuses a domain its ranks do not cover", "[ice]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto shared = six_by_three(rank, size);
  shared.num_global = 36; // as if the ocean ran on twice as many ranks
  coupling::Exchange ex;
  coupling::publish_domain(ex, "ocn", shared);
  const TempFile ice_in("ice_in", ice_in_text());
  EmulatorComponent ice(EmulatorType::ICE_COMP, "emulatorice", ex);
  REQUIRE_THROWS_WITH(
      ice.create_instance(fcomm(), 5, ice_in.path, "", 0, 20000101, 0),
      Catch::Contains("18 of ocn's 36"));
}

TEST_CASE("The sea ice reports the ocean's domain and ice through the "
          "coupler's buffers", "[ice]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto shared = six_by_three(rank, size);
  const std::size_t n = shared.domain.size();
  coupling::Exchange ex;
  coupling::publish_domain(ex, "ocn", shared);
  // The ocean's fraction: global cell c has c / 20, land included, so the
  // domain mask is what keeps it off the land.
  std::vector<double> fraction(n);
  for (std::size_t p = 0; p < n; ++p) {
    fraction[p] = (shared.domain.global_ids[p] - 1) / 20.0;
  }
  ex.publish("ocn.sea_ice_fraction", fraction);

  ice::register_ice_operators();
  const TempFile ice_in("ice_in", ice_in_text());
  EmulatorComponent ice(EmulatorType::ICE_COMP, "emulatorice", ex);
  ice.create_instance(fcomm(), 5, ice_in.path, "", 0, 20000901, 0);
  REQUIRE(ice.get_num_global_cols() == 18);
  REQUIRE(ice.get_nx() == 6);
  REQUIRE(ice.get_ny() == 3);
  std::vector<double> lat(n), lon(n), mask(n), frac(n);
  ice.get_cols_latlon(lat.data(), lon.data());
  ice.get_cols_mask_frac(mask.data(), frac.data());
  REQUIRE(lat == shared.domain.lat);
  REQUIRE(mask == shared.domain.mask);
  REQUIRE(frac == shared.domain.frac);

  AttrVect x2i(kX2i, n), i2x(kI2x, n, -999.0);
  std::fill(x2i.data.begin(), x2i.data.end(), 0.0); // as at the driver's init
  ice.set_coupler_field_lists(kX2i, kI2x);
  ice.setup_coupling(coupling(x2i, i2x, n));
  REQUIRE(ice.export_binding()->unbound() ==
          std::vector<std::string>{"Si_u10", "Fioi_bcphi"});

  ice.initialize();
  for (std::size_t p = 0; p < n; ++p) {
    const int c = shared.domain.global_ids[p] - 1;
    INFO("cell " << c);
    const bool land = c == 6 || c == 7;
    REQUIRE(i2x.at("Si_ifrac", p) == (land ? 0.0 : c / 20.0));
    REQUIRE(i2x.at("Faii_sen", p) == 0.0); // x2i still zero: no NaN
    REQUIRE(std::isfinite(i2x.at("Si_tref", p)));
    REQUIRE(i2x.at("Si_u10", p) == 0.0);
    REQUIRE(i2x.at("Si_t", p) ==
            (land ? Approx(271.35)
                  : Approx(shared.domain.lat[p] > 0 ? 270.0 : 250.0)));
  }

  // The atmosphere arrives; the ocean publishes again.
  for (std::size_t p = 0; p < n; ++p) {
    x2i.at("Sa_z", p) = 10.0;
    x2i.at("Sa_u", p) = 5.0;
    x2i.at("Sa_v", p) = 2.0;
    x2i.at("Sa_ptem", p) = 265.3;
    x2i.at("Sa_tbot", p) = 265.0;
    x2i.at("Sa_shum", p) = 1.5e-3;
    x2i.at("Sa_dens", p) = 1.33;
    x2i.at("Faxa_swvdr", p) = 50.0;
  }
  std::vector<double> less(n);
  std::transform(fraction.begin(), fraction.end(), less.begin(),
                 [](double f) { return f / 2.0; });
  ex.publish("ocn.sea_ice_fraction", less);
  ice.run(1800, {20000901, 1800});

  std::size_t domain_cells = 0;
  for (std::size_t p = 0; p < n; ++p) {
    const int c = shared.domain.global_ids[p] - 1;
    INFO("cell " << c);
    const bool land = c == 6 || c == 7;
    REQUIRE(i2x.at("Si_ifrac", p) == (land ? 0.0 : c / 40.0));
    if (land) {
      REQUIRE(i2x.at("Faii_sen", p) == 0.0);
      REQUIRE(i2x.at("Faii_swnet", p) == 0.0);
      continue;
    }
    ++domain_cells;
    REQUIRE(i2x.at("Faii_lwup", p) < 0.0);
    REQUIRE(i2x.at("Faii_taux", p) > 0.0);
    REQUIRE(i2x.at("Fioi_taux", p) == i2x.at("Faii_taux", p));
    REQUIRE(i2x.at("Fioi_melth", p) == 0.0);
    REQUIRE(i2x.at("Faii_swnet", p) ==
            Approx(50.0 * (1.0 - ice::albedo::vsdr)));
  }
  REQUIRE_THROWS_WITH(ice.run(1800), Catch::Contains("emulator_run_at"));
  ice.finalize();
}

TEST_CASE("Under a model atmosphere the ice skin balances its energy",
          "[ice]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto shared = six_by_three(rank, size);
  const std::size_t n = shared.domain.size();
  coupling::Exchange ex;
  coupling::publish_domain(ex, "ocn", shared);
  ex.publish("ocn.sea_ice_fraction", std::vector<double>(n, 0.5));

  ice::register_ice_operators();
  const TempFile ice_in(
      "ice_in", "spec: " + spec_path("samudra-e3smv3-sea-ice-energy-balance.yaml") +
                    "\ncoupler_dt: 1800\ngrid: {domain: shared, shared_from: ocn}\n");
  EmulatorComponent ice(EmulatorType::ICE_COMP, "emulatorice", ex);
  ice.create_instance(fcomm(), 5, ice_in.path, "", 0, 20000115, 0);
  AttrVect x2i(kX2i, n), i2x(kI2x, n, -999.0);
  std::fill(x2i.data.begin(), x2i.data.end(), 0.0);
  ice.set_coupler_field_lists(kX2i, kI2x);
  ice.setup_coupling(coupling(x2i, i2x, n));
  ice.initialize(); // x2i all zero: no atmosphere, so the prescribed skin
  for (std::size_t p = 0; p < n; ++p) {
    REQUIRE(std::isfinite(i2x.at("Si_t", p)));
  }

  const ice::AtmosphereAtIce arctic{10.0, 5.0, 2.0, 265.3, 1.5e-3, 1.33, 265.0};
  for (std::size_t p = 0; p < n; ++p) {
    x2i.at("Sa_z", p) = arctic.z;
    x2i.at("Sa_u", p) = arctic.u;
    x2i.at("Sa_v", p) = arctic.v;
    x2i.at("Sa_ptem", p) = arctic.ptem;
    x2i.at("Sa_tbot", p) = arctic.tbot;
    x2i.at("Sa_shum", p) = arctic.shum;
    x2i.at("Sa_dens", p) = arctic.dens;
    x2i.at("Faxa_lwdn", p) = 200.0;
  }
  ice.run(1800, {20000115, 1800});
  ice::SkinOptions skin;
  skin.mode = ice::SkinOptions::Mode::EnergyBalance;
  for (std::size_t p = 0; p < n; ++p) {
    const int c = shared.domain.global_ids[p] - 1;
    INFO("cell " << c);
    if (c == 6 || c == 7) {
      REQUIRE(i2x.at("Si_t", p) == Approx(271.35));
      continue;
    }
    const double want = ice::balanced_skin_temperature(
        arctic, 0.0, 200.0, shared.domain.lat[p], skin);
    REQUIRE(i2x.at("Si_t", p) == want);
    // Not dice's January skin, and far less heat drawn from the air.
    REQUIRE(std::abs(i2x.at("Si_t", p) -
                     ice::prescribed_skin_temperature(shared.domain.lat[p],
                                                      20000115, 1800)) > 1.0);
  }
  // What a coupler-forced ocean completes its cell means with.
  for (const char *name : {"Faii_lwup", "Faii_lat", "Si_ifrac"}) {
    const auto published = ex.get(std::string("ice.") + name);
    for (std::size_t p = 0; p < n; ++p) {
      REQUIRE(published[p] == i2x.at(name, p));
    }
  }
  REQUIRE(ex.get("ice.Faxa_lwdn")[0] == 200.0);
  ice.finalize();
}

} // namespace test
} // namespace emulator

EMULATOR_TEST_MPI_MAIN
