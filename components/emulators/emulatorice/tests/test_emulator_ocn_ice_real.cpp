// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "emulator_component.hpp"
#include "emulator_test_support.hpp"
#include "ocean_operators.hpp"
#include "sea_ice_operators.hpp"
#include "scrip_reader.hpp"

#include <mpi.h>
#ifdef EMULATOR_ENABLE_LIBTORCH
#include <torch/cuda.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace emulator {
namespace test {

namespace {

const std::string kGrid = "/global/cfs/cdirs/e3sm/inputdata/share/meshes/"
                          "gaussian_180x360_latlon.scrip.20260127.nc";
const std::string kRoot = "/pscratch/sd/m/mahf708/SamudrACE-E3SMv3/";
const std::string kModel = kRoot + "eocn/samudra_ocn_traced_masked_cuda.pt";
const std::string kIc = kRoot + "eocn/samudra_ocn_ic_0_icemask.nc";

const std::string kX2o = "Foxx_taux:Foxx_tauy:Faxa_rain:Faxa_snow:Foxx_lwup:"
                         "Faxa_lwdn:Foxx_swnet:Foxx_lat:Foxx_sen:Si_ifrac:"
                         "Sa_pslv:Foxx_evap";
const std::string kO2x = "So_t:So_s:So_u:So_v:So_dhdx:So_dhdy:So_ssh:Fioo_q:"
                         "So_bldepth";
const std::string kX2i = "Sa_z:Sa_u:Sa_v:Sa_ptem:Sa_tbot:Sa_shum:Sa_dens:"
                         "Faxa_swndr:Faxa_swvdr:Faxa_swndf:Faxa_swvdf:So_t";
const std::string kI2x =
    "Si_avsdr:Si_anidr:Si_avsdf:Si_anidf:Si_tref:Si_qref:Si_t:Si_snowh:"
    "Si_ifrac:Faii_taux:Fioi_taux:Faii_tauy:Fioi_tauy:Faii_lat:Faii_sen:"
    "Faii_lwup:Faii_evap:Faii_swnet:Fioi_swpen:Fioi_melth:Fioi_meltw:"
    "Fioi_salt";

coupling::ModelTime after(int n) {
  const int s = n * 1800;
  return {20000101 + s / 86400, s % 86400};
}

} // namespace

TEST_CASE("The emulated ocean and its sea ice run through the coupler's "
          "buffers, the ice on the ocean's own domain", "[ocn][ice][real]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ok = 1;
#ifndef EMULATOR_ENABLE_LIBTORCH
  ok = 0;
#else
  ok = grid::have_scrip_reader() && std::filesystem::exists(kModel) &&
       std::filesystem::exists(kIc) && torch::cuda::is_available();
#endif
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ok) {
    WARN("skipped: needs libtorch, netCDF, a GPU and the Samudra files");
    return;
  }

  const std::string ocn_in =
      (std::filesystem::temp_directory_path() /
       ("ocn_in_" + std::to_string(::getpid()) + "_" + std::to_string(rank)))
          .string();
  std::ofstream(ocn_in)
      << "spec: " << spec_path("samudra-e3smv3-ocean-coupler-forced.yaml")
      << "\ncoupler_dt: 1800\n"
      << "grid: {file: " << kGrid
      << ", domain: ocean_mask, mask_variable: mask_2d, publish_as: ocn}\n"
      << "initial_condition: " << kIc << "\n"
      << "inference: {backend: libtorch, model_path: " << kModel
      << ", device: cuda}\n";
  const TempFile ice_in(
      "ice_in", "spec: " + spec_path("samudrace-e3smv3-sea-ice.yaml") +
                    "\ncoupler_dt: 1800\ngrid: {domain: shared, shared_from: ocn}\n");
  ocn::register_ocn_operators();
  ice::register_ice_operators();
  coupling::Exchange exchange;

  // The driver's order: the ocean is created and initialized, then the ice.
  EmulatorComponent ocn(EmulatorType::OCN_COMP, "emulatorocn", exchange);
  ocn.create_instance(MPI_Comm_c2f(MPI_COMM_WORLD), 4, ocn_in, "", 0,
                      20000101, 0);
  std::remove(ocn_in.c_str());
  const auto n = static_cast<std::size_t>(ocn.get_num_local_cols());
  std::vector<double> mask(n), frac(n), area(n);
  ocn.get_cols_mask_frac(mask.data(), frac.data());
  ocn.get_cols_area(area.data());
  double ocean_cells = 0.0, ocean_area = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    ocean_cells += mask[p];
    ocean_area += mask[p] * area[p];
  }
  REQUIRE(global_sum(ocean_cells) == 44892.0);
  ocean_area = global_sum(ocean_area);

  AttrVect x2o(kX2o, n), o2x(kO2x, n), x2i(kX2i, n), i2x(kI2x, n);
  ocn.set_coupler_field_lists(kX2o, kO2x);
  ocn.setup_coupling(coupling(x2o, o2x, n));
  ocn.initialize();

  EmulatorComponent ice(EmulatorType::ICE_COMP, "emulatorice", exchange);
  ice.create_instance(MPI_Comm_c2f(MPI_COMM_WORLD), 5, ice_in.path, "", 0, 20000101, 0);
  REQUIRE(ice.get_num_local_cols() == static_cast<int>(n));
  std::vector<double> ice_mask(n), ice_frac(n);
  ice.get_cols_mask_frac(ice_mask.data(), ice_frac.data());
  REQUIRE(ice_mask == mask);
  ice.set_coupler_field_lists(kX2i, kI2x);
  ice.setup_coupling(coupling(x2i, i2x, n));
  ice.initialize();

  const auto report = [&](const char *when) {
    double sst = 0.0, ice_area = 0.0, ice_cells = 0.0, off_mask = 0.0;
    const auto own = ocn.model()->statics().get("mask_ocean_sea_ice_fraction");
    for (std::size_t p = 0; p < n; ++p) {
      sst += mask[p] * area[p] * o2x.at("So_t", p);
      ice_area += area[p] * i2x.at("Si_ifrac", p);
      ice_cells += i2x.at("Si_ifrac", p) > 0.0;
      off_mask += own[p] == 0.0 && i2x.at("Si_ifrac", p) != 0.0;
    }
    sst = global_sum(sst) / ocean_area;
    const double fraction = global_sum(ice_area) / ocean_area;
    ice_cells = global_sum(ice_cells);
    off_mask = global_sum(off_mask);
    if (rank == 0) {
      std::printf("  %s: So_t %.2f K and ice fraction %.4f over the ocean "
                  "(area-weighted); %.0f cells with ice (%d ranks)\n",
                  when, sst, fraction, ice_cells, size);
    }
    REQUIRE(sst > 285.0);
    REQUIRE(sst < 295.0);
    REQUIRE(fraction > 0.01);
    REQUIRE(fraction < 0.2);
    REQUIRE(ice_cells > 0.0);
    REQUIRE(ice_cells <= 25923.0); // the ice channel's own mask
    REQUIRE(off_mask == 0.0);
  };
  report("init");

  // A plausible open-ocean forcing.  It cannot move the state within a day:
  // the forcing enters at the first 5-day window close.
  for (std::size_t p = 0; p < n; ++p) {
    x2o.at("Foxx_taux", p) = 0.05;
    x2o.at("Faxa_rain", p) = 3.0e-5;
    x2o.at("Foxx_lwup", p) = -400.0;
    x2o.at("Faxa_lwdn", p) = 340.0;
    x2o.at("Foxx_swnet", p) = 170.0;
    x2o.at("Foxx_lat", p) = -100.0;
    x2o.at("Foxx_sen", p) = -15.0;
    x2i.at("Sa_z", p) = 10.0;
    x2i.at("Sa_u", p) = 5.0;
    x2i.at("Sa_v", p) = 2.0;
    x2i.at("Sa_ptem", p) = 265.3;
    x2i.at("Sa_tbot", p) = 265.0;
    x2i.at("Sa_shum", p) = 1.5e-3;
    x2i.at("Sa_dens", p) = 1.33;
  }
  std::vector<double> previous(ocn.model()->aux().get("sea_ice_fraction").begin(),
                               ocn.model()->aux().get("sea_ice_fraction").end());
  for (int step = 1; step <= 48; ++step) {
    // The driver runs ice before ocn: the ice reports the fraction the
    // ocean exported one step earlier, exactly.
    ice.run(1800, after(step));
    for (std::size_t p = 0; p < n; ++p) {
      REQUIRE(i2x.at("Si_ifrac", p) == previous[p]);
    }
    ocn.run(1800, after(step));
    const auto now = ocn.model()->aux().get("sea_ice_fraction");
    previous.assign(now.begin(), now.end());
  }
  report("one day");
  double fluxes = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    fluxes += i2x.at("Faii_lwup", p) < 0.0 ? 1.0 : 0.0;
  }
  REQUIRE(global_sum(fluxes) == 44892.0);
  ocn.finalize();
  ice.finalize();
}

} // namespace test
} // namespace emulator

EMULATOR_TEST_MPI_MAIN
