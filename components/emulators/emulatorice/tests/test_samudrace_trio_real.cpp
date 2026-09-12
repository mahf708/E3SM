// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "atm.hpp"
#include "grid_field_reader.hpp"
#include "ice.hpp"
#include "ocn.hpp"
#include "samudra_ocean.hpp"
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
const std::string kAtmModel = kRoot + "eatm/samudrace_atm_traced_cuda.pt";
const std::string kAtmIc = kRoot + "eatm/samudrace_atm_ic_0.nc";
const std::string kOcnModel = kRoot + "eocn/samudra_ocn_traced_masked_cuda.pt";
const std::string kOcnIc = kRoot + "eocn/samudra_ocn_ic_0_icemask.nc";

const std::string kX2a = "Sf_lfrac:Sf_ifrac:Sf_ofrac:Sx_t:So_t:Sx_avsdr";
const std::string kA2x =
    "Sa_z:Sa_topo:Sa_u:Sa_v:Sa_tbot:Sa_ptem:Sa_shum:Sa_pbot:Sa_dens:Sa_pslv:"
    "Faxa_rainc:Faxa_rainl:Faxa_snowc:Faxa_snowl:Faxa_lwdn:Faxa_swndr:"
    "Faxa_swvdr:Faxa_swndf:Faxa_swvdf:Faxa_swnet";
const std::string kO2x = "So_t:So_s:So_u:So_v:So_dhdx:So_dhdy:So_ssh";
const std::string kX2o = "Foxx_taux:Sa_pslv"; // atmosphere forcing: unused
const std::string kX2i = "Sa_z:Sa_u:Sa_v:Sa_ptem:Sa_tbot:Sa_shum:Sa_dens:"
                         "Faxa_swndr:Faxa_swvdr:Faxa_swndf:Faxa_swvdf";
const std::string kI2x =
    "Si_avsdr:Si_anidr:Si_avsdf:Si_anidf:Si_tref:Si_qref:Si_t:Si_ifrac:"
    "Faii_taux:Fioi_taux:Faii_tauy:Fioi_tauy:Faii_lat:Faii_sen:Faii_lwup:"
    "Faii_evap:Faii_swnet:Fioi_swpen:Fioi_melth:Fioi_meltw:Fioi_salt";

struct AttrVect {
  std::vector<std::string> names;
  std::vector<double> data;
  AttrVect(const std::string &list, std::size_t np) {
    std::size_t start = 0;
    while (true) {
      const auto colon = list.find(':', start);
      names.push_back(list.substr(start, colon - start));
      if (colon == std::string::npos) {
        break;
      }
      start = colon + 1;
    }
    data.assign(names.size() * np, 0.0);
  }
  double &at(const std::string &n, std::size_t p) {
    const auto row = static_cast<std::size_t>(
        std::find(names.begin(), names.end(), n) - names.begin());
    return data[p * names.size() + row];
  }
};

EmulatorCouplingDesc desc(AttrVect &in, AttrVect &out, std::size_t n) {
  return {in.data.data(), out.data.data(), static_cast<int>(in.names.size()),
          static_cast<int>(out.names.size()), static_cast<int>(n)};
}

std::string write_input(const std::string &stem, const std::string &text) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto path = (std::filesystem::temp_directory_path() /
                     (stem + "_" + std::to_string(::getpid()) + "_" +
                      std::to_string(rank)))
                        .string();
  std::ofstream(path) << text;
  return path;
}

double global_sum(double local) {
  double total = 0.0;
  MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return total;
}

coupling::ModelTime after(int n) {
  const int s = n * 1800;
  return {20000101 + s / 86400, s % 86400};
}

} // namespace

TEST_CASE("SamudrACE's atmosphere, ocean and sea ice run as three components "
          "configured from their input files", "[samudrace][real]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ok = 1;
#ifndef EMULATOR_ENABLE_LIBTORCH
  ok = 0;
#else
  ok = grid::have_scrip_reader() && std::filesystem::exists(kAtmModel) &&
       std::filesystem::exists(kOcnModel) && torch::cuda::is_available();
#endif
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ok) {
    WARN("skipped: needs libtorch, netCDF, a GPU and the SamudrACE files");
    return;
  }
  const int fcomm = MPI_Comm_c2f(MPI_COMM_WORLD);
  coupling::Exchange exchange;

  // The driver's order: atm, then ocn, then ice.
  const auto atm_in = write_input(
      "atm_in", "grid_file: " + kGrid + "\nemulator: SamudrACE-E3SMv3\n" +
                    "model_path: " + kAtmModel + "\nic_file: " + kAtmIc +
                    "\nseed: 2026\nsurface_layer: near_surface\n" +
                    "publish_ocean_forcing: true\nsurface_from_ocean: true\n");
  const auto ocn_in = write_input(
      "ocn_in", "grid_file: " + kGrid + "\nic_file: " + kOcnIc +
                    "\nemulator: Samudra-E3SMv3\nmodel_path: " + kOcnModel +
                    "\nforcing: atmosphere\n");
  EmulatorAtm atm(exchange);
  atm.create_instance(fcomm, 1, atm_in, "", 0, 20000101, 0);
  const auto n = static_cast<std::size_t>(atm.get_num_local_cols());
  AttrVect x2a(kX2a, n), a2x(kA2x, n), x2o(kX2o, n), o2x(kO2x, n),
           x2i(kX2i, n), i2x(kI2x, n);
  // The coupler's fractions, as the model-level SamudrACE test sets them:
  // the initial condition's, made a partition.
  {
    const auto ic = grid::read_grid_fields(
        kAtmIc, {"LANDFRAC", "OCNFRAC", "ICEFRAC", "TS"}, 180, 360);
    std::vector<int> gids(n);
    atm.get_local_col_gids(gids.data());
    for (std::size_t p = 0; p < n; ++p) {
      const auto c = static_cast<std::size_t>(gids[p] - 1);
      auto get = [&](int k) {
        const double v = ic[k].values[c];
        return std::isfinite(v) ? std::clamp(v, 0.0, 1.0e30) : 0.0;
      };
      double l = std::min(get(0), 1.0), o = std::min(get(1), 1.0),
             i = std::min(get(2), 1.0);
      const double sum = l + o + i;
      if (sum > 1.0) {
        l /= sum;
        o /= sum;
        i /= sum;
      }
      x2a.at("Sf_lfrac", p) = l;
      x2a.at("Sf_ofrac", p) = o;
      x2a.at("Sf_ifrac", p) = i;
      x2a.at("Sx_t", p) = (l + o + i) * get(3);
    }
  }
  atm.set_coupler_field_lists(kX2a, kA2x);
  atm.setup_coupling(desc(x2a, a2x, n));
  atm.initialize();

  EmulatorOcn ocn(exchange);
  ocn.create_instance(fcomm, 4, ocn_in, "", 0, 20000101, 0);
  ocn.set_coupler_field_lists(kX2o, kO2x);
  ocn.setup_coupling(desc(x2o, o2x, n));
  ocn.initialize();

  EmulatorIce ice(exchange);
  ice.create_instance(fcomm, 5, "", "", 0, 20000101, 0);
  ice.set_coupler_field_lists(kX2i, kI2x);
  ice.setup_coupling(desc(x2i, i2x, n));
  ice.initialize();
  std::remove(atm_in.c_str());
  std::remove(ocn_in.c_str());

  std::vector<double> area(n), mask(n), frac(n);
  ocn.get_cols_area(area.data());
  ocn.get_cols_mask_frac(mask.data(), frac.data());
  double ocean_area = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    ocean_area += area[p] * mask[p];
  }
  ocean_area = global_sum(ocean_area);

  for (int step = 1; step <= 240; ++step) { // five days: one ocean step
    // The same grid for all three: the coupler's a2x -> x2i is a copy.
    for (std::size_t p = 0; p < n; ++p) {
      for (const auto &name : x2i.names) {
        x2i.at(name, p) = a2x.at(name, p);
      }
    }
    ice.run(1800, after(step));
    ocn.run(1800, after(step));
    atm.run(1800, after(step));
  }

  const auto *ocean = ocn.model();
  double sst = 0.0, ifrac = 0.0, sen_ice = 0.0, ice_area = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    sst += area[p] * mask[p] * ocean->brackets().upper("sst")[p];
    ifrac += area[p] * i2x.at("Si_ifrac", p);
    sen_ice += area[p] * i2x.at("Si_ifrac", p) * i2x.at("Faii_sen", p);
    ice_area += area[p] * i2x.at("Si_ifrac", p);
  }
  sst = global_sum(sst) / ocean_area;
  sen_ice = global_sum(sen_ice) / global_sum(ice_area);
  ifrac = global_sum(ifrac) / ocean_area;
  const double fluxes =
      global_sum(static_cast<double>(ice.last_counts().fluxes));
  if (rank == 0) {
    std::printf("  day 5 through three components: ocean SST %.6f K "
                "(predicted, area-weighted); ice fraction %.4f; sensible "
                "heat into the ice %.1f W/m2 (ice-weighted); %d ranks\n",
                sst, ifrac, sen_ice, size);
  }
  REQUIRE(ocean->clock().completed_steps() == 1);
  REQUIRE(fluxes == 44892.0); // the atmosphere reached every ocean cell
  REQUIRE(std::abs(sst - 291.04) < 1.0);
  REQUIRE(ifrac > 0.01);
  REQUIRE(ifrac < 0.2);
  REQUIRE(std::isfinite(sen_ice));
  atm.finalize();
  ocn.finalize();
  ice.finalize();
}

} // namespace test
} // namespace emulator

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Catch::Session session;
  if (rank != 0) {
    session.configData().outputFilename = "%debug";
  }
  int status = session.run(argc, argv);
  int worst = 0;
  MPI_Allreduce(&status, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return worst;
}
