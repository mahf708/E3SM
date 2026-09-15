// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "ace_operators.hpp"
#include "emulator_component.hpp"
#include "emulator_test_support.hpp"
#include "grid_field_reader.hpp"
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

const std::string &kGrid = emulator::test::kGaussianGrid;
const std::string &kModel = emulator::test::kAce2Model;
const std::string &kIc = emulator::test::kAce2Ic;

// A realistic slice of the coupler's lists: fields the atmosphere reads
// mixed with ones it ignores, and exports it does not produce.
const std::string kX2a = "Sx_avsdr:Sx_anidr:Sf_lfrac:Sf_ifrac:Sf_ofrac:Sx_t:"
                         "So_t:Sl_snowh:Faxx_lat:Faxx_sen";
const std::string kA2x =
    "Sa_z:Sa_topo:Sa_u:Sa_v:Sa_tbot:Sa_ptem:Sa_shum:Sa_pbot:Sa_dens:Sa_pslv:"
    "Sa_co2prog:Faxa_rainc:Faxa_rainl:Faxa_snowc:Faxa_snowl:Faxa_lwdn:"
    "Faxa_swndr:Faxa_swvdr:Faxa_swndf:Faxa_swvdf:Faxa_swnet:Faxa_bcphidry";

} // namespace

TEST_CASE("The ACE2 spec runs as a component through MCT-laid-out "
          "buffers", "[atm][real]") {
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
    WARN("skipped: needs libtorch, netCDF, a GPU and the ACE2 files");
    return;
  }

  const std::string atm_in =
      (std::filesystem::temp_directory_path() /
       ("atm_in_real_" + std::to_string(::getpid()) + "_" +
        std::to_string(rank)))
          .string();
  std::ofstream(atm_in) << "spec: " << spec_path("ace2-eamv3.yaml") << "\n"
                        << "coupler_dt: 1800\n"
                        << "grid: {file: " << kGrid << ", domain: full}\n"
                        << "initial_condition: " << kIc << "\n"
                        << "inference: {backend: libtorch, model_path: "
                        << kModel << ", device: cuda}\n";

  atm::register_atm_operators();
  EmulatorComponent atm(EmulatorType::ATM_COMP, "emulatoratm");
  atm.create_instance(MPI_Comm_c2f(MPI_COMM_WORLD), 1, atm_in, "", 0,
                      19710101, 0);
  std::remove(atm_in.c_str());
  const auto n = static_cast<std::size_t>(atm.get_num_local_cols());

  AttrVect x2a(kX2a, n, -999.0), a2x(kA2x, n, -999.0);
  // The stand-in surface: the initial condition's fractions made a
  // partition, and a merged surface temperature to match.
  {
    const auto ic = grid::read_grid_fields(
        kIc, {"LANDFRAC", "OCNFRAC", "ICEFRAC", "TS"}, 180, 360);
    std::vector<int> gids(n);
    atm.get_local_col_gids(gids.data());
    for (std::size_t p = 0; p < n; ++p) {
      const auto c = static_cast<std::size_t>(gids[p] - 1);
      auto clean = [&](int k) {
        const double v = ic[k].values[c];
        return std::isfinite(v) ? std::clamp(v, 0.0, 1.0) : 0.0;
      };
      double l = clean(0), o = clean(1), i = clean(2);
      const double sum = l + o + i;
      if (sum > 1.0) {
        l /= sum;
        o /= sum;
        i /= sum;
      }
      x2a.at("Sf_lfrac", p) = l;
      x2a.at("Sf_ofrac", p) = o;
      x2a.at("Sf_ifrac", p) = i;
      x2a.at("Sx_t", p) = (l + o + i) * ic[3].values[c];
    }
  }

  atm.set_coupler_field_lists(kX2a, kA2x);
  atm.setup_coupling(coupling(x2a, a2x, n));
  REQUIRE(atm.export_binding()->unbound() ==
          std::vector<std::string>{"Sa_co2prog", "Faxa_bcphidry"});

  atm.initialize();
  // The initial state reached the coupler before any run.
  REQUIRE(a2x.at("Sa_tbot", 0) > 150.0);
  REQUIRE(a2x.at("Sa_co2prog", 0) == 0.0);

  for (int step = 1; step <= 24; ++step) {
    const int s = step * 1800;
    atm.run(1800, {19710101 + s / 86400, s % 86400});
  }

  double sums[4] = {0, 0, 0, 0};
  for (std::size_t p = 0; p < n; ++p) {
    sums[0] += a2x.at("Sa_tbot", p);
    sums[1] += a2x.at("Faxa_lwdn", p);
    sums[2] += a2x.at("Sa_co2prog", p) == 0.0 &&
                       a2x.at("Faxa_bcphidry", p) == 0.0
                   ? 0.0
                   : 1.0;
    sums[3] += 1.0;
  }
  double total[4] = {0, 0, 0, 0};
  MPI_Allreduce(sums, total, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("  after 12 h through the buffers: mean Sa_tbot %.2f K, "
                "Faxa_lwdn %.1f W/m2 (unweighted, %d ranks)\n",
                total[0] / total[3], total[1] / total[3], size);
  }
  REQUIRE(total[0] / total[3] > 260.0);
  REQUIRE(total[0] / total[3] < 295.0);
  REQUIRE(total[2] == 0.0); // fields the atmosphere does not produce stay 0

  // Without the driver's time the component refuses to guess it.
  REQUIRE_THROWS_WITH(atm.run(1800), Catch::Contains("emulator_run_at"));
  atm.finalize();
}

} // namespace test
} // namespace emulator

EMULATOR_TEST_MPI_MAIN
