// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"
#include "samudra_channels.hpp"
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
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace emulator {
namespace ocn {
namespace test {

namespace {

const std::string kGrid = "/global/cfs/cdirs/e3sm/inputdata/share/meshes/"
                          "gaussian_180x360_latlon.scrip.20260127.nc";
const std::string kModel = "/pscratch/sd/m/mahf708/SamudrACE-E3SMv3/eocn/"
                           "samudra_ocn_traced_masked_cuda.pt";
const std::string kIc = "/pscratch/sd/m/mahf708/SamudrACE-E3SMv3/eocn/"
                        "samudra_ocn_ic_0_icemask.nc";

coupling::ModelTime after(int n) {
  const int s = n * 1800;
  return {20000101 + s / 86400, s % 86400};
}

struct Stats {
  double min, max, mean;
};

/// Unweighted min, max and mean over the ocean mask, across ranks.
Stats over_mask(std::span<const double> f, std::span<const double> mask) {
  double lo = std::numeric_limits<double>::max();
  double hi = -lo;
  double s[2] = {0, 0};
  for (std::size_t i = 0; i < f.size(); ++i) {
    if (mask[i] == 1.0) {
      lo = std::min(lo, f[i]);
      hi = std::max(hi, f[i]);
      s[0] += f[i];
      s[1] += 1.0;
    }
  }
  double glo = 0, ghi = 0, gs[2] = {0, 0};
  MPI_Allreduce(&lo, &glo, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&hi, &ghi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(s, gs, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return {glo, ghi, gs[0] / gs[1]};
}

} // namespace

TEST_CASE("Samudra's first step reproduces an independent Python run, and a "
          "mid-window restart is exact", "[samudra][real]") {
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

  const auto g = grid::read_scrip(kGrid);
  const auto decomp =
      grid::Decomposition::contiguous_blocks(g.size(), size, rank);
  std::shared_ptr<inference::InferenceBackend> backend;
  if (rank == 0) {
    inference::InferenceConfig config;
    config.backend = "libtorch";
    config.model_path = kModel;
    config.set("device", "cuda");
    backend = inference::create_backend(config, inference::InferenceContext{});
  }
  SamudraOcean::Config config;
  config.layout = samudra_e3smv3();

  auto make = [&] {
    return SamudraOcean(config, MPI_COMM_WORLD, g, decomp, backend);
  };
  SamudraOcean ocean = make();
  const auto ic = grid::read_grid_fields(kIc, ocean.initial_condition_names(),
                                         g.ny, g.nx);
  ocean.initialize(after(0), ic);

  // An independent reference: the same traced model and initial condition,
  // assembled and run from Python (tests/samudra_reference.py, torch 2.10,
  // netCDF4) on an A100, 2026-09-12.  emulator_comps/eocn/VERIFICATION.md
  // section 1 records a different table for "the published initial
  // condition" (sst 269.13-309.14, mean 286.76); it does not reproduce with
  // samudra_ocn_traced_masked_cuda.pt and either published IC file, so it
  // came from an earlier trace or input and is not used here.
  const std::map<std::string, Stats> recorded{
      {"sst", {270.24, 305.32, 286.65}},
      {"ssh", {-1.23, 0.86, -0.05}},
      {"salinityCoarsened_0", {0.00, 49.28, 33.39}},
      {"temperatureCoarsened_0", {-2.87, 32.10, 13.47}},
      {"velocityZonalCoarsened_0", {-1.08, 1.09, 0.00}},
      {"ocean_sea_ice_fraction", {0.00, 1.00, 0.28}},
      {"iceVolumeTotal", {0.00, 50.31, 0.60}}};
  for (const auto &[name, want] : recorded) {
    const auto got = over_mask(ocean.brackets().upper(name), ocean.ocean_mask());
    if (rank == 0) {
      std::printf("  %-26s min %8.2f (%8.2f)  max %8.2f (%8.2f)  mean %7.2f "
                  "(%7.2f)\n",
                  name.c_str(), got.min, want.min, got.max, want.max, got.mean,
                  want.mean);
    }
    INFO(name);
    // The reference is printed to two decimals.
    CHECK(got.min == Approx(want.min).margin(0.006));
    CHECK(got.max == Approx(want.max).margin(0.006));
    CHECK(got.mean == Approx(want.mean).margin(0.006));
  }

  // A stand-in coupler that hands back the initial condition's forcing, so
  // the window mean is the forcing the model started from.
  fields::FieldSet imports(decomp.num_local());
  for (const auto &n : coupler_forcing_imports()) {
    imports.add(n);
  }
  auto local = [&](const char *name) {
    for (const auto &f : ic) {
      if (f.name == name) {
        return decomp.local(f.values);
      }
    }
    throw std::runtime_error(name);
  };
  {
    const auto taux = local("TAUX"), tauy = local("TAUY");
    const auto prec = local("surface_precipitation_rate");
    const auto frz = local("frozen_precipitation_rate");
    const auto flus = local("FLUS"), flds = local("FLDS"), fsds = local("FSDS");
    const auto lh = local("LHFLX"), sh = local("SHFLX");
    for (std::size_t i = 0; i < decomp.num_local(); ++i) {
      imports.get("Foxx_taux")[i] = -taux[i];
      imports.get("Foxx_tauy")[i] = -tauy[i];
      imports.get("Faxa_snow")[i] = frz[i];
      imports.get("Faxa_rain")[i] = prec[i] - frz[i];
      imports.get("Foxx_lwup")[i] = -flus[i];
      imports.get("Faxa_lwdn")[i] = flds[i];
      imports.get("Foxx_swnet")[i] = fsds[i] * 0.94;
      imports.get("Foxx_lat")[i] = -lh[i];
      imports.get("Foxx_sen")[i] = -sh[i];
    }
  }
  auto make_exports = [&] {
    fields::FieldSet e(decomp.num_local());
    for (const auto &n : samudra_export_names()) {
      e.add(n);
    }
    return e;
  };

  const int total = 300, restart_at = 100; // a window closes at step 240
  std::vector<std::vector<double>> reference(total + 1);
  auto exports = make_exports();
  for (int n = 1; n <= total; ++n) {
    ocean.run(after(n), imports, exports);
    reference[n].assign(exports.get("So_t").begin(), exports.get("So_t").end());
    const auto ice = ocean.sea_ice_fraction();
    reference[n].insert(reference[n].end(), ice.begin(), ice.end());
  }
  REQUIRE(ocean.clock().completed_steps() == 1);

  // The ice fraction never leaves its own mask, and So_t never drops below
  // freezing.
  int outside = 0;
  double coldest = 1e9;
  for (std::size_t i = 0; i < decomp.num_local(); ++i) {
    outside += (ocean.ice_mask()[i] == 0.0 && ocean.sea_ice_fraction()[i] != 0.0);
    coldest = std::min(coldest, exports.get("So_t")[i]);
  }
  int all_outside = 0;
  MPI_Allreduce(&outside, &all_outside, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(all_outside == 0);
  REQUIRE(coldest >= 271.35);
  const auto sst2 = over_mask(ocean.brackets().upper("sst"), ocean.ocean_mask());
  if (rank == 0) {
    std::printf("  after the second step (day 10): sst min %.2f max %.2f mean "
                "%.2f K\n", sst2.min, sst2.max, sst2.mean);
  }
  REQUIRE(sst2.min > 265.0);
  REQUIRE(sst2.max < 315.0);

  // Restart at step 100, in the middle of the first window.
  coupling::MemoryRestartStore store;
  {
    SamudraOcean first = make();
    first.initialize(after(0), ic);
    auto e = make_exports();
    for (int n = 1; n <= restart_at; ++n) {
      first.run(after(n), imports, e);
    }
    first.save_to(store);
  }
  SamudraOcean second = make();
  second.restart(store, ic);
  int differ = 0;
  for (int n = restart_at + 1; n <= total; ++n) {
    second.run(after(n), imports, exports);
    std::vector<double> now(exports.get("So_t").begin(), exports.get("So_t").end());
    const auto ice = second.sea_ice_fraction();
    now.insert(now.end(), ice.begin(), ice.end());
    differ += now == reference[n] ? 0 : 1;
  }
  int all_differ = 0;
  MPI_Allreduce(&differ, &all_differ, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("  restart at step %d: %d of %d later steps differ, across a "
                "window close at 240\n", restart_at, all_differ,
                total - restart_at);
  }
  REQUIRE(all_differ == 0);
}

} // namespace test
} // namespace ocn
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
