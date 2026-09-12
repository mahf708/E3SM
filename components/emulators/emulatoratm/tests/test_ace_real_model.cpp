// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "ace_channels.hpp"
#include "create_inference_backend.hpp"
#include "global_gather.hpp"
#include "grid_field_reader.hpp"
#include "network_stepper.hpp"
#include "scrip_reader.hpp"

#include <mpi.h>
#ifdef EMULATOR_ENABLE_LIBTORCH
#include <torch/cuda.h>
#endif

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace emulator {
namespace atm {
namespace test {

namespace {

const std::string kGrid = "/global/cfs/cdirs/e3sm/inputdata/share/meshes/"
                          "gaussian_180x360_latlon.scrip.20260127.nc";
const std::string kModel =
    "/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/ace_traced_cuda.pt";
const std::string kIc = "/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/"
                        "initial_conditions/1971010100_time_1.nc";

bool real_model_available(std::string &why) {
#ifndef EMULATOR_ENABLE_LIBTORCH
  why = "this build has no libtorch backend";
  return false;
#else
  if (!grid::have_scrip_reader()) {
    why = "this build has no netCDF";
    return false;
  }
  for (const auto &f : {kGrid, kModel, kIc}) {
    if (!std::filesystem::exists(f)) {
      why = f + " is not readable here";
      return false;
    }
  }
  if (!torch::cuda::is_available()) {
    why = "the checkpoint was traced on CUDA and there is no GPU here";
    return false;
  }
  return true;
#endif
}

/// Area-weighted global mean of a decomposed field.
double global_mean(std::span<const double> local, std::span<const double> area) {
  double sums[2] = {0, 0};
  for (std::size_t i = 0; i < local.size(); ++i) {
    sums[0] += local[i] * area[i];
    sums[1] += area[i];
  }
  double total[2] = {0, 0};
  MPI_Allreduce(sums, total, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return total[0] / total[1];
}

} // namespace

TEST_CASE("The traced ACE2-EAMv3 checkpoint steps a day from its initial "
          "condition across the ranks", "[ace][real]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string why;
  int available = real_model_available(why) ? 1 : 0;
  MPI_Bcast(&available, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!available) {
    WARN("skipped: " << why);
    return;
  }

  const auto layout = ace2_eamv3();
  const auto g = grid::read_scrip(kGrid);
  const auto decomp = grid::Decomposition::contiguous_blocks(g.size(), size,
                                                             rank);
  const grid::GlobalGather gather(MPI_COMM_WORLD, decomp);
  const auto area = decomp.local(g.area);

  std::shared_ptr<inference::InferenceBackend> backend;
  if (gather.is_root()) {
    inference::InferenceConfig config;
    config.backend = "libtorch";
    config.model_path = kModel;
    config.set("device", "cuda");
    config.set("dtype", "float32");
    backend = inference::create_backend(config, inference::InferenceContext{});
  }
  coupling::NetworkStepper stepper(layout, MPI_COMM_WORLD, gather, g.nx, g.ny,
                                   backend);

  // The initial condition has every input channel, and the flux outputs of
  // the same time, which make a reference for the first prediction.
  std::vector<std::string> names = layout.inputs;
  for (const char *extra : {"FLDS", "FSDS", "LHFLX"}) {
    names.emplace_back(extra);
  }
  const auto ic = grid::read_grid_fields(kIc, names, g.ny, g.nx);
  std::map<std::string, std::vector<double>> reference;
  for (const auto &field : ic) {
    INFO(field.name << ": " << field.non_finite << " non-finite, "
                    << field.fill_like << " fill-like");
    const bool fraction = field.name.find("FRAC") != std::string::npos;
    if (!fraction) {
      REQUIRE(field.unusable() == 0);
    }
    auto values = field.values;
    for (auto &v : values) {
      if (!std::isfinite(v) || std::abs(v) >= 1e30) {
        v = 0.0; // fractions only, by the check above
      }
    }
    auto local = decomp.local(values);
    if (stepper.inputs().contains(field.name)) {
      auto dest = stepper.inputs().get(field.name);
      std::copy(local.begin(), local.end(), dest.begin());
    }
    reference[field.name] = std::move(local);
  }

  // Collectives: every rank computes every mean, whoever prints them.
  const double ps0 = global_mean(reference["PS"], area);
  const double flds0 = global_mean(reference["FLDS"], area);
  const double ts0 = global_mean(reference["TS"], area);
  const double fsds0 = global_mean(reference["FSDS"], area);
  const double lh0 = global_mean(reference["LHFLX"], area);
  if (rank == 0) {
    std::printf("  ranks %d; IC global means: PS %.1f Pa, TS %.2f K, "
                "FLDS %.2f, FSDS %.2f, LHFLX %.2f W/m2\n",
                size, ps0, ts0, flds0, fsds0, lh0);
  }

  double ps_checksum = 0.0;
  for (int step = 1; step <= 4; ++step) {
    stepper.step(step);
    const auto &p = stepper.prediction();
    const double ps = global_mean(p.get("PS"), area);
    const double t7 = global_mean(p.get("T_7"), area);
    const double flds = global_mean(p.get("FLDS"), area);
    const double fsds = global_mean(p.get("FSDS"), area);
    const double lh = global_mean(p.get("LHFLX"), area);
    const double pr = global_mean(p.get("surface_precipitation_rate"), area);
    if (rank == 0) {
      std::printf("  step %d (+%2d h): PS %.1f  T_7 %.2f  FLDS %.2f  FSDS "
                  "%.2f  LHFLX %.2f  P %.3f mm/day\n",
                  step, 6 * step, ps, t7, flds, fsds, lh, pr * 86400.0);
    }
    // Plausibility, not skill: a day from a real state stays a real state.
    REQUIRE(std::abs(ps - ps0) / ps0 < 0.005);
    REQUIRE(t7 > 270.0);
    REQUIRE(t7 < 295.0);
    REQUIRE(std::abs(flds - flds0) < 30.0);
    REQUIRE(lh > 50.0);
    REQUIRE(lh < 130.0);
    if (step == 4) {
      const auto local_ps = p.get("PS");
      double local = 0.0;
      for (const double v : local_ps) {
        local += v;
      }
      MPI_Allreduce(&local, &ps_checksum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
    }
  }
  if (rank == 0) {
    // The same number on any rank count: gather and scatter move values,
    // they must not change them.
    std::printf("  PS checksum after 4 steps: %.17g\n", ps_checksum);
  }
}

} // namespace test
} // namespace atm
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
