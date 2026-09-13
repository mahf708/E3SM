// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "ace_atmosphere.hpp"
#include "ace_operators.hpp"
#include "emulated_model.hpp"
#include "ace_channels.hpp"
#include "create_inference_backend.hpp"
#include "scrip_reader.hpp"

#include <mpi.h>
#ifdef EMULATOR_ENABLE_LIBTORCH
#include <torch/cuda.h>
#endif

#include <algorithm>
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

bool available(std::string &why) {
#ifndef EMULATOR_ENABLE_LIBTORCH
  why = "no libtorch backend";
  return false;
#else
  if (!grid::have_scrip_reader()) {
    why = "no netCDF";
    return false;
  }
  for (const auto &f : {kGrid, kModel, kIc}) {
    if (!std::filesystem::exists(f)) {
      why = f + " is not readable";
      return false;
    }
  }
  if (!torch::cuda::is_available()) {
    why = "no GPU";
    return false;
  }
  return true;
#endif
}

coupling::ModelTime after(int n) { // from 1971-01-01 00:00, 30 min steps
  const int s = n * 1800;
  return {19710101 + s / 86400, s % 86400};
}

double global_mean(std::span<const double> f, std::span<const double> area) {
  double s[2] = {0, 0}, t[2] = {0, 0};
  for (std::size_t i = 0; i < f.size(); ++i) {
    s[0] += f[i] * area[i];
    s[1] += area[i];
  }
  MPI_Allreduce(s, t, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return t[0] / t[1];
}

} // namespace

TEST_CASE("The ACE2 spec reproduces AceAtmosphere bit for bit on the real "
          "checkpoint, through a mid-interval restart", "[ace][spec][real]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::string why;
  int ok = available(why) ? 1 : 0;
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ok) {
    WARN("skipped: " << why);
    return;
  }
  register_atm_operators();

  const auto g = grid::read_scrip(kGrid);
  const auto decomp =
      grid::Decomposition::contiguous_blocks(g.size(), size, rank);
  const auto area = decomp.local(g.area);
  const auto layout = ace2_eamv3();
  std::vector<std::string> ic_names = layout.inputs;
  const auto ic = grid::read_grid_fields(kIc, ic_names, g.ny, g.nx);
  const auto fsds_ic = grid::read_grid_fields(kIc, {"FSDS"}, g.ny, g.nx);

  std::shared_ptr<inference::InferenceBackend> backend;
  if (rank == 0) {
    inference::InferenceConfig config;
    config.backend = "libtorch";
    config.model_path = kModel;
    config.set("device", "cuda");
    backend = inference::create_backend(config, inference::InferenceContext{});
  }

  AceAtmosphere::Config config;
  config.layout = layout;
  config.surface.layer = SurfaceLayer::LowestLevel; // ACE2 has no 2 m / 10 m
  config.orbit = Orbit::from_elements(0.016715, 23.4441, 102.7);

  // The coupler's side: the initial condition's own surface, held fixed.
  fields::FieldSet imports(decomp.num_local());
  auto local_ic = [&](const char *name) {
    for (const auto &f : ic) {
      if (f.name == name) {
        auto v = f.values;
        for (auto &x : v) {
          x = std::isfinite(x) && std::abs(x) < 1e30 ? x : 0.0;
        }
        return decomp.local(v);
      }
    }
    throw std::runtime_error(name);
  };
  // The initial condition's fractions are coarsened training data, not a
  // partition: outside [0, 1] by up to 0.10 and summing above one by up to
  // 0.075 in most cells.  A coupler's are exact, so the stand-in coupler
  // clips and renormalizes them, as a real surface would deliver.
  {
    auto l = local_ic("LANDFRAC");
    auto o = local_ic("OCNFRAC");
    auto i = local_ic("ICEFRAC");
    for (std::size_t k = 0; k < l.size(); ++k) {
      l[k] = std::clamp(l[k], 0.0, 1.0);
      o[k] = std::clamp(o[k], 0.0, 1.0);
      i[k] = std::clamp(i[k], 0.0, 1.0);
      const double sum = l[k] + o[k] + i[k];
      if (sum > 1.0) {
        l[k] /= sum;
        o[k] /= sum;
        i[k] /= sum;
      }
    }
    const auto ts = local_ic("TS");
    for (auto [cpl, v] : {std::pair{"Sf_lfrac", &l}, std::pair{"Sf_ofrac", &o},
                          std::pair{"Sf_ifrac", &i}}) {
      auto dest = imports.add(cpl);
      std::copy(v->begin(), v->end(), dest.begin());
    }
    auto sx_t = imports.add("Sx_t");
    for (std::size_t k = 0; k < ts.size(); ++k) {
      sx_t[k] = (l[k] + o[k] + i[k]) * ts[k]; // merged: fraction-weighted
    }
  }
  auto make_exports = [&] {
    fields::FieldSet e(decomp.num_local());
    for (const auto &n : ace_export_names()) {
      e.add(n);
    }
    return e;
  };

  const auto spec = model::ModelSpec::read(config::Section::load_file(
      std::string(EMULATOR_SPEC_DIR) + "/ace2-eamv3.yaml"));
  auto geometry = [&] {
    return model::Geometry::from_grid(MPI_COMM_WORLD, g, decomp);
  };

  // One backend for both: the network is deterministic, and each step's
  // seed comes from the step index either way.
  AceAtmosphere old_atm(config, MPI_COMM_WORLD, g, decomp, backend);
  model::EmulatedModel new_atm(spec, 1800, geometry(), backend, nullptr);
  REQUIRE(new_atm.initial_condition_names().size() == layout.inputs.size() - 1);

  old_atm.initialize(after(0), ic);
  new_atm.initialize(after(0), ic);
  auto old_exports = make_exports(), new_exports = make_exports();
  old_atm.initial_exports(after(0), old_exports);
  new_atm.initial_exports(after(0), imports, new_exports);

  int differ = 0;
  auto compare = [&](int n) {
    for (const auto &name : ace_export_names()) {
      const auto a = old_exports.get(name), b = new_exports.get(name);
      if (!std::equal(a.begin(), a.end(), b.begin())) {
        if (differ == 0) {
          INFO(name << " differs at step " << n);
          CHECK(false);
        }
        ++differ;
      }
    }
  };
  compare(0);
  const int total = 48, restart_at = 19;
  coupling::MemoryRestartStore store;
  for (int n = 1; n <= restart_at; ++n) {
    old_atm.run(after(n), imports, old_exports);
    new_atm.run(after(n), imports, new_exports);
    compare(n);
  }
  new_atm.save_to(store);
  model::EmulatedModel restarted(spec, 1800, geometry(), backend, nullptr);
  restarted.restart(store, ic);
  for (int n = restart_at + 1; n <= total; ++n) {
    old_atm.run(after(n), imports, old_exports);
    if (n % 5 == 0) {
      old_atm.run(after(n), imports, old_exports);
      restarted.run(after(n), imports, new_exports);
    }
    restarted.run(after(n), imports, new_exports);
    compare(n);
  }
  int all = 0;
  MPI_Allreduce(&differ, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("  ACE2 spec vs AceAtmosphere: %d export arrays differ over "
                "%d steps, restarted at %d (%d ranks)\n", all, total,
                restart_at, size);
  }
  REQUIRE(all == 0);
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
