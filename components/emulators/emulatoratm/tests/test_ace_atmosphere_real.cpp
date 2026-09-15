// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "ace_operators.hpp"
#include "emulated_model.hpp"
#include "emulator_test_support.hpp"
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

const std::string &kGrid = emulator::test::kGaussianGrid;
const std::string &kModel = emulator::test::kAce2Model;
const std::string &kIc = emulator::test::kAce2Ic;

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

TEST_CASE("The ACE atmosphere runs a coupled day and restarts mid-interval "
          "bit for bit", "[ace][real]") {
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

  const auto g = grid::read_scrip(kGrid);
  const auto decomp =
      grid::Decomposition::contiguous_blocks(g.size(), size, rank);
  const auto area = decomp.local(g.area);
  register_atm_operators();
  const auto spec = model::ModelSpec::read(config::Section::load_spec(
      emulator::test::spec_path("ace2-eamv3.yaml")));
  const auto &layout = *spec.layout;
  std::vector<std::string> export_names;
  for (const auto &e : spec.exports) {
    export_names.push_back(e.name);
  }
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
  // The initial condition's fractions are coarsened training data, not an
  // exact partition, so the stand-in coupler clips and renormalizes them,
  // as a real surface would deliver.
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
    for (const auto &n : export_names) {
      e.add(n);
    }
    return e;
  };

  const int total = 48;
  const int restart_at = 19; // inside the second 6 h interval
  std::vector<std::map<std::string, std::vector<double>>> reference(total + 1);
  std::vector<double> swdown_daily(decomp.num_local(), 0.0);

  // --- the continuous run ---
  {
    model::EmulatedModel atm(spec, 1800, model::Geometry::from_grid(MPI_COMM_WORLD, g, decomp), backend, nullptr);
    atm.initialize(after(0), ic);
    auto exports = make_exports();
    for (int n = 1; n <= total; ++n) {
      atm.run(after(n), imports, exports);
      if (n % 5 == 0) {
        atm.run(after(n), imports, exports); // the driver's repeat
      }
      for (const auto &name : export_names) {
        const auto v = exports.get(name);
        reference[n][name].assign(v.begin(), v.end());
      }
      for (std::size_t i = 0; i < swdown_daily.size(); ++i) {
        swdown_daily[i] += (exports.get("Faxa_swvdr")[i] +
                            exports.get("Faxa_swndr")[i] +
                            exports.get("Faxa_swvdf")[i] +
                            exports.get("Faxa_swndf")[i]) /
                           total;
      }
      if (n % 12 == 0) {
        const double tbot = global_mean(exports.get("Sa_tbot"), area);
        const double z = global_mean(exports.get("Sa_z"), area);
        const double lw = global_mean(exports.get("Faxa_lwdn"), area);
        const double sw = global_mean(exports.get("Faxa_swnet"), area);
        const double rain = global_mean(exports.get("Faxa_rainl"), area);
        const double snow = global_mean(exports.get("Faxa_snowl"), area);
        if (rank == 0) {
          std::printf("  +%2d h: Sa_tbot %.2f K  Sa_z %.0f m  lwdn %.1f  "
                      "swnet %.1f W/m2  rain %.2f snow %.2f mm/day\n",
                      n / 2, tbot, z, lw, sw, rain * 86400, snow * 86400);
        }
        REQUIRE(tbot > 270.0);
        REQUIRE(tbot < 295.0);
        REQUIRE(z > 300.0);
        REQUIRE(z < 600.0);
      }
    }
    REQUIRE(atm.clock().completed_steps() == 4);
  }

  // The diurnal rescaling moves shortwave within the day, not across it:
  // the daily mean of the four bands stays close to the network's FSDS.
  const auto fsds_local = decomp.local(fsds_ic[0].values);
  const double daily = global_mean(swdown_daily, area);
  const double fsds0 = global_mean(fsds_local, area);
  if (rank == 0) {
    std::printf("  daily-mean downwelling shortwave %.2f W/m2 against the "
                "initial condition's FSDS %.2f\n", daily, fsds0);
  }
  REQUIRE(std::abs(daily - fsds0) < 0.1 * fsds0);

  // --- the same day, restarted mid-interval into a new component ---
  coupling::MemoryRestartStore store;
  {
    model::EmulatedModel first(spec, 1800, model::Geometry::from_grid(MPI_COMM_WORLD, g, decomp), backend, nullptr);
    first.initialize(after(0), ic);
    auto exports = make_exports();
    int before_restart = 0;
    double worst_before = 0.0;
    for (int n = 1; n <= restart_at; ++n) {
      first.run(after(n), imports, exports);
      if (n % 5 == 0) {
        first.run(after(n), imports, exports);
      }
      bool same = true;
      for (const auto &name : export_names) {
        const auto v = exports.get(name);
        const auto &ref = reference[n][name];
        for (std::size_t k = 0; k < v.size(); ++k) {
          if (v[k] != ref[k]) {
            same = false;
            worst_before = std::max(worst_before, std::abs(v[k] - ref[k]));
          }
        }
      }
      before_restart += same ? 0 : 1;
    }
    if (rank == 0) {
      std::printf("  a second run from the same initial condition differs "
                  "from the first at %d of %d steps (max |diff| %.3g)\n",
                  before_restart, restart_at, worst_before);
    }
    first.save_to(store);
  }
  model::EmulatedModel second(spec, 1800, model::Geometry::from_grid(MPI_COMM_WORLD, g, decomp), backend, nullptr);
  second.restart(store, ic);
  auto exports = make_exports();
  second.run(after(restart_at), imports, exports); // the restarted repeat
  int mismatched_steps = 0;
  double worst_after = 0.0;
  std::string worst_field;
  int worst_step = 0;
  for (int n = restart_at + 1; n <= total; ++n) {
    second.run(after(n), imports, exports);
    if (n % 5 == 0) {
      second.run(after(n), imports, exports);
    }
    bool same = true;
    for (const auto &name : export_names) {
      const auto v = exports.get(name);
      const auto &ref = reference[n][name];
      for (std::size_t k = 0; k < v.size(); ++k) {
        if (v[k] != ref[k]) {
          same = false;
          if (std::abs(v[k] - ref[k]) > worst_after) {
            worst_after = std::abs(v[k] - ref[k]);
            worst_field = name;
            worst_step = n;
          }
        }
      }
    }
    mismatched_steps += same ? 0 : 1;
  }
  int all = 0;
  MPI_Allreduce(&mismatched_steps, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("  restart at step %d: %d of %d later steps differ from the "
                "continuous run (summed over ranks)\n",
                restart_at, all, total - restart_at);
    std::printf("  worst difference %.3g in %s at step %d\n", worst_after,
                worst_field.c_str(), worst_step);
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
