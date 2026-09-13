// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"
#include "emulated_model.hpp"
#include "ocean_operators.hpp"
#include "samudra_channels.hpp"
#include "samudra_ocean.hpp"
#include "scrip_reader.hpp"

#include <mpi.h>
#ifdef EMULATOR_ENABLE_LIBTORCH
#include <torch/cuda.h>
#endif

#include <algorithm>
#include <cstdio>
#include <filesystem>
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

struct Setup {
  int rank = 0, size = 1;
  bool ok = false;
  grid::HorizontalGrid g;
  grid::Decomposition decomp;
  std::shared_ptr<inference::InferenceBackend> backend;
  std::vector<grid::GridField> ic;

  Setup() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int avail = 1;
#ifndef EMULATOR_ENABLE_LIBTORCH
    avail = 0;
#else
    avail = grid::have_scrip_reader() && std::filesystem::exists(kModel) &&
            std::filesystem::exists(kIc) && torch::cuda::is_available();
#endif
    MPI_Bcast(&avail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ok = avail != 0;
    if (!ok) {
      return;
    }
    register_ocn_operators();
    g = grid::read_scrip(kGrid);
    decomp = grid::Decomposition::contiguous_blocks(g.size(), size, rank);
    if (rank == 0) {
      inference::InferenceConfig c;
      c.backend = "libtorch";
      c.model_path = kModel;
      c.set("device", "cuda");
      backend = inference::create_backend(c, inference::InferenceContext{});
    }
    std::vector<std::string> names;
    for (const auto &in : samudra_e3smv3().inputs) {
      if (in.find(":next") == std::string::npos) {
        names.push_back(in);
      }
    }
    names.push_back("mask_2d");
    names.push_back("mask_ocean_sea_ice_fraction");
    ic = grid::read_grid_fields(kIc, names, g.ny, g.nx);
  }

  std::vector<double> local(const char *name) const {
    for (const auto &f : ic) {
      if (f.name == name) {
        return decomp.local(f.values);
      }
    }
    throw std::runtime_error(name);
  }

  fields::FieldSet exports() const {
    fields::FieldSet e(decomp.num_local());
    for (const auto &n : samudra_export_names()) {
      e.add(n);
    }
    return e;
  }
};

/// Every export and both exchange fields, identical?  Counts differences.
int compare(const fields::FieldSet &a, const fields::FieldSet &b,
            const coupling::Exchange &ea, const coupling::Exchange &eb,
            int step, int &first_step, std::string &first_name) {
  int differ = 0;
  auto same = [&](std::span<const double> x, std::span<const double> y,
                  const std::string &name) {
    if (!std::equal(x.begin(), x.end(), y.begin())) {
      if (differ == 0 && first_step < 0) {
        first_step = step;
        first_name = name;
      }
      ++differ;
    }
  };
  for (const auto &n : samudra_export_names()) {
    same(a.get(n), b.get(n), n);
  }
  for (const char *n : {"ocn.sst", "ocn.sea_ice_fraction"}) {
    same(ea.get(n), eb.get(n), n);
  }
  return differ;
}

int total(int local) {
  int all = 0;
  MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return all;
}

} // namespace

TEST_CASE("The SamudrACE ocean spec reproduces SamudraOcean bit for bit, "
          "through a window close and a restart", "[samudra][spec][real]") {
  Setup s;
  if (!s.ok) {
    WARN("skipped: needs libtorch, netCDF, a GPU and the Samudra files");
    return;
  }
  const auto n = s.decomp.num_local();
  coupling::Exchange ex_old, ex_new;
  // The atmosphere's side: the initial condition's own forcing, every step.
  for (auto *ex : {&ex_old, &ex_new}) {
    for (const auto &name : samudra_forcing_names()) {
      ex->publish("atm." + name, s.local(name.c_str()));
    }
  }
  SamudraOcean::Config config;
  config.layout = samudra_e3smv3();
  config.forcing_source = SamudraOcean::ForcingSource::Atmosphere;
  config.exchange = &ex_old;
  SamudraOcean old_ocn(config, MPI_COMM_WORLD, s.g, s.decomp, s.backend);

  const auto spec = model::ModelSpec::read(config::Section::load_spec(
      std::string(EMULATOR_SPEC_DIR) + "/samudra-e3smv3-ocean.yaml"));
  auto geometry = [&] {
    return model::Geometry::from_grid(MPI_COMM_WORLD, s.g, s.decomp);
  };
  model::EmulatedModel new_ocn(spec, 1800, geometry(), s.backend, &ex_new);

  fields::FieldSet imports(n);
  auto e_old = s.exports(), e_new = s.exports();
  old_ocn.initialize(after(0), s.ic);
  new_ocn.initialize(after(0), s.ic);
  old_ocn.initial_exports(e_old);
  new_ocn.initial_exports(after(0), imports, e_new);
  int first_step = -1;
  std::string first_name;
  int differ = compare(e_old, e_new, ex_old, ex_new, 0, first_step, first_name);

  const int steps = 300, restart_at = 100; // a window closes at 240
  coupling::MemoryRestartStore store;
  for (int k = 1; k <= restart_at; ++k) {
    old_ocn.run(after(k), imports, e_old);
    new_ocn.run(after(k), imports, e_new);
    differ += compare(e_old, e_new, ex_old, ex_new, k, first_step, first_name);
  }
  new_ocn.save_to(store);
  model::EmulatedModel restarted(spec, 1800, geometry(), s.backend, &ex_new);
  restarted.restart(store, s.ic);
  for (int k = restart_at + 1; k <= steps; ++k) {
    old_ocn.run(after(k), imports, e_old);
    restarted.run(after(k), imports, e_new);
    differ += compare(e_old, e_new, ex_old, ex_new, k, first_step, first_name);
  }
  REQUIRE(old_ocn.clock().completed_steps() == 1);
  REQUIRE(restarted.clock().completed_steps() == 1);
  const int all = total(differ);
  if (s.rank == 0) {
    std::printf("  SamudrACE ocean spec vs SamudraOcean: %d arrays differ over "
                "%d steps, restarted at %d (%d ranks)%s%s\n", all, steps,
                restart_at, s.size, all ? "; first: " : "",
                all ? first_name.c_str() : "");
  }
  REQUIRE(all == 0);
}

TEST_CASE("The coupler-forced ocean spec reproduces SamudraOcean bit for bit",
          "[samudra][spec][real]") {
  Setup s;
  if (!s.ok) {
    WARN("skipped: needs libtorch, netCDF, a GPU and the Samudra files");
    return;
  }
  const auto n = s.decomp.num_local();
  // A stand-in coupler handing back the initial condition's forcing.
  fields::FieldSet imports(n);
  for (const auto &name : coupler_forcing_imports()) {
    imports.add(name);
  }
  {
    const auto taux = s.local("TAUX"), tauy = s.local("TAUY");
    const auto prec = s.local("surface_precipitation_rate");
    const auto frz = s.local("frozen_precipitation_rate");
    const auto flus = s.local("FLUS"), flds = s.local("FLDS");
    const auto fsds = s.local("FSDS"), lh = s.local("LHFLX");
    const auto sh = s.local("SHFLX"), ice = s.local("ocean_sea_ice_fraction");
    for (std::size_t i = 0; i < n; ++i) {
      imports.get("Foxx_taux")[i] = -taux[i];
      imports.get("Foxx_tauy")[i] = -tauy[i];
      imports.get("Faxa_snow")[i] = frz[i];
      imports.get("Faxa_rain")[i] = prec[i] - frz[i];
      imports.get("Foxx_lwup")[i] = -flus[i];
      imports.get("Faxa_lwdn")[i] = flds[i];
      imports.get("Foxx_swnet")[i] = fsds[i] * 0.94;
      imports.get("Foxx_lat")[i] = -lh[i];
      imports.get("Foxx_sen")[i] = -sh[i];
      imports.get("Si_ifrac")[i] = ice[i];
    }
  }
  coupling::Exchange ex_old, ex_new;
  SamudraOcean::Config config;
  config.layout = samudra_e3smv3();
  config.forcing_source = SamudraOcean::ForcingSource::Coupler;
  config.exchange = &ex_old;
  SamudraOcean old_ocn(config, MPI_COMM_WORLD, s.g, s.decomp, s.backend);
  const auto spec = model::ModelSpec::read(config::Section::load_spec(
      std::string(EMULATOR_SPEC_DIR) +
      "/samudra-e3smv3-ocean-coupler-forced.yaml"));
  REQUIRE(spec.imports.size() == 10);
  REQUIRE(spec.layout->inputs == samudra_e3smv3().inputs); // from extends
  model::EmulatedModel new_ocn(
      spec, 1800, model::Geometry::from_grid(MPI_COMM_WORLD, s.g, s.decomp),
      s.backend, &ex_new);

  auto e_old = s.exports(), e_new = s.exports();
  old_ocn.initialize(after(0), s.ic);
  new_ocn.initialize(after(0), s.ic);
  old_ocn.initial_exports(e_old);
  new_ocn.initial_exports(after(0), imports, e_new);
  int first_step = -1;
  std::string first_name;
  int differ = compare(e_old, e_new, ex_old, ex_new, 0, first_step, first_name);
  for (int k = 1; k <= 250; ++k) {
    old_ocn.run(after(k), imports, e_old);
    new_ocn.run(after(k), imports, e_new);
    differ += compare(e_old, e_new, ex_old, ex_new, k, first_step, first_name);
  }
  REQUIRE(new_ocn.clock().completed_steps() == 1);
  const int all = total(differ);
  if (s.rank == 0) {
    std::printf("  coupler-forced ocean spec vs SamudraOcean: %d arrays differ "
                "over 250 steps (%d ranks)\n", all, s.size);
  }
  REQUIRE(all == 0);
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
