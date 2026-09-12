// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "ace_atmosphere.hpp"
#include "ace_channels.hpp"
#include "create_inference_backend.hpp"
#include "exchange.hpp"
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
#include <string>
#include <vector>

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

coupling::ModelTime after(int n) {
  const int s = n * 1800;
  return {20000101 + s / 86400, s % 86400};
}

std::shared_ptr<inference::InferenceBackend>
backend_on_root(int rank, const std::string &model, const std::string &seed) {
  if (rank != 0) {
    return nullptr;
  }
  inference::InferenceConfig c;
  c.backend = "libtorch";
  c.model_path = model;
  c.set("device", "cuda");
  if (!seed.empty()) {
    c.set("seed", seed);
  }
  return inference::create_backend(c, inference::InferenceContext{});
}

double mean_where(std::span<const double> f, std::span<const double> w) {
  double s[2] = {0, 0}, t[2] = {0, 0};
  for (std::size_t i = 0; i < f.size(); ++i) {
    s[0] += f[i] * w[i];
    s[1] += w[i];
  }
  MPI_Allreduce(s, t, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return t[0] / t[1];
}

} // namespace

TEST_CASE("SamudrACE runs coupled in one process: the ocean forced by the "
          "atmosphere's own fluxes, the atmosphere on the ocean's SST",
          "[samudrace][real]") {
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

  const auto g = grid::read_scrip(kGrid);
  const auto decomp =
      grid::Decomposition::contiguous_blocks(g.size(), size, rank);
  const auto n = decomp.num_local();
  const auto area = decomp.local(g.area);
  coupling::Exchange exchange;

  ocn::SamudraOcean::Config oc;
  oc.layout = ocn::samudra_e3smv3();
  oc.forcing_source = ocn::SamudraOcean::ForcingSource::Atmosphere;
  oc.exchange = &exchange;
  ocn::SamudraOcean ocean(oc, MPI_COMM_WORLD, g, decomp,
                          backend_on_root(rank, kOcnModel, ""));
  const auto ocn_ic = grid::read_grid_fields(
      kOcnIc, ocean.initial_condition_names(), g.ny, g.nx);

  atm::AceAtmosphere::Config ac;
  ac.layout = atm::samudrace_e3smv3();
  ac.surface.layer = atm::SurfaceLayer::NearSurface;
  ac.orbit = atm::Orbit::from_elements(0.016715, 23.4441, 102.7);
  ac.exchange = &exchange;
  ac.publish_ocean_forcing = true;
  ac.surface_from_ocean = true;
  // Stochastic: reseeded from (seed, step) at every network step.
  atm::AceAtmosphere atmosphere(ac, MPI_COMM_WORLD, g, decomp,
                                backend_on_root(rank, kAtmModel, "2026"));
  const auto atm_ic =
      grid::read_grid_fields(kAtmIc, ac.layout.inputs, g.ny, g.nx);

  // The coupler's fractions for the atmosphere: the IC's, made a partition.
  fields::FieldSet atm_imports(n);
  {
    auto get = [&](const char *name) {
      for (const auto &f : atm_ic) {
        if (f.name == name) {
          auto v = decomp.local(f.values);
          for (auto &x : v) {
            x = std::isfinite(x) ? std::clamp(x, 0.0, 1.0e30) : 0.0;
          }
          return v;
        }
      }
      throw std::runtime_error(name);
    };
    auto l = get("LANDFRAC"), o = get("OCNFRAC"), i = get("ICEFRAC");
    const auto ts = get("TS");
    auto lf = atm_imports.add("Sf_lfrac"), of = atm_imports.add("Sf_ofrac"),
         ifr = atm_imports.add("Sf_ifrac"), sx = atm_imports.add("Sx_t");
    for (std::size_t k = 0; k < n; ++k) {
      double a = std::min(l[k], 1.0), b = std::min(o[k], 1.0),
             c = std::min(i[k], 1.0);
      const double sum = a + b + c;
      if (sum > 1.0) {
        a /= sum;
        b /= sum;
        c /= sum;
      }
      lf[k] = a;
      of[k] = b;
      ifr[k] = c;
      sx[k] = (a + b + c) * ts[k];
    }
  }
  fields::FieldSet ocn_imports(n); // the atmosphere source reads the exchange
  fields::FieldSet atm_exports(n), ocn_exports(n);
  for (const auto &name : atm::ace_export_names()) {
    atm_exports.add(name);
  }
  for (const auto &name : ocn::samudra_export_names()) {
    ocn_exports.add(name);
  }

  // The ocean starts first, so its SST and ice are there for the
  // atmosphere; the atmosphere's first fluxes are then there for the ocean.
  ocean.initialize(after(0), ocn_ic);
  ocean.initial_exports(ocn_exports);
  atmosphere.initialize(after(0), atm_ic);
  atmosphere.initial_exports(after(0), atm_exports);
  REQUIRE(exchange.publishes("ocn.sst") == 1);
  REQUIRE(exchange.publishes("atm.LHFLX") == 1);

  const auto ocean_mask = ocean.ocean_mask();
  std::vector<double> ocean_area(n);
  for (std::size_t k = 0; k < n; ++k) {
    ocean_area[k] = area[k] * ocean_mask[k];
  }
  const double sst0 = mean_where(ocean.brackets().lower("sst"), ocean_area);
  if (rank == 0) {
    std::printf("  day  0: ocean SST %.2f K (initial condition, area-weighted "
                "over the ocean mask)\n", sst0);
  }
  for (int step = 1; step <= 480; ++step) { // 10 days: 2 ocean, 40 atm steps
    ocean.run(after(step), ocn_imports, ocn_exports);
    atmosphere.run(after(step), atm_imports, atm_exports);
    if (step % 240 == 0) {
      // The state the ocean has just predicted for the end of the next window.
      const double sst = mean_where(ocean.brackets().upper("sst"), ocean_area);
      const double lh = mean_where(exchange.get("atm.LHFLX"), ocean_area);
      const double fsds = mean_where(exchange.get("atm.FSDS"), ocean_area);
      const double tbot = mean_where(atm_exports.get("Sa_tbot"), area);
      const double ice = mean_where(ocean.sea_ice_fraction(), ocean_area);
      if (rank == 0) {
        std::printf("  day %2d: ocean SST %.2f K, ice %.3f; over ocean the "
                    "atmosphere's LHFLX %.1f, FSDS %.1f W/m2; Tat2m %.2f K "
                    "(%d ranks)\n",
                    step / 48, sst, ice, lh, fsds, tbot, size);
      }
      // Plausibility over ten days, not skill: no runaway from the start.
      REQUIRE(std::abs(sst - sst0) < 1.5);
      REQUIRE(tbot > 270.0);
      REQUIRE(tbot < 295.0);
      REQUIRE(lh > 40.0);
      REQUIRE(lh < 160.0);
    }
  }
  REQUIRE(ocean.clock().completed_steps() == 2);
  REQUIRE(atmosphere.clock().completed_steps() == 40);
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
