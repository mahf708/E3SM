// Catch2 v2 single header, with our own main so MPI brackets the run
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "ace_operators.hpp"
#include "emulator_component.hpp"
#include "emulator_test_support.hpp"
#include "grid_field_reader.hpp"
#include "ocean_operators.hpp"
#include "restart_file.hpp"
#include "scrip_reader.hpp"
#include "sea_ice_operators.hpp"

#ifdef EMULATOR_ENABLE_LIBTORCH
#include <torch/cuda.h>
#endif

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <memory>

namespace emulator {
namespace test {

namespace {

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

coupling::ModelTime after(int n) {
  const int s = n * 1800;
  return {20000101 + s / 86400, s % 86400};
}

bool available() {
  int ok = 1;
#ifndef EMULATOR_ENABLE_LIBTORCH
  ok = 0;
#else
  ok = grid::have_scrip_reader() &&
       std::filesystem::exists(kSamudraceAtmModel) &&
       std::filesystem::exists(kSamudraOcnModel) && torch::cuda::is_available();
#endif
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (ok) {
    atm::register_atm_operators();
    ocn::register_ocn_operators();
    ice::register_ice_operators();
  }
  return ok != 0;
}

/**
 * The three components in the driver's order, with the atmosphere spec
 * given, on their own exchange (a process of their own).  The coupler's
 * fractions for the atmosphere are the initial condition's, made a
 * partition.  Given a restart prefix, each component restores from
 * `<prefix>.<component>.nc` and starts at step `start`.
 */
class Trio {
public:
  Trio(const std::string &atm_spec, int start = 0,
       const std::string &restart_prefix = "")
      : m_spec(atm_spec), m_step(start),
        m_atm_in("atm_in",
                 "spec: " + spec_path(atm_spec) + "\n"
                 "coupler_dt: 1800\n"
                 "grid: {file: " + kGaussianGrid + ", domain: full}\n"
                 "initial_condition: " + kSamudraceAtmIc + "\n"
                 "inference: {backend: libtorch, model_path: " +
                     kSamudraceAtmModel + ", device: cuda, seed: 2026}\n"),
        m_ocn_in("ocn_in",
                 "spec: " + spec_path("samudra-e3smv3-ocean.yaml") + "\n"
                 "coupler_dt: 1800\n"
                 "grid: {file: " + kGaussianGrid +
                     ", domain: ocean_mask, mask_variable: mask_2d, "
                     "publish_as: ocn}\n"
                 "initial_condition: " + kSamudraOcnIc + "\n"
                 "inference: {backend: libtorch, model_path: " +
                     kSamudraOcnModel + ", device: cuda}\n"),
        m_ice_in("ice_in",
                 "spec: " + spec_path("samudrace-e3smv3-sea-ice.yaml") + "\n"
                 "coupler_dt: 1800\n"
                 "grid: {domain: shared, shared_from: ocn}\n") {
    const int fcomm = MPI_Comm_c2f(MPI_COMM_WORLD);
    const int run_type = restart_prefix.empty() ? 0 : 1;
    const auto t = after(start);
    auto create = [&](std::unique_ptr<EmulatorComponent> &c, EmulatorType type,
                      const char *name, int id, const TempFile &in) {
      c = std::make_unique<EmulatorComponent>(type, name, m_exchange);
      c->create_instance(fcomm, id, in.path, "", run_type, t.ymd, t.tod);
      if (!restart_prefix.empty()) {
        c->set_restart_file(restart_prefix + "." + name + ".nc");
      }
    };

    create(atm, EmulatorType::ATM_COMP, "emulatoratm", 1, m_atm_in);
    n = static_cast<std::size_t>(atm->get_num_local_cols());
    x2a = std::make_unique<AttrVect>(kX2a, n);
    a2x = std::make_unique<AttrVect>(kA2x, n);
    x2o = std::make_unique<AttrVect>(kX2o, n);
    o2x = std::make_unique<AttrVect>(kO2x, n);
    x2i = std::make_unique<AttrVect>(kX2i, n);
    i2x = std::make_unique<AttrVect>(kI2x, n);
    set_fractions();
    atm->set_coupler_field_lists(kX2a, kA2x);
    atm->setup_coupling(coupling(*x2a, *a2x, n));
    atm->initialize();

    create(ocn, EmulatorType::OCN_COMP, "emulatorocn", 4, m_ocn_in);
    ocn->set_coupler_field_lists(kX2o, kO2x);
    ocn->setup_coupling(coupling(*x2o, *o2x, n));
    ocn->initialize();

    create(ice, EmulatorType::ICE_COMP, "emulatorice", 5, m_ice_in);
    ice->set_coupler_field_lists(kX2i, kI2x);
    ice->setup_coupling(coupling(*x2i, *i2x, n));
    ice->initialize();
  }

  ~Trio() {
    atm->finalize();
    ocn->finalize();
    ice->finalize();
  }

  /// Coupler steps up to and including `last`.
  void run_to(int last) {
    for (++m_step; m_step <= last; ++m_step) {
      // The same grid for all three: the coupler's a2x -> x2i is a copy.
      for (std::size_t p = 0; p < n; ++p) {
        for (const auto &name : x2i->names) {
          x2i->at(name, p) = a2x->at(name, p);
        }
      }
      ice->run(1800, after(m_step));
      ocn->run(1800, after(m_step));
      atm->run(1800, after(m_step));
    }
    --m_step;
  }

  void write_restarts(const std::string &prefix) const {
    atm->write_restart(prefix + ".emulatoratm.nc");
    ocn->write_restart(prefix + ".emulatorocn.nc");
    ice->write_restart(prefix + ".emulatorice.nc");
  }

  /// What the coupler sees and the ocean's state, in one vector.  The sea
  /// ice's exports are made from the coupler's imports, which a restarted
  /// component has not been given at initialization (CIME's coupler restores
  /// its own copy), so `with_ice` is false there.
  std::vector<double> snapshot(bool with_ice = true) const {
    std::vector<double> v;
    for (const auto *av : {a2x.get(), o2x.get()}) {
      v.insert(v.end(), av->data.begin(), av->data.end());
    }
    if (with_ice) {
      v.insert(v.end(), i2x->data.begin(), i2x->data.end());
    }
    const auto sst = ocn->model()->brackets().upper("sst");
    v.insert(v.end(), sst.begin(), sst.end());
    return v;
  }

  std::unique_ptr<EmulatorComponent> atm, ocn, ice;
  std::unique_ptr<AttrVect> x2a, a2x, x2o, o2x, x2i, i2x;
  std::size_t n = 0;

private:
  void set_fractions() {
    const auto ic = grid::read_grid_fields(
        kSamudraceAtmIc, {"LANDFRAC", "OCNFRAC", "ICEFRAC", "TS"}, 180, 360);
    std::vector<int> gids(n);
    atm->get_local_col_gids(gids.data());
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
      x2a->at("Sf_lfrac", p) = l;
      x2a->at("Sf_ofrac", p) = o;
      x2a->at("Sf_ifrac", p) = i;
      x2a->at("Sx_t", p) = (l + o + i) * get(3);
    }
  }

  std::string m_spec;
  int m_step;
  coupling::Exchange m_exchange;
  TempFile m_atm_in, m_ocn_in, m_ice_in;
};

/// Five days (one ocean step); returns the ocean's predicted SST,
/// area-weighted over its mask.
double run_trio(const std::string &atm_spec) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Trio trio(atm_spec);
  trio.run_to(240);
  const auto n = trio.n;
  auto &i2x = *trio.i2x;

  std::vector<double> area(n), mask(n), frac(n);
  trio.ocn->get_cols_area(area.data());
  trio.ocn->get_cols_mask_frac(mask.data(), frac.data());
  double ocean_area = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    ocean_area += area[p] * mask[p];
  }
  ocean_area = global_sum(ocean_area);

  const auto *ocean = trio.ocn->model();
  double sst = 0.0, ifrac = 0.0, sen_ice = 0.0, ice_area = 0.0, fluxes = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    sst += area[p] * mask[p] * ocean->brackets().upper("sst")[p];
    ifrac += area[p] * i2x.at("Si_ifrac", p);
    sen_ice += area[p] * i2x.at("Si_ifrac", p) * i2x.at("Faii_sen", p);
    ice_area += area[p] * i2x.at("Si_ifrac", p);
    fluxes += i2x.at("Faii_lwup", p) < 0.0 ? 1.0 : 0.0;
  }
  sst = global_sum(sst) / ocean_area;
  sen_ice = global_sum(sen_ice) / global_sum(ice_area);
  ifrac = global_sum(ifrac) / ocean_area;
  fluxes = global_sum(fluxes);
  if (rank == 0) {
    std::printf("  %s, day 5: ocean SST %.6f K (predicted, area-weighted); "
                "ice fraction %.4f; sensible heat into the ice %.1f W/m2 "
                "(ice-weighted); %d ranks\n",
                atm_spec.c_str(), sst, ifrac, sen_ice, size);
  }
  REQUIRE(ocean->clock().completed_steps() == 1);
  REQUIRE(fluxes == 44892.0); // the bulk scheme reached every ocean cell
  REQUIRE(ifrac > 0.01);
  REQUIRE(ifrac < 0.2);
  REQUIRE(std::isfinite(sen_ice));
  return sst;
}

} // namespace

TEST_CASE("SamudrACE's atmosphere, ocean and sea ice run as three generic "
          "components configured by their specs and input files",
          "[samudrace][real]") {
  if (!available()) {
    WARN("skipped: needs libtorch, netCDF, a GPU and the SamudrACE files");
    return;
  }
  // The model-level SamudrACE test's day 5 (test_samudrace_coupled_real).
  // 291.104738 on the Greenwich-centred grid, whose insolation was half a
  // degree west of the data's; 291.104761 with the v1 ocean trace, which left
  // sea_surface_fraction unmasked on land (+0.08 K).
  REQUIRE(run_trio("samudrace-e3smv3-atmosphere.yaml") ==
          Approx(291.022110).epsilon(0).margin(5e-7));
}

TEST_CASE("The atmosphere with fme's ocean-to-atmosphere exchange runs the "
          "three components", "[samudrace][real]") {
  if (!available()) {
    WARN("skipped: needs libtorch, netCDF, a GPU and the SamudrACE files");
    return;
  }
  const double sst = run_trio("samudrace-e3smv3-atmosphere-fme-surface.yaml");
  REQUIRE(std::abs(sst - 291.02) < 1.0);
}

TEST_CASE("The three components restarted from files mid-window continue "
          "bit for bit", "[samudrace][real][restart]") {
  if (!available() || !coupling::have_restart_files()) {
    WARN("skipped: needs libtorch, netCDF, a GPU and the SamudrACE files");
    return;
  }
  // Step 125 is 5 coupler steps into an atmosphere interval and halfway
  // through the ocean's window; step 250 is past the ocean's first step, so
  // the window mean carried across the restart reaches its prediction.  The
  // fme exchange carries the most operator state (the blended TS and what
  // it last saw of the ocean).
  const std::string spec = "samudrace-e3smv3-atmosphere-fme-surface.yaml";
  const int stop = 125, end = 250;
  int pid = static_cast<int>(::getpid());
  MPI_Bcast(&pid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  const std::string prefix = "trio_restart_" + std::to_string(pid);

  std::vector<double> continuous, at_stop, restarted, restarted_at_stop;
  {
    Trio trio(spec);
    trio.run_to(stop);
    at_stop = trio.snapshot(false);
    trio.write_restarts(prefix);
    trio.run_to(end);
    continuous = trio.snapshot();
  }
  {
    Trio trio(spec, stop, prefix);
    // What the coupler receives from the restarted components' initial
    // exports is what the continuous run exported at that step.
    restarted_at_stop = trio.snapshot(false);
    trio.run_to(end);
    restarted = trio.snapshot();
    REQUIRE(trio.ocn->model()->clock().completed_steps() == 1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    for (const char *c : {"emulatoratm", "emulatorocn", "emulatorice"}) {
      std::remove((prefix + "." + c + ".nc").c_str());
    }
  }
  std::size_t differ_at_stop = 0, differ = 0;
  for (std::size_t k = 0; k < at_stop.size(); ++k) {
    differ_at_stop += at_stop[k] != restarted_at_stop[k];
  }
  for (std::size_t k = 0; k < continuous.size(); ++k) {
    differ += continuous[k] != restarted[k];
  }
  UNSCOPED_INFO("values differing at the restart " << differ_at_stop
                << ", at the end " << differ << " of " << continuous.size());
  REQUIRE(restarted.size() == continuous.size());
  REQUIRE(differ_at_stop == 0);
  REQUIRE(differ == 0);
}

} // namespace test
} // namespace emulator

EMULATOR_TEST_MPI_MAIN
