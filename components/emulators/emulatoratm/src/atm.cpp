/**
 * @file atm.cpp
 * @brief Atmosphere emulator component implementation.
 *
 * Stub implementation — fill in details for AI/ML inference,
 * coupling, and I/O.
 */

#include "atm.hpp"
#include "emulator_c_api.hpp"
#include "scrip_reader.hpp"
#include "ace_atmosphere.hpp"
#include "ace_channels.hpp"
#include "create_inference_backend.hpp"
#include "grid_field_reader.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <map>
#include <utility>
#include <sstream>
#include <stdexcept>
#include <mpi.h>

namespace emulator {

EmulatorAtm::EmulatorAtm()
    : Emulator(EmulatorType::ATM_COMP, -1, "emulatoratm") {}

void EmulatorAtm::create_instance(int comm, int comp_id,
                                  const std::string &input_file,
                                  const std::string &log_file,
                                  int run_type, int start_ymd,
                                  int start_tod) {
  m_comm = comm;
  m_id = comp_id;  // set base class ID
  m_input_file = input_file;
  m_log_file = log_file;
  m_run_type = run_type;
  set_start_time({start_ymd, start_tod});

  auto &config = m_settings;
  if (!input_file.empty()) {
    std::ifstream ifs(input_file);
    std::string line;
    while (std::getline(ifs, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      size_t pos = line.find(':');
      if (pos != std::string::npos) {
        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);
        // trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t") + 1);
        config[key] = val;
      }
    }
  }

  const auto grid_file = config.find("grid_file");
  if (grid_file == config.end()) {
    if (config.count("nx") || config.count("ny")) {
      throw std::invalid_argument(
          "emulatoratm: " + input_file +
          " sets nx/ny but no grid_file. Grid dimensions without "
          "coordinates put every column at latitude 0, which the coupler's "
          "domain check rejects; name the SCRIP file with `grid_file:`.");
    }
    return; // a caller will provide the grid with set_grid_data()
  }

  const auto g = grid::read_scrip(grid_file->second);
  int rank = 0;
  int size = 1;
  MPI_Comm c_comm = MPI_Comm_f2c(m_comm);
  MPI_Comm_rank(c_comm, &rank);
  MPI_Comm_size(c_comm, &size);
  m_decomp = grid::Decomposition::contiguous_blocks(g.size(), size, rank);
  m_grid = g;
  set_domain(grid::Domain::full(g, m_decomp), g.nx, g.ny, g.size());
}

std::string EmulatorAtm::setting(const std::string &key,
                                 const std::string &fallback) const {
  const auto it = m_settings.find(key);
  return it == m_settings.end() ? fallback : it->second;
}

EmulatorAtm::CouplingFields EmulatorAtm::coupling_fields() const {
  if (setting("emulator", "").empty()) {
    return {};
  }
  using fields::Need;
  CouplingFields f;
  using Named = std::pair<const char *, const char *>;
  for (auto [name, units] : std::initializer_list<Named>{
           {"Sf_lfrac", "1"}, {"Sf_ofrac", "1"}, {"Sf_ifrac", "1"},
           {"Sx_t", "K"}}) {
    f.imports.push_back({name, Need::Required, units});
  }
  const std::map<std::string, std::string> units{
      {"Sa_z", "m"},          {"Sa_u", "m/s"},        {"Sa_v", "m/s"},
      {"Sa_tbot", "K"},       {"Sa_ptem", "K"},       {"Sa_shum", "kg/kg"},
      {"Sa_pbot", "Pa"},      {"Sa_pslv", "Pa"},      {"Sa_dens", "kg/m3"},
      {"Sa_topo", "m"},       {"Faxa_lwdn", "W/m2"},  {"Faxa_rainc", "kg/m2/s"},
      {"Faxa_rainl", "kg/m2/s"}, {"Faxa_snowc", "kg/m2/s"},
      {"Faxa_snowl", "kg/m2/s"}, {"Faxa_swndr", "W/m2"}, {"Faxa_swvdr", "W/m2"},
      {"Faxa_swndf", "W/m2"}, {"Faxa_swvdf", "W/m2"}, {"Faxa_swnet", "W/m2"}};
  for (const auto &name : atm::ace_export_names()) {
    f.exports.push_back({name, Need::Required, units.at(name)});
  }
  return f;
}

// =========================================================================
// Lifecycle implementations
// =========================================================================

void EmulatorAtm::init_impl() {
  const std::string emulator_name = setting("emulator", "");
  if (emulator_name.empty()) {
    return; // no model configured: exchange nothing, run nothing
  }
  if (!has_domain() || m_grid.size() == 0) {
    throw std::invalid_argument(
        "emulatoratm: `emulator` is set but no `grid_file`; the network runs "
        "on the whole grid and needs to read it.");
  }
  if (start_time().ymd < 0) {
    throw std::invalid_argument("emulatoratm: no start time from the driver.");
  }

  MPI_Comm comm = MPI_Comm_f2c(m_comm);
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  atm::AceAtmosphere::Config config;
  config.layout = atm::ace_layout(emulator_name);
  config.coupler_dt = std::stoi(setting("coupler_dt", "1800"));
  const bool has_near = std::find(config.layout.outputs.begin(),
                                  config.layout.outputs.end(),
                                  "Tat2m") != config.layout.outputs.end();
  const std::string layer =
      setting("surface_layer", has_near ? "near_surface" : "lowest_level");
  if (layer == "near_surface") {
    config.surface.layer = atm::SurfaceLayer::NearSurface;
  } else if (layer == "lowest_level") {
    config.surface.layer = atm::SurfaceLayer::LowestLevel;
  } else {
    throw std::invalid_argument("emulatoratm: surface_layer '" + layer +
                                "' is neither near_surface nor lowest_level.");
  }
  config.orbit = atm::Orbit::from_elements(
      std::stod(setting("orbit_eccen", "0.016715")),
      std::stod(setting("orbit_obliq", "23.4441")),
      std::stod(setting("orbit_mvelp", "102.7")));

  std::shared_ptr<inference::InferenceBackend> backend;
  if (rank == 0) {
    inference::InferenceConfig ic;
    ic.backend = setting("backend", "libtorch");
    ic.model_path = setting("model_path", "");
    for (const char *key : {"device", "dtype", "jit_optimize", "seed"}) {
      const auto v = setting(key, "");
      if (!v.empty()) {
        ic.set(key, v);
      }
    }
    if (ic.get("device").empty()) {
      ic.set("device", "cuda");
    }
    backend = inference::create_backend(ic, inference::InferenceContext{});
  }

  const auto initial = grid::read_grid_fields(setting("ic_file", ""),
                                              config.layout.inputs, m_grid.ny,
                                              m_grid.nx);
  m_ace = std::make_shared<atm::AceAtmosphere>(std::move(config), comm, m_grid,
                                               m_decomp, backend);
  m_ace->initialize(start_time(), initial);
  m_ace->initial_exports(start_time(), mutable_exports());
}

void EmulatorAtm::run_impl(int dt) {
  if (!m_ace) {
    return;
  }
  const auto now = current_time();
  if (now.ymd < 0) {
    throw std::logic_error(
        "emulatoratm: run without a model time. The emulated atmosphere "
        "needs the driver's time each step (emulator_run_at).");
  }
  if (dt != m_ace->clock().coupler_dt()) {
    throw std::invalid_argument(
        "emulatoratm: the driver's step is " + std::to_string(dt) +
        " s but coupler_dt is " +
        std::to_string(m_ace->clock().coupler_dt()) + " s.");
  }
  m_ace->run(now, imports(), mutable_exports());
}

void EmulatorAtm::final_impl() { m_ace.reset(); }

} // namespace emulator
