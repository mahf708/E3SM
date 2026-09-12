/**
 * @file ocn.cpp
 * @brief Implementation of EmulatorOcn.
 */

#include "ocn.hpp"

#include "create_inference_backend.hpp"
#include "grid_field_reader.hpp"
#include "ocean_forcing.hpp"
#include "samudra_channels.hpp"
#include "samudra_ocean.hpp"
#include "scrip_reader.hpp"

#include <mpi.h>

#include <map>
#include <stdexcept>

namespace emulator {

EmulatorOcn::EmulatorOcn(coupling::Exchange &exchange)
    : Emulator(EmulatorType::OCN_COMP, -1, "emulatorocn"),
      m_exchange(exchange) {}

EmulatorOcn::~EmulatorOcn() = default;

void EmulatorOcn::create_instance(int comm, int comp_id,
                                  const std::string &input_file,
                                  const std::string &log_file, int run_type,
                                  int start_ymd, int start_tod) {
  (void)log_file;
  (void)run_type;
  m_comm = comm;
  m_id = comp_id;
  set_start_time({start_ymd, start_tod});
  m_settings = ComponentSettings::read(input_file);

  if (!m_settings.has("grid_file")) {
    return; // a caller will provide the grid with set_grid_data()
  }
  if (!m_settings.has("ic_file")) {
    throw std::invalid_argument(
        "emulatorocn: " + input_file +
        " names a grid_file but no ic_file. The ocean's domain mask is the "
        "initial condition's mask_2d; a grid alone would put ocean on land.");
  }
  MPI_Comm c_comm = MPI_Comm_f2c(m_comm);
  int rank = 0, size = 1;
  MPI_Comm_rank(c_comm, &rank);
  MPI_Comm_size(c_comm, &size);
  m_grid = grid::read_scrip(m_settings.get("grid_file", ""));
  m_decomp = grid::Decomposition::contiguous_blocks(m_grid.size(), size, rank);
  const auto mask = grid::read_grid_fields(m_settings.get("ic_file", ""),
                                           {"mask_2d"}, m_grid.ny, m_grid.nx);
  auto domain = grid::Domain::masked(m_grid, m_decomp, mask.at(0).values);
  coupling::publish_domain(m_exchange, "ocn",
                           {domain, m_grid.nx, m_grid.ny, m_grid.size()});
  set_domain(std::move(domain), m_grid.nx, m_grid.ny, m_grid.size());
}

EmulatorOcn::CouplingFields EmulatorOcn::coupling_fields() const {
  if (m_settings.get("emulator", "").empty()) {
    return {};
  }
  using fields::Need;
  CouplingFields f;
  if (m_settings.get("forcing", "coupler") == "coupler") {
    const std::map<std::string, std::string> units{
        {"Foxx_taux", "N/m2"},    {"Foxx_tauy", "N/m2"},
        {"Faxa_rain", "kg/m2/s"}, {"Faxa_snow", "kg/m2/s"},
        {"Foxx_lwup", "W/m2"},    {"Faxa_lwdn", "W/m2"},
        {"Foxx_swnet", "W/m2"},   {"Foxx_lat", "W/m2"},
        {"Foxx_sen", "W/m2"},     {"Si_ifrac", "1"}};
    for (const auto &name : ocn::coupler_forcing_imports()) {
      f.imports.push_back({name, Need::Required, units.at(name)});
    }
  }
  const std::map<std::string, std::string> units{
      {"So_t", "K"},     {"So_s", "g/kg"},  {"So_u", "m/s"},
      {"So_v", "m/s"},   {"So_ssh", "m"},   {"So_dhdx", "m/m"},
      {"So_dhdy", "m/m"}};
  for (const auto &name : ocn::samudra_export_names()) {
    f.exports.push_back({name, Need::Required, units.at(name)});
  }
  return f;
}

void EmulatorOcn::init_impl() {
  const std::string emulator_name = m_settings.get("emulator", "");
  if (emulator_name.empty()) {
    return;
  }
  if (emulator_name != "Samudra-E3SMv3") {
    throw std::invalid_argument("emulatorocn: unknown emulator '" +
                                emulator_name + "'; known: Samudra-E3SMv3.");
  }
  if (!has_domain() || m_grid.size() == 0) {
    throw std::invalid_argument(
        "emulatorocn: `emulator` is set but no `grid_file`; the network runs "
        "on the whole grid and needs to read it.");
  }

  ocn::SamudraOcean::Config config;
  config.layout = ocn::samudra_e3smv3();
  config.coupler_dt = std::stoi(m_settings.get("coupler_dt", "1800"));
  config.exchange = &m_exchange;
  const auto forcing = m_settings.get("forcing", "coupler");
  if (forcing == "coupler") {
    config.forcing_source = ocn::SamudraOcean::ForcingSource::Coupler;
  } else if (forcing == "atmosphere") {
    config.forcing_source = ocn::SamudraOcean::ForcingSource::Atmosphere;
  } else {
    throw std::invalid_argument("emulatorocn: forcing '" + forcing +
                                "' is neither coupler nor atmosphere.");
  }

  MPI_Comm comm = MPI_Comm_f2c(m_comm);
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  std::shared_ptr<inference::InferenceBackend> backend;
  if (rank == 0) {
    inference::InferenceConfig ic;
    ic.backend = m_settings.get("backend", "libtorch");
    ic.model_path = m_settings.get("model_path", "");
    ic.set("device", "cuda");
    for (const char *key : {"device", "dtype", "jit_optimize", "seed"}) {
      if (m_settings.has(key)) {
        ic.set(key, m_settings.get(key, ""));
      }
    }
    backend = inference::create_backend(ic, inference::InferenceContext{});
  }

  m_ocean = std::make_unique<ocn::SamudraOcean>(std::move(config), comm,
                                                m_grid, m_decomp, backend);
  const auto initial = grid::read_grid_fields(
      m_settings.get("ic_file", ""), m_ocean->initial_condition_names(),
      m_grid.ny, m_grid.nx);
  m_ocean->initialize(start_time(), initial);
  m_ocean->initial_exports(mutable_exports());
}

void EmulatorOcn::run_impl(int dt) {
  if (!m_ocean) {
    return;
  }
  const auto now = current_time();
  if (now.ymd < 0) {
    throw std::logic_error(
        "emulatorocn: run without a model time. The emulated ocean needs the "
        "driver's time each step (emulator_run_at).");
  }
  if (dt != m_ocean->clock().coupler_dt()) {
    throw std::invalid_argument(
        "emulatorocn: the driver's step is " + std::to_string(dt) +
        " s but coupler_dt is " +
        std::to_string(m_ocean->clock().coupler_dt()) + " s.");
  }
  m_ocean->run(now, imports(), mutable_exports());
}

void EmulatorOcn::final_impl() { m_ocean.reset(); }

} // namespace emulator
