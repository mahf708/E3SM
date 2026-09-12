/**
 * @file ice.cpp
 * @brief Implementation of EmulatorIce.
 */

#include "ice.hpp"

#include <mpi.h>

#include <map>
#include <stdexcept>

namespace emulator {

EmulatorIce::EmulatorIce(coupling::Exchange &exchange)
    : Emulator(EmulatorType::ICE_COMP, -1, "emulatorice"),
      m_exchange(exchange) {}

void EmulatorIce::create_instance(int comm, int comp_id,
                                  const std::string &input_file,
                                  const std::string &log_file, int run_type,
                                  int start_ymd, int start_tod) {
  (void)input_file;
  (void)log_file;
  (void)run_type;
  m_id = comp_id;
  set_start_time({start_ymd, start_tod});

  // Collective, so a rank without the ocean fails with the others instead
  // of leaving them waiting.
  MPI_Comm c_comm = MPI_Comm_f2c(comm);
  const bool here = coupling::has_domain(m_exchange, "ocn");
  coupling::SharedDomain shared;
  long local[2] = {here ? 1 : 0, 0};
  if (here) {
    shared = coupling::shared_domain(m_exchange, "ocn");
    local[1] = static_cast<long>(shared.domain.size());
  }
  long global[2] = {0, 0};
  long least = 0;
  MPI_Allreduce(&local[0], &least, 1, MPI_LONG, MPI_MIN, c_comm);
  MPI_Allreduce(&local[1], &global[1], 1, MPI_LONG, MPI_SUM, c_comm);
  if (least == 0) {
    throw std::runtime_error(
        "emulatorice: no emulated ocean domain on at least one of this "
        "component's ranks. The sea ice is the emulated ocean's: it needs "
        "emulatorocn in this run, created before the ice, on the same ranks "
        "(NTASKS_ICE = NTASKS_OCN, ROOTPE_ICE = ROOTPE_OCN).");
  }
  if (static_cast<std::size_t>(global[1]) != shared.num_global) {
    throw std::runtime_error(
        "emulatorice: this component's ranks hold " +
        std::to_string(global[1]) + " of the ocean's " +
        std::to_string(shared.num_global) +
        " cells. It must run on exactly the ocean's ranks.");
  }
  set_domain(std::move(shared.domain), shared.nx, shared.ny,
             shared.num_global);
}

EmulatorIce::CouplingFields EmulatorIce::coupling_fields() const {
  using fields::Need;
  const std::map<std::string, std::string> units{
      {"Sa_z", "m"},          {"Sa_u", "m/s"},        {"Sa_v", "m/s"},
      {"Sa_ptem", "K"},       {"Sa_shum", "kg/kg"},   {"Sa_dens", "kg/m3"},
      {"Sa_tbot", "K"},       {"Faxa_swvdr", "W/m2"}, {"Faxa_swndr", "W/m2"},
      {"Faxa_swvdf", "W/m2"}, {"Faxa_swndf", "W/m2"}, {"Si_ifrac", "1"},
      {"Si_t", "K"},          {"Si_tref", "K"},       {"Si_qref", "kg/kg"},
      {"Si_snowh", "m"},      {"Si_avsdr", "1"},      {"Si_anidr", "1"},
      {"Si_avsdf", "1"},      {"Si_anidf", "1"},      {"Faii_swnet", "W/m2"},
      {"Faii_sen", "W/m2"},   {"Faii_lat", "W/m2"},   {"Faii_lwup", "W/m2"},
      {"Faii_evap", "kg/m2/s"}, {"Faii_taux", "N/m2"}, {"Faii_tauy", "N/m2"},
      {"Fioi_melth", "W/m2"}, {"Fioi_meltw", "kg/m2/s"},
      {"Fioi_salt", "kg/m2/s"}, {"Fioi_swpen", "W/m2"},
      {"Fioi_taux", "N/m2"},  {"Fioi_tauy", "N/m2"}};
  CouplingFields f;
  for (const auto &name : ice::sea_ice_import_names()) {
    f.imports.push_back({name, Need::Required, units.at(name)});
  }
  for (const auto &name : ice::sea_ice_export_names()) {
    f.exports.push_back({name,
                         name == "Si_snowh" ? Need::Optional : Need::Required,
                         units.at(name)});
  }
  return f;
}

void EmulatorIce::init_impl() {
  if (start_time().ymd < 0) {
    throw std::invalid_argument("emulatorice: no start time from the driver.");
  }
  // The ocean exported its initial state before the ice was created, so the
  // coupler's first fraction pass sees its ice, not an open polar ocean.
  export_at(start_time());
}

void EmulatorIce::run_impl(int dt) {
  (void)dt;
  const auto now = current_time();
  if (now.ymd < 0) {
    throw std::logic_error(
        "emulatorice: run without a model time. The ice skin temperature is "
        "seasonal and needs the driver's time each step (emulator_run_at).");
  }
  export_at(now);
}

void EmulatorIce::export_at(coupling::ModelTime now) {
  if (!is_coupled()) {
    return; // nothing to report to
  }
  const auto &d = domain();
  m_counts = ice::compute_sea_ice_exports(
      now, {d.lat, d.mask, m_exchange.get("ocn.sea_ice_fraction")}, imports(),
      mutable_exports());
}

} // namespace emulator
