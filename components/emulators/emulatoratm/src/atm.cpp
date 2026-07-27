/**
 * @file atm.cpp
 * @brief Atmosphere emulator component implementation.
 *
 * Stub implementation — fill in details for AI/ML inference,
 * coupling, and I/O.
 */

#include "atm.hpp"
#include "emulator_c_api.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
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
  (void)start_ymd;
  (void)start_tod;

  // Simple configuration parsing
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

        if (key == "nx") {
          m_nx = std::stoi(val);
        }
        if (key == "ny") {
          m_ny = std::stoi(val);
        }
        if (key == "grid") {
          // grid name identified
        }
      }
    }
  }

  // Compute global column count from grid dimensions
  if (m_nx > 0) {
    m_num_global_cols = m_nx * std::max(1, m_ny);
  }

  // If we have a global size but no local size, create a default decomposition
  if (m_num_global_cols > 0 && m_num_local_cols == 0) {
    int rank, size;
    MPI_Comm_rank(MPI_Comm_f2c(m_comm), &rank);
    MPI_Comm_size(MPI_Comm_f2c(m_comm), &size);

    int n_per_rank = m_num_global_cols / size;
    int remainder = m_num_global_cols % size;

    int start_idx = rank * n_per_rank + std::min(rank, remainder);
    m_num_local_cols = n_per_rank + (rank < remainder ? 1 : 0);

    m_col_gids.resize(m_num_local_cols);
    for (int i = 0; i < m_num_local_cols; ++i) {
      m_col_gids[i] = start_idx + i + 1; // 1-based GIDs for MCT
    }

    m_lat.assign(m_num_local_cols, 0.0);
    m_lon.assign(m_num_local_cols, 0.0);
    m_area.assign(m_num_local_cols, 1.0);
  }
}

void EmulatorAtm::set_grid_data(const EmulatorGridDesc& grid) {
  m_nx = grid.nx;
  m_ny = grid.ny;
  m_num_local_cols = grid.num_local_cols;
  m_num_global_cols = grid.num_global_cols;

  m_col_gids.assign(grid.col_gids, grid.col_gids + grid.num_local_cols);
  m_lat.assign(grid.lat, grid.lat + grid.num_local_cols);
  m_lon.assign(grid.lon, grid.lon + grid.num_local_cols);
  m_area.assign(grid.area, grid.area + grid.num_local_cols);
}

void EmulatorAtm::init_coupling_indices(
    const std::string &export_fields,
    const std::string &import_fields) {
  // TODO: Parse colon-separated MCT field lists and populate
  // m_coupling_idx with index positions.
  (void)export_fields;
  (void)import_fields;
}

void EmulatorAtm::setup_coupling(const EmulatorCouplingDesc& cpl) {
  m_import_data = cpl.import_data;
  m_export_data = cpl.export_data;
  m_num_imports = cpl.num_imports;
  m_num_exports = cpl.num_exports;
  (void)cpl.field_size;
}

void EmulatorAtm::get_local_col_gids(int *gids) const {
  std::memcpy(gids, m_col_gids.data(),
              m_col_gids.size() * sizeof(int));
}

void EmulatorAtm::get_cols_latlon(double *lat, double *lon) const {
  std::memcpy(lat, m_lat.data(),
              m_lat.size() * sizeof(double));
  std::memcpy(lon, m_lon.data(),
              m_lon.size() * sizeof(double));
}

void EmulatorAtm::get_cols_area(double *area) const {
  std::memcpy(area, m_area.data(),
              m_area.size() * sizeof(double));
}

// =========================================================================
// Lifecycle implementations
// =========================================================================

void EmulatorAtm::init_impl() {
  // Inference settings live in atm_in alongside everything else, prefixed so
  // they cannot collide with the component's own keys.
  auto config =
      inference::InferenceConfig::from_file(m_input_file, "inference.");

  // Hand the model the resources the coupler gave us: the *component*
  // communicator, and the columns this rank owns.  Both matter — the first
  // so a distributed model builds a process group over our ranks rather than
  // over the whole coupled job, the second so it can work out how our
  // decomposition relates to its own grid.
  auto context = inference::make_context(m_comm);
  context.set_grid(m_nx, m_ny, m_num_global_cols, m_col_gids.data(),
                   m_lat.data(), m_lon.data(), m_num_local_cols);

  m_infer_inputs = config.inputs;
  m_infer_outputs = config.outputs;
  m_inference = inference::create_backend(config, context);

  m_infer_in.assign(
      static_cast<std::size_t>(m_num_local_cols) * m_infer_inputs.size(), 0.0);
  m_infer_out.assign(
      static_cast<std::size_t>(m_num_local_cols) * m_infer_outputs.size(), 0.0);

  // TODO: Read initial conditions
  // TODO: Set up diagnostic output manager
  // TODO: Export initial values to coupler
}

void EmulatorAtm::run_impl(int dt) {
  (void)dt;

  // 1. Import fields from coupler
  import_coupling_fields();

  // 2. Prepare AI model inputs
  prepare_inputs();

  // 3. Run AI inference
  run_inference();

  // 4. Process AI outputs
  process_outputs();

  // 5. TODO: Diagnostic output

  // 6. Export fields to coupler
  export_coupling_fields();
}

void EmulatorAtm::final_impl() {
  // TODO: Write final restart files
  // TODO: Finalize output manager

  if (m_inference) {
    m_inference->finalize();
    m_inference.reset();
  }

  // TODO: Deallocate field storage
  std::cout << "emulatoratm c++ side ... bye!" << std::endl;
}

/**
 * @brief Evaluate the model on this rank's columns.
 *
 * Every field is wrapped as a `[ncol]` view of the packed buffers — nothing
 * is copied on the way in or out.  For a distributed model this call is
 * collective over the component communicator, which is why it sits on the
 * unconditional path of run_impl() rather than behind a rank test.
 */
void EmulatorAtm::run_inference() {
  if (!m_inference) {
    return;
  }

  const auto ncol = static_cast<std::int64_t>(m_num_local_cols);

  inference::TensorMap inputs;
  for (std::size_t i = 0; i < m_infer_inputs.size(); ++i) {
    inputs.wrap(m_infer_inputs[i],
                static_cast<const double *>(m_infer_in.data()) + i * ncol,
                {ncol});
  }
  inference::TensorMap outputs;
  for (std::size_t i = 0; i < m_infer_outputs.size(); ++i) {
    outputs.wrap(m_infer_outputs[i], m_infer_out.data() + i * ncol, {ncol});
  }

  m_inference->infer(inputs, outputs);
}

// =========================================================================
// Coupling helpers
// =========================================================================

void EmulatorAtm::import_coupling_fields() {
  // TODO: Transfer coupler import data → internal fields
}

void EmulatorAtm::export_coupling_fields() {
  // TODO: Transfer internal fields → coupler export data
}

void EmulatorAtm::prepare_inputs() {
  // TODO: Pack the imported coupling fields into m_infer_in, one contiguous
  // [ncol] block per name in m_infer_inputs. Needs init_coupling_indices()
  // to have resolved the MCT field lists first.
}

void EmulatorAtm::process_outputs() {
  // TODO: Unpack m_infer_out into the export coupling fields, one contiguous
  // [ncol] block per name in m_infer_outputs.
}

} // namespace emulator
