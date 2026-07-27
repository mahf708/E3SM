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
#include <map>
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

namespace {

/// Split an MCT field list ("Sa_z:Sa_u:Sa_v") into name -> row index.
std::map<std::string, int> parse_field_list(const std::string &fields) {
  std::map<std::string, int> index;
  std::istringstream iss(fields);
  std::string name;
  int position = 0;
  while (std::getline(iss, name, ':')) {
    // Field lists arrive straight from a Fortran string, so they can carry
    // trailing blanks.
    const auto first = name.find_first_not_of(" \t");
    if (first == std::string::npos) {
      continue;
    }
    const auto last = name.find_last_not_of(" \t");
    index.emplace(name.substr(first, last - first + 1), position);
    ++position;
  }
  return index;
}

} // namespace

void EmulatorAtm::init_coupling_indices(
    const std::string &export_fields,
    const std::string &import_fields) {
  m_import_idx = parse_field_list(import_fields);
  m_export_idx = parse_field_list(export_fields);
}

void EmulatorAtm::setup_coupling(const EmulatorCouplingDesc& cpl) {
  m_import_data = cpl.import_data;
  m_export_data = cpl.export_data;
  m_num_imports = cpl.num_imports;
  m_num_exports = cpl.num_exports;
  m_field_size = cpl.field_size;
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

  validate_coupling(config.get_bool("allow_unmatched_inputs", false));

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

/**
 * @brief Check the coupling descriptor before a single value moves.
 *
 * Everything here is a mismatch that would otherwise show up as a partly
 * filled field: plausible numbers over part of the globe and zeros or stale
 * values over the rest, which no downstream check would catch.  A component
 * that has not been given coupling buffers at all is a different matter — a
 * unit test, or a driver bringing the emulator up — and is left alone.
 *
 * @param allow_unmatched_inputs Permit a declared input that the coupler does
 *        not carry.  Off by default: unlike an output, an unmatched input has
 *        no other source, so allowing one is permission to run on zeros.
 */
void EmulatorAtm::validate_coupling(bool allow_unmatched_inputs) {
  const bool has_import = m_import_data != nullptr;
  const bool has_export = m_export_data != nullptr;

  m_input_src.assign(m_infer_inputs.size(), -1);
  m_output_dst.assign(m_infer_outputs.size(), -1);
  if (!has_import && !has_export) {
    if (!m_infer_inputs.empty() || !m_infer_outputs.empty()) {
      std::cout << "[emulatoratm] no coupling buffers; the model will run on "
                   "whatever is in its input buffers.\n";
    }
    return;
  }

  if (m_field_size != m_num_local_cols) {
    throw std::runtime_error(
        "[emulatoratm] the coupler's field size (" +
        std::to_string(m_field_size) + ") and this rank's column count (" +
        std::to_string(m_num_local_cols) +
        ") disagree. Truncating to the shorter of the two would leave part of "
        "every field unset, so this is fatal.");
  }
  if (m_num_imports < 0 || m_num_exports < 0 || m_field_size < 0) {
    throw std::runtime_error(
        "[emulatoratm] negative coupling extents: num_imports=" +
        std::to_string(m_num_imports) +
        " num_exports=" + std::to_string(m_num_exports) +
        " field_size=" + std::to_string(m_field_size) + ".");
  }
  // The field lists and the attribute vectors have to describe the same
  // thing; if they do not, every row index below is off by an unknown amount.
  // An *empty* list is a different matter — a driver may set up buffers
  // without naming their contents — and is left to the per-name resolution
  // below, which fails loudly for any input that then cannot be found.
  if (has_import && !m_import_idx.empty() &&
      static_cast<int>(m_import_idx.size()) != m_num_imports) {
    throw std::runtime_error(
        "[emulatoratm] the import field list names " +
        std::to_string(m_import_idx.size()) + " field(s) but x2a holds " +
        std::to_string(m_num_imports) + ".");
  }
  if (has_export && !m_export_idx.empty() &&
      static_cast<int>(m_export_idx.size()) != m_num_exports) {
    throw std::runtime_error(
        "[emulatoratm] the export field list names " +
        std::to_string(m_export_idx.size()) + " field(s) but a2x holds " +
        std::to_string(m_num_exports) + ".");
  }

  // Resolve each declared field against the coupler's lists once, here,
  // rather than looking names up every step.
  const auto resolve = [](const std::string &name,
                          const std::map<std::string, int> &index, int width) {
    const auto it = index.find(name);
    if (it == index.end()) {
      return -1;
    }
    if (it->second < 0 || it->second >= width) {
      throw std::runtime_error("[emulatoratm] coupling field '" + name +
                               "' resolves to row " +
                               std::to_string(it->second) +
                               ", outside the buffer's " +
                               std::to_string(width) + " row(s).");
    }
    return it->second;
  };

  std::vector<std::string> unmatched_inputs;
  for (std::size_t i = 0; i < m_infer_inputs.size(); ++i) {
    m_input_src[i] =
        has_import ? resolve(m_infer_inputs[i], m_import_idx, m_num_imports)
                   : -1;
    if (m_input_src[i] < 0) {
      unmatched_inputs.push_back(m_infer_inputs[i]);
    }
  }
  if (!unmatched_inputs.empty() && !allow_unmatched_inputs) {
    std::string names;
    for (const auto &name : unmatched_inputs) {
      names += (names.empty() ? "" : ", ") + name;
    }
    throw std::runtime_error(
        "[emulatoratm] the model declares input(s) the coupler does not "
        "carry: " +
        names +
        ". An unmatched input has no other source, so the model would run on "
        "zeros. Fix the name, check that init_coupling_indices was given the "
        "x2a field list (it named " +
        std::to_string(m_import_idx.size()) +
        " field(s)), or set `inference.allow_unmatched_inputs: true` if the "
        "field really is supplied some other way.");
  }

  // An unmatched *output* is different: a model may legitimately produce
  // diagnostics the coupler does not consume. Report it and carry on.
  for (std::size_t i = 0; i < m_infer_outputs.size(); ++i) {
    m_output_dst[i] =
        has_export ? resolve(m_infer_outputs[i], m_export_idx, m_num_exports)
                   : -1;
    if (m_output_dst[i] < 0) {
      std::cout << "[emulatoratm] inference output '" << m_infer_outputs[i]
                << "' is not a coupling field; it will not be sent to the "
                   "coupler.\n";
    }
  }
}

void EmulatorAtm::import_coupling_fields() {
  // The pack step reads x2a directly, so there is no separate copy into
  // internal field storage yet.  It gets one when the component grows state
  // of its own beyond what the model carries.
}

void EmulatorAtm::export_coupling_fields() {
  // Likewise: the unpack step writes a2x directly.
}

/**
 * @brief Gather the model's inputs out of the coupler's import buffer.
 *
 * `x2a%rAttr` is Fortran `(nflds, lsize)`, so it is contiguous in *fields*
 * and strided in columns: field `f` of column `c` lives at
 * `import_data[c * num_imports + f]`.  The model wants each field contiguous
 * in columns, which is the transpose, so this is a real gather rather than a
 * pointer.  It is also why `m_infer_in` exists at all.
 */
void EmulatorAtm::prepare_inputs() {
  if (m_import_data == nullptr) {
    return;
  }
  // validate_coupling() has already established that m_field_size and
  // m_num_local_cols agree, so one bound serves both.
  const int ncol = m_num_local_cols;
  for (std::size_t i = 0; i < m_infer_inputs.size(); ++i) {
    double *dst = m_infer_in.data() + i * m_num_local_cols;
    const int row = m_input_src[i];
    if (row < 0) {
      continue; // not a coupling field; leave whatever is there
    }
    for (int c = 0; c < ncol; ++c) {
      dst[c] = m_import_data[static_cast<std::size_t>(c) * m_num_imports + row];
    }
  }
}

/**
 * @brief Scatter the model's outputs into the coupler's export buffer.
 *
 * The inverse of prepare_inputs(), into `a2x%rAttr(nflds, lsize)`.
 */
void EmulatorAtm::process_outputs() {
  if (m_export_data == nullptr) {
    return;
  }
  const int ncol = m_num_local_cols;
  for (std::size_t i = 0; i < m_infer_outputs.size(); ++i) {
    const double *src = m_infer_out.data() + i * m_num_local_cols;
    const int row = m_output_dst[i];
    if (row < 0) {
      continue; // the model produced something the coupler does not want
    }
    for (int c = 0; c < ncol; ++c) {
      m_export_data[static_cast<std::size_t>(c) * m_num_exports + row] = src[c];
    }
  }
}

} // namespace emulator
