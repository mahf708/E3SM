/**
 * @file emulator_output_stream.cpp
 * @brief Implementation of EmulatorOutputStream.
 */

#include "emulator_output_stream.hpp"
#include "emulator_io.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace emulator {

// ============================================================================
// OutputControl implementation
// ============================================================================

bool OutputControl::is_write_step(int step) const {
  if (!output_enabled()) {
    return false;
  }

  return step >= next_write_step && step > 0;
}

double OutputControl::seconds_per_unit() const {
  switch (frequency_unit) {
  case FrequencyUnit::NSTEPS:
    return dt; // Will be multiplied by frequency
  case FrequencyUnit::NSECS:
    return 1.0;
  case FrequencyUnit::NMINS:
    return 60.0;
  case FrequencyUnit::NHOURS:
    return 3600.0;
  case FrequencyUnit::NDAYS:
    return 86400.0;
  case FrequencyUnit::NMONTHS:
    return 30.0 * 86400.0; // Approximate
  case FrequencyUnit::NYEARS:
    return 365.0 * 86400.0; // Approximate
  case FrequencyUnit::NONE:
  default:
    return 0.0;
  }
}

void OutputControl::compute_next_write_step(int current_step, double dt_in) {
  dt = dt_in;

  if (!output_enabled() || dt <= 0) {
    next_write_step = -1;
    return;
  }

  if (frequency_unit == FrequencyUnit::NSTEPS) {
    // Simple: every N steps
    next_write_step = ((current_step / frequency) + 1) * frequency;
  } else {
    // Compute based on wall time
    double seconds_per_output = seconds_per_unit() * frequency;
    int steps_per_output = static_cast<int>(seconds_per_output / dt);
    if (steps_per_output < 1) {
      steps_per_output = 1;
    }
    next_write_step =
        ((current_step / steps_per_output) + 1) * steps_per_output;
  }
}

// ============================================================================
// EmulatorOutputStream implementation
// ============================================================================

void EmulatorOutputStream::initialize(const OutputStreamConfig &config,
                                      MPI_Comm comm,
                                      const std::vector<int> &col_gids,
                                      int nlat, int nlon, Logger &logger) {
  m_config = config;
  m_comm = comm;
  m_col_gids = col_gids;
  m_nlat = nlat;
  m_nlon = nlon;
  m_ncols_local = static_cast<int>(col_gids.size());
  m_logger = &logger;

  // Check if MPI communicator is valid
  if (comm != MPI_COMM_NULL) {
    MPI_Comm_rank(comm, &m_rank);
    m_is_root = (m_rank == 0);
  } else {
    // Fallback for non-MPI runs or uninitialized comm
    m_rank = 0;
    m_is_root = true;
  }

  // Set up control
  m_control.frequency = config.frequency;
  m_control.frequency_unit = config.frequency_unit;
  m_control.nsamples_since_last_write = 0;

  // Initialize averaging buffers if needed
  if (config.avg_type != OutputAvgType::INSTANT) {
    for (const auto &field_name : config.fields) {
      m_avg_buffer[field_name].resize(m_ncols_local, 0.0);
      if (config.avg_type == OutputAvgType::STD) {
        m_avg_buffer_sq[field_name].resize(m_ncols_local, 0.0);
      }
    }
  }

  m_initialized = true;

  if (m_is_root) {
    m_logger->info("Initialized output stream '" + config.stream_name +
                   "' with " + std::to_string(config.fields.size()) +
                   " fields, averaging=" + avg_type_to_str(config.avg_type));
  }
}

void EmulatorOutputStream::init_timestep(int current_step, double dt) {
  m_control.current_step = current_step;

  if (m_control.last_write_step < 0) {
    // First call - compute initial next write step
    m_control.compute_next_write_step(0, dt);
    m_control.last_write_step = 0;
  }

  m_control.dt = dt;
}

void EmulatorOutputStream::run(int current_step,
                               const FieldDataProvider &fields,
                               const std::string &case_name) {
  if (!m_initialized || !m_control.output_enabled()) {
    return;
  }

  m_control.current_step = current_step;

  // Update averaging buffers (for non-instant streams)
  if (m_config.avg_type != OutputAvgType::INSTANT) {
    update_averaging(fields);
  }

  // Check if we should write
  if (m_control.is_write_step(current_step)) {
    write_output(fields, case_name);

    // Update control
    m_control.last_write_step = current_step;
    m_control.compute_next_write_step(current_step, m_control.dt);

    // Reset averaging
    if (m_config.avg_type != OutputAvgType::INSTANT) {
      reset_averaging_buffers();
    }
  }
}

void EmulatorOutputStream::finalize() {
  // Close any open file
  if (m_current_file_ncid >= 0) {
    EmulatorIO::close_file(m_current_file_ncid);
    m_current_file_ncid = -1;
  }

  if (m_is_root && m_initialized) {
    m_logger->info("Finalized output stream '" + m_config.stream_name + "'");
  }
}

void EmulatorOutputStream::update_averaging(const FieldDataProvider &fields) {
  m_control.nsamples_since_last_write++;

  for (const auto &field_name : m_config.fields) {
    const auto *field_data = fields.get_field(field_name);
    if (!field_data) {
      continue;
    }

    auto &buffer = m_avg_buffer[field_name];
    const int n = std::min(static_cast<int>(buffer.size()),
                           static_cast<int>(field_data->size()));

    switch (m_config.avg_type) {
    case OutputAvgType::AVERAGE:
    case OutputAvgType::SUM:
      for (int i = 0; i < n; ++i) {
        buffer[i] += (*field_data)[i];
      }
      break;

    case OutputAvgType::MIN:
      if (m_control.nsamples_since_last_write == 1) {
        for (int i = 0; i < n; ++i) {
          buffer[i] = (*field_data)[i];
        }
      } else {
        for (int i = 0; i < n; ++i) {
          buffer[i] = std::min(buffer[i], (*field_data)[i]);
        }
      }
      break;

    case OutputAvgType::MAX:
      if (m_control.nsamples_since_last_write == 1) {
        for (int i = 0; i < n; ++i) {
          buffer[i] = (*field_data)[i];
        }
      } else {
        for (int i = 0; i < n; ++i) {
          buffer[i] = std::max(buffer[i], (*field_data)[i]);
        }
      }
      break;

    case OutputAvgType::STD:
      // Welford's online algorithm components
      for (int i = 0; i < n; ++i) {
        buffer[i] += (*field_data)[i];
        m_avg_buffer_sq[field_name][i] += (*field_data)[i] * (*field_data)[i];
      }
      break;

    default:
      break;
    }
  }
}

std::vector<double>
EmulatorOutputStream::get_output_data(const std::string &field_name,
                                      const FieldDataProvider &fields) const {
  if (m_config.avg_type == OutputAvgType::INSTANT) {
    const auto *field_data = fields.get_field(field_name);
    if (field_data) {
      return *field_data;
    }
    return std::vector<double>(m_ncols_local, 0.0);
  }

  // Get from averaging buffer
  auto it = m_avg_buffer.find(field_name);
  if (it == m_avg_buffer.end()) {
    return std::vector<double>(m_ncols_local, 0.0);
  }

  const auto &buffer = it->second;
  const int n = static_cast<int>(buffer.size());
  std::vector<double> output(n);

  const int nsamples = std::max(1, m_control.nsamples_since_last_write);

  switch (m_config.avg_type) {
  case OutputAvgType::AVERAGE:
    for (int i = 0; i < n; ++i) {
      output[i] = buffer[i] / nsamples;
    }
    break;

  case OutputAvgType::SUM:
  case OutputAvgType::MIN:
  case OutputAvgType::MAX:
    output = buffer;
    break;

  case OutputAvgType::STD: {
    auto it_sq = m_avg_buffer_sq.find(field_name);
    if (it_sq != m_avg_buffer_sq.end()) {
      const auto &buffer_sq = it_sq->second;
      for (int i = 0; i < n; ++i) {
        double mean = buffer[i] / nsamples;
        double mean_sq = buffer_sq[i] / nsamples;
        double variance = mean_sq - mean * mean;
        output[i] = std::sqrt(std::max(0.0, variance));
      }
    }
    break;
  }

  default:
    output = buffer;
    break;
  }

  return output;
}

void EmulatorOutputStream::reset_averaging_buffers() {
  m_control.nsamples_since_last_write = 0;

  for (auto &pair : m_avg_buffer) {
    std::fill(pair.second.begin(), pair.second.end(), 0.0);
  }

  for (auto &pair : m_avg_buffer_sq) {
    std::fill(pair.second.begin(), pair.second.end(), 0.0);
  }
}

std::string
EmulatorOutputStream::generate_filename(const std::string &case_name,
                                        int step) const {
  // Format: {case_name}.{prefix}.{step:010d}.nc
  std::ostringstream oss;
  oss << case_name << "." << m_config.filename_prefix << "."
      << std::setfill('0') << std::setw(10) << step << ".nc";
  return oss.str();
}

void EmulatorOutputStream::setup_file(const std::string &filename) {
  // Close existing file if open
  if (m_current_file_ncid >= 0) {
    EmulatorIO::close_file(m_current_file_ncid);
  }

  m_current_file_ncid = EmulatorIO::create_file(filename);
  m_current_filename = filename;
  m_snapshots_in_file = 0;

  if (m_current_file_ncid < 0 && m_is_root) {
    m_logger->error("Failed to create output file: " + filename);
  }
}

void EmulatorOutputStream::write_output(const FieldDataProvider &fields,
                                        const std::string &case_name) {
  // Generate filename
  std::string filename = generate_filename(case_name, m_control.current_step);

  // Check if we need a new file
  if (m_current_file_ncid < 0 ||
      m_snapshots_in_file >= m_config.max_snapshots_per_file) {
    setup_file(filename);
  }

  if (m_current_file_ncid < 0) {
    return; // Failed to create file
  }

  // Write each field
  for (const auto &field_name : m_config.fields) {
    std::vector<double> data = get_output_data(field_name, fields);

    // For now, assume 2D fields (ncols)
    // TODO: Handle 3D stacked fields
    EmulatorIO::write_var_1d(m_current_file_ncid, field_name, data.data(),
                             static_cast<int>(data.size()));
  }

  m_snapshots_in_file++;

  if (m_is_root) {
    m_logger->info("Wrote output stream '" + m_config.stream_name +
                   "' at step " + std::to_string(m_control.current_step));
  }
}

bool EmulatorOutputStream::write_history_restart(const std::string &filename) {
  if (!needs_history_restart()) {
    return true;
  }

  int ncid = EmulatorIO::create_file(filename);
  if (ncid < 0) {
    return false;
  }

  // Write control state as attributes
  // TODO: Write averaging buffers and control state

  // Write each buffer
  for (const auto &pair : m_avg_buffer) {
    EmulatorIO::write_var_1d(ncid, pair.first + "_avg_buffer",
                             pair.second.data(),
                             static_cast<int>(pair.second.size()));
  }

  for (const auto &pair : m_avg_buffer_sq) {
    EmulatorIO::write_var_1d(ncid, pair.first + "_avg_buffer_sq",
                             pair.second.data(),
                             static_cast<int>(pair.second.size()));
  }

  EmulatorIO::close_file(ncid);
  return true;
}

bool EmulatorOutputStream::read_history_restart(const std::string &filename) {
  if (!needs_history_restart()) {
    return true;
  }

  int ncid = EmulatorIO::open_file(filename);
  if (ncid < 0) {
    return false;
  }

  // Read each buffer
  for (auto &pair : m_avg_buffer) {
    EmulatorIO::read_var_1d(ncid, pair.first + "_avg_buffer",
                            pair.second.data(),
                            static_cast<int>(pair.second.size()));
  }

  for (auto &pair : m_avg_buffer_sq) {
    EmulatorIO::read_var_1d(ncid, pair.first + "_avg_buffer_sq",
                            pair.second.data(),
                            static_cast<int>(pair.second.size()));
  }

  // TODO: Read control state

  EmulatorIO::close_file(ncid);
  return true;
}

} // namespace emulator
