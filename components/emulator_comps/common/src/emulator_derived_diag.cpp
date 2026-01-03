/**
 * @file emulator_derived_diag.cpp
 * @brief Implementation of derived diagnostics.
 */

#include "emulator_derived_diag.hpp"
#include <algorithm>
#include <cmath>
#include <regex>
#include <stdexcept>

namespace emulator {

// ============================================================================
// HorizAvgDiagnostic implementation
// ============================================================================

HorizAvgDiagnostic::HorizAvgDiagnostic(const std::string &field_name,
                                       const std::vector<double> &area_weights,
                                       MPI_Comm comm)
    : m_source_field(field_name), m_area_weights(area_weights), m_comm(comm) {
  m_name = field_name + "_horiz_avg";
}

void HorizAvgDiagnostic::compute(const FieldDataProvider &fields,
                                 std::vector<double> &output) {
  const auto *field_data = fields.get_field(m_source_field);
  if (!field_data || field_data->empty()) {
    output.assign(1, 0.0);
    return;
  }

  int ncols = fields.get_ncols();
  int nlevs = fields.get_field_nlevs(m_source_field);
  output.resize(nlevs);

  // Check if we have area weights
  bool have_weights = (m_area_weights.size() == static_cast<size_t>(ncols));

  // Compute local weighted sum for each level
  std::vector<double> local_sum(nlevs, 0.0);
  double local_weight_sum = 0.0;

  for (int lev = 0; lev < nlevs; ++lev) {
    for (int col = 0; col < ncols; ++col) {
      int idx = lev * ncols + col; // Assuming [nlevs, ncols] layout
      if (idx >= static_cast<int>(field_data->size())) {
        // Fallback for [ncols] layout
        idx = col;
      }

      double weight = have_weights ? m_area_weights[col] : 1.0;
      local_sum[lev] += (*field_data)[idx] * weight;

      if (lev == 0) {
        local_weight_sum += weight;
      }
    }
  }

  // Global reduction
  std::vector<double> global_sum(nlevs);
  double global_weight_sum = 0.0;

  MPI_Allreduce(local_sum.data(), global_sum.data(), nlevs, MPI_DOUBLE, MPI_SUM,
                m_comm);
  MPI_Allreduce(&local_weight_sum, &global_weight_sum, 1, MPI_DOUBLE, MPI_SUM,
                m_comm);

  // Compute average
  if (global_weight_sum > 0.0) {
    for (int lev = 0; lev < nlevs; ++lev) {
      output[lev] = global_sum[lev] / global_weight_sum;
    }
  } else {
    std::fill(output.begin(), output.end(), 0.0);
  }
}

// ============================================================================
// VertSliceDiagnostic implementation
// ============================================================================

VertSliceDiagnostic::VertSliceDiagnostic(const std::string &field_name,
                                         int level_idx, int nlevs)
    : m_source_field(field_name), m_level_idx(level_idx), m_nlevs(nlevs) {
  m_name = field_name + "_at_lev" + std::to_string(level_idx);
}

void VertSliceDiagnostic::compute(const FieldDataProvider &fields,
                                  std::vector<double> &output) {
  const auto *field_data = fields.get_field(m_source_field);
  if (!field_data || field_data->empty()) {
    output.clear();
    return;
  }

  int ncols = fields.get_ncols();
  output.resize(ncols);

  // Check if source is 3D (stacked slices) or 2D
  int total_size = static_cast<int>(field_data->size());
  int detected_nlevs = total_size / ncols;

  if (detected_nlevs <= 1 || m_level_idx >= detected_nlevs) {
    // 2D field or invalid level - just copy the field
    for (int col = 0; col < ncols && col < total_size; ++col) {
      output[col] = (*field_data)[col];
    }
    return;
  }

  // Extract slice: assuming [nlevs, ncols] layout
  for (int col = 0; col < ncols; ++col) {
    int idx = m_level_idx * ncols + col;
    output[col] = (*field_data)[idx];
  }
}

// ============================================================================
// Diagnostic Factory
// ============================================================================

namespace {
// Regex patterns for diagnostic parsing
const std::regex HORIZ_AVG_PATTERN(R"((.+)_(horiz_avg|global_mean)$)");
const std::regex VERT_SLICE_PATTERN(R"((.+)_at_lev(\d+)$)");
} // namespace

bool is_derived_diagnostic(const std::string &name) {
  return std::regex_match(name, HORIZ_AVG_PATTERN) ||
         std::regex_match(name, VERT_SLICE_PATTERN);
}

std::string get_base_field_name(const std::string &diag_name) {
  std::smatch match;

  if (std::regex_match(diag_name, match, HORIZ_AVG_PATTERN)) {
    return match[1].str();
  }

  if (std::regex_match(diag_name, match, VERT_SLICE_PATTERN)) {
    return match[1].str();
  }

  return diag_name; // Not a diagnostic pattern
}

std::unique_ptr<DerivedDiagnostic>
create_diagnostic(const std::string &diag_name,
                  const DiagnosticMetadata &metadata) {
  std::smatch match;

  // Check horizontal average pattern
  if (std::regex_match(diag_name, match, HORIZ_AVG_PATTERN)) {
    std::string field_name = match[1].str();
    return std::make_unique<HorizAvgDiagnostic>(
        field_name, metadata.area_weights, metadata.comm);
  }

  // Check vertical slice pattern
  if (std::regex_match(diag_name, match, VERT_SLICE_PATTERN)) {
    std::string field_name = match[1].str();
    int level_idx = std::stoi(match[2].str());
    return std::make_unique<VertSliceDiagnostic>(field_name, level_idx,
                                                 metadata.nlevs);
  }

  // Not a diagnostic pattern
  return nullptr;
}

} // namespace emulator
