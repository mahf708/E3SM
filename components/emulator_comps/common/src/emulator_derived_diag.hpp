/**
 * @file emulator_derived_diag.hpp
 * @brief Derived diagnostic infrastructure.
 *
 * Provides base class and concrete implementations for computing
 * diagnostics from model fields, including horizontal averaging
 * and vertical slicing.
 */

#ifndef EMULATOR_DERIVED_DIAG_HPP
#define EMULATOR_DERIVED_DIAG_HPP

#include "emulator_output_stream.hpp"
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace emulator {

/**
 * @brief Base class for derived diagnostics.
 *
 * Derived diagnostics compute output values from one or more input fields.
 * Examples: horizontal averages, vertical slices, pressure interpolation.
 */
class DerivedDiagnostic {
public:
  virtual ~DerivedDiagnostic() = default;

  /**
   * @brief Get the diagnostic name.
   */
  virtual std::string name() const = 0;

  /**
   * @brief Get the source field name.
   */
  virtual std::string source_field() const = 0;

  /**
   * @brief Compute the diagnostic.
   * @param fields Input field provider
   * @param output Output data buffer (caller must allocate)
   */
  virtual void compute(const FieldDataProvider &fields,
                       std::vector<double> &output) = 0;

  /**
   * @brief Get output size.
   * @param ncols Number of local columns
   * @param nlevs Number of vertical levels (for source field)
   */
  virtual int output_size(int ncols, int nlevs) const = 0;
};

// ============================================================================
// Horizontal Average Diagnostic
// ============================================================================

/**
 * @brief Computes area-weighted horizontal average of a field.
 *
 * Reduces a field from [ncols] or [ncols, nlevs] to a scalar (or [nlevs]).
 *
 * ## Pattern
 * Triggered by field names ending in "_horiz_avg" or "_global_mean".
 *
 * ## Example
 * - Input: "surface_temperature" with shape [ncols]
 * - Request: "surface_temperature_horiz_avg"
 * - Output: scalar (global mean)
 */
class HorizAvgDiagnostic : public DerivedDiagnostic {
public:
  /**
   * @brief Construct horizontal averaging diagnostic.
   * @param field_name Source field name
   * @param area_weights Area weights for each column (normalized to sum to 1)
   * @param comm MPI communicator for global reduction
   */
  HorizAvgDiagnostic(const std::string &field_name,
                     const std::vector<double> &area_weights, MPI_Comm comm);

  std::string name() const override { return m_name; }
  std::string source_field() const override { return m_source_field; }

  void compute(const FieldDataProvider &fields,
               std::vector<double> &output) override;

  int output_size(int ncols, int nlevs) const override {
    (void)ncols;
    return nlevs; // Returns one value per level (or 1 for 2D fields)
  }

private:
  std::string m_name;
  std::string m_source_field;
  std::vector<double> m_area_weights;
  MPI_Comm m_comm;
};

// ============================================================================
// Vertical Slice Diagnostic
// ============================================================================

/**
 * @brief Extracts a single vertical level from a 3D field.
 *
 * Reduces a field from [ncols, nlevs] to [ncols].
 *
 * ## Pattern
 * Triggered by field names with "_at_lev{N}" suffix.
 *
 * ## Example
 * - Input: "air_temperature" with shape [ncols, 8]
 * - Request: "air_temperature_at_lev3"
 * - Output: "air_temperature" at level index 3, shape [ncols]
 */
class VertSliceDiagnostic : public DerivedDiagnostic {
public:
  /**
   * @brief Construct vertical slicing diagnostic.
   * @param field_name Source field name
   * @param level_idx Level index to extract (0-based)
   * @param nlevs Total number of levels in source field
   */
  VertSliceDiagnostic(const std::string &field_name, int level_idx, int nlevs);

  std::string name() const override { return m_name; }
  std::string source_field() const override { return m_source_field; }

  void compute(const FieldDataProvider &fields,
               std::vector<double> &output) override;

  int output_size(int ncols, int nlevs) const override {
    (void)nlevs;
    return ncols; // Returns one value per column
  }

private:
  std::string m_name;
  std::string m_source_field;
  int m_level_idx;
  int m_nlevs;
};

// ============================================================================
// Diagnostic Factory
// ============================================================================

/**
 * @brief Metadata for creating diagnostics.
 */
struct DiagnosticMetadata {
  std::vector<double> area_weights; ///< Area weights for horiz averaging
  MPI_Comm comm = MPI_COMM_WORLD;   ///< MPI communicator
  int nlevs = 1;                    ///< Default number of levels
};

/**
 * @brief Parse a field name and create appropriate diagnostic if needed.
 *
 * Recognized patterns:
 * - "{field}_horiz_avg" or "{field}_global_mean" → HorizAvgDiagnostic
 * - "{field}_at_lev{N}" → VertSliceDiagnostic at level N
 *
 * @param diag_name Requested diagnostic/field name
 * @param metadata Context for creating diagnostics
 * @return Unique pointer to diagnostic, or nullptr if name is not a diagnostic
 */
std::unique_ptr<DerivedDiagnostic>
create_diagnostic(const std::string &diag_name,
                  const DiagnosticMetadata &metadata);

/**
 * @brief Check if a field name is a derived diagnostic pattern.
 * @param name Field name to check
 * @return true if name matches a diagnostic pattern
 */
bool is_derived_diagnostic(const std::string &name);

/**
 * @brief Extract base field name from diagnostic name.
 *
 * Examples:
 * - "T_horiz_avg" → "T"
 * - "T_at_lev3" → "T"
 *
 * @param diag_name Diagnostic name
 * @return Base field name
 */
std::string get_base_field_name(const std::string &diag_name);

} // namespace emulator

#endif // EMULATOR_DERIVED_DIAG_HPP
