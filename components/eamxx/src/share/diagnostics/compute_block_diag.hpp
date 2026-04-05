#ifndef EAMXX_COMPUTE_BLOCK_DIAG_HPP
#define EAMXX_COMPUTE_BLOCK_DIAG_HPP

#include "share/atm_process/atmosphere_diagnostic.hpp"
#include "diagnostics_core/compute_block.hpp"
#include "diagnostics_core/eamxx_field_accessor.hpp"

#include <memory>
#include <string>
#include <vector>
#include <map>

namespace scream {

// ---- ComputeBlockDiag diagnostic class ----
//
// Evaluates an multi-statement compute block over fields.
// Supports intermediate variables and basic reductions.
//
// Parameters:
//   compute    - the compute block string
//   grid_name  - grid name
//   inputs     - list of input variables (vector of strings)

class ComputeBlockDiag : public AtmosphereDiagnostic {
 public:
  ComputeBlockDiag(const ekat::Comm &comm, const ekat::ParameterList &params);

  std::string name() const override { return "ComputeBlockDiag"; }

  void create_requests() override;

 protected:
  void compute_diagnostic_impl() override;
  void initialize_impl(const RunType /*run_type*/) override;

  // Compute block representation
  diag_utils::ComputeBlock m_block;

  // Mapping from variable name to field (inputs, intermediates, and result)
  std::map<std::string, Field> m_vars;

  // We need to keep evaluator instances around for each statement
  std::vector<std::unique_ptr<diag_utils::EamxxExpressionEvaluator>> m_evaluators;
};

}  // namespace scream

#endif  // EAMXX_COMPUTE_BLOCK_DIAG_HPP
