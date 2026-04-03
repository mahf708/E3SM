#ifndef EAMXX_EXPRESSION_DIAG_HPP
#define EAMXX_EXPRESSION_DIAG_HPP

#include "share/atm_process/atmosphere_diagnostic.hpp"
#include "diagnostics_core/eamxx_field_accessor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace scream {

// ---- ExpressionDiag diagnostic class ----
//
// Evaluates an arbitrary arithmetic expression over fields.
// Uses the standalone diag_utils::ExpressionEvaluator engine,
// adapted to EAMxx fields via EamxxFieldAccessor.
//
// YAML naming convention:  expr_{expression}
//   e.g.  expr_qc*pseudo_density/gravit
//         expr_sqrt(u**2+v**2)
//
// Parameters:
//   expression - the expression string
//   grid_name  - grid name

class ExpressionDiag : public AtmosphereDiagnostic {
 public:
  ExpressionDiag(const ekat::Comm &comm, const ekat::ParameterList &params);

  std::string name() const override { return "ExpressionDiag"; }

  void create_requests() override;

 protected:
  void compute_diagnostic_impl() override;
  void initialize_impl(const RunType /*run_type*/) override;

  // The standalone expression evaluator (from diagnostics_core)
  std::unique_ptr<diag_utils::EamxxExpressionEvaluator> m_evaluator;

  // The raw expression string (kept for diagnostics/logging)
  std::string m_expr_str;
};

}  // namespace scream

#endif  // EAMXX_EXPRESSION_DIAG_HPP
