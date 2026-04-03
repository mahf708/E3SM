#ifndef EAMXX_UNARY_OPS_DIAG_HPP
#define EAMXX_UNARY_OPS_DIAG_HPP

#include "share/atm_process/atmosphere_diagnostic.hpp"

namespace scream {

/*
 * This diagnostic applies a unary operation (sqrt, abs, log, exp, square)
 * to an input field or a known physical constant.
 *
 * YAML naming convention:  {op}_of_{field}
 *   e.g.  sqrt_of_wind_speed, abs_of_T_mid, square_of_qc
 *
 * Parameters:
 *   arg      - field name or physics constant name
 *   unary_op - one of: sqrt, abs, log, exp, square
 */

class UnaryOpsDiag : public AtmosphereDiagnostic {
 public:
  UnaryOpsDiag(const ekat::Comm &comm, const ekat::ParameterList &params);

  std::string name() const override { return "UnaryOpsDiag"; }

  void create_requests() override;

 protected:
  void compute_diagnostic_impl() override;
  void initialize_impl(const RunType /*run_type*/) override;

  std::string m_arg_name;
  std::string m_unary_op_str;
  int         m_unary_op_code;
  bool        m_arg_is_field;
};

}  // namespace scream

#endif  // EAMXX_UNARY_OPS_DIAG_HPP
