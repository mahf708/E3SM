#include "expression_diag.hpp"

#include "share/physics/physics_constants.hpp"

namespace scream {

// Create a constant resolver that maps physics constant names to their values
static diag_utils::ConstResolver make_physics_resolver() {
  return [](const std::string& name, double& val) -> bool {
    const auto& dict = physics::Constants<Real>::dictionary();
    ekat::CaseInsensitiveString ci_name(name);
    if (dict.count(ci_name) > 0) {
      val = dict.at(ci_name).value;
      return true;
    }
    return false;
  };
}

ExpressionDiag::
ExpressionDiag(const ekat::Comm &comm, const ekat::ParameterList &params)
 : AtmosphereDiagnostic(comm, params)
{
  m_expr_str = m_params.get<std::string>("expression");

  EKAT_REQUIRE_MSG(!m_expr_str.empty(),
    "Error! ExpressionDiag requires a non-empty 'expression' parameter.\n");

  // Create the evaluator with EAMxx physics constants resolver
  m_evaluator = std::make_unique<diag_utils::EamxxExpressionEvaluator>(
    m_expr_str, make_physics_resolver());

  EKAT_REQUIRE_MSG(!m_evaluator->program().empty(),
    "Error! ExpressionDiag: expression '" + m_expr_str + "' produced no instructions.\n");
}

void ExpressionDiag::
create_requests()
{
  const auto &gname = m_params.get<std::string>("grid_name");
  for (const auto& fname : m_evaluator->field_names()) {
    add_field<Required>(fname, gname);
  }
}

void ExpressionDiag::initialize_impl(const RunType /*run_type*/)
{
  const auto& field_names = m_evaluator->field_names();

  EKAT_REQUIRE_MSG(!field_names.empty(),
    "Error! ExpressionDiag expression '" + m_expr_str +
    "' references no fields. Use BinaryOpsDiag for constant-only expressions.\n");

  const auto& first_fid = get_field_in(field_names[0]).get_header().get_identifier();
  const auto& dl = first_fid.get_layout();

  for (size_t i = 1; i < field_names.size(); ++i) {
    const auto& fid = get_field_in(field_names[i]).get_header().get_identifier();
    EKAT_REQUIRE_MSG(fid.get_layout() == dl,
      "Error! ExpressionDiag requires all fields to have the same layout.\n"
      " - expression: " + m_expr_str + "\n"
      " - field '" + field_names[0] + "' layout: " + dl.to_string() + "\n"
      " - field '" + field_names[i] + "' layout: " + fid.get_layout().to_string() + "\n");
  }

  auto gn = m_params.get<std::string>("grid_name");
  FieldIdentifier d_fid("expr_" + m_expr_str, dl,
                        ekat::units::Units::nondimensional(), gn);
  m_diagnostic_output = Field(d_fid, true);
}

void ExpressionDiag::compute_diagnostic_impl()
{
  const auto& field_names = m_evaluator->field_names();

  // Gather input fields in the order expected by the evaluator
  std::vector<const Field*> input_fields;
  input_fields.reserve(field_names.size());
  for (const auto& fname : field_names) {
    input_fields.push_back(&get_field_in(fname));
  }

  // Delegate to the standalone evaluator
  m_evaluator->evaluate(input_fields, m_diagnostic_output);
}

}  // namespace scream
