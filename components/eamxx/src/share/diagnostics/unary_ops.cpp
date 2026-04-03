#include "unary_ops.hpp"

#include "share/physics/physics_constants.hpp"

#include <cmath>

namespace {
  constexpr int OP_SQRT   = 0;
  constexpr int OP_ABS    = 1;
  constexpr int OP_LOG    = 2;
  constexpr int OP_EXP    = 3;
  constexpr int OP_SQUARE = 4;
}

namespace scream {

int get_unary_operator_code(const std::string& op) {
  if (op == "sqrt")   return OP_SQRT;
  if (op == "abs")    return OP_ABS;
  if (op == "log")    return OP_LOG;
  if (op == "exp")    return OP_EXP;
  if (op == "square") return OP_SQUARE;
  return -1;
}

ekat::units::Units apply_unary_op_units(const ekat::units::Units& u, const int op_code) {
  using namespace ekat::units;
  switch (op_code) {
    case OP_SQRT:
      // sqrt of units: require nondimensional for now, since ekat::units
      // doesn't provide a general sqrt() for units
      EKAT_REQUIRE_MSG(u == Units::nondimensional(),
        "Error! sqrt() currently requires nondimensional input units.\n"
        "  Hint: divide by appropriate units first, then apply sqrt.\n");
      return Units::nondimensional();
    case OP_ABS:
      return u;
    case OP_LOG:
      EKAT_REQUIRE_MSG(u == Units::nondimensional(),
        "Error! log() requires nondimensional input units.\n");
      return Units::nondimensional();
    case OP_EXP:
      EKAT_REQUIRE_MSG(u == Units::nondimensional(),
        "Error! exp() requires nondimensional input units.\n");
      return Units::nondimensional();
    case OP_SQUARE:
      return u * u;
    default:
      EKAT_ERROR_MSG("Error! Unrecognized/unsupported unary op code (" + std::to_string(op_code) + ").\n");
  }
  return Units::invalid();
}

UnaryOpsDiag::
UnaryOpsDiag(const ekat::Comm &comm,
             const ekat::ParameterList &params)
 : AtmosphereDiagnostic(comm, params)
{
  m_arg_name     = m_params.get<std::string>("arg");
  m_unary_op_str = m_params.get<std::string>("unary_op");
  m_unary_op_code = get_unary_operator_code(m_unary_op_str);

  EKAT_REQUIRE_MSG(m_unary_op_code >= 0,
                   "Error! Invalid unary operator: '" + m_unary_op_str + "'\n"
                   "Valid operators are: sqrt, abs, log, exp, square\n");
}

void UnaryOpsDiag::
create_requests()
{
  const auto &gname = m_params.get<std::string>("grid_name");
  const auto& pc_dict = physics::Constants<Real>::dictionary();
  m_arg_is_field = pc_dict.count(m_arg_name) == 0;
  if (m_arg_is_field)
    add_field<Required>(m_arg_name, gname);
}

void UnaryOpsDiag::initialize_impl(const RunType /*run_type*/)
{
  const auto& dict = physics::Constants<Real>::dictionary();

  auto dl = m_arg_is_field
            ? get_field_in(m_arg_name).get_header().get_identifier().get_layout()
            : FieldLayout({},{});
  const auto& u = m_arg_is_field
                  ? get_field_in(m_arg_name).get_header().get_identifier().get_units()
                  : dict.at(m_arg_name).units;

  auto diag_units = apply_unary_op_units(u, m_unary_op_code);
  auto gn = m_params.get<std::string>("grid_name");
  auto diag_name = m_unary_op_str + "_of_" + m_arg_name;
  FieldIdentifier d_fid(diag_name, dl, diag_units, gn);
  m_diagnostic_output = Field(d_fid, true);

  if (!m_arg_is_field) {
    // Constant input: pre-compute now
    compute_diagnostic_impl();
    m_diagnostic_output.get_header().get_tracking().update_time_stamp(start_of_step_ts());
  }
}

void UnaryOpsDiag::compute_diagnostic_impl()
{
  const auto& dict = physics::Constants<Real>::dictionary();

  if (m_arg_is_field) {
    const auto& f = get_field_in(m_arg_name);
    auto& d = m_diagnostic_output;

    // Compute the unary operation element-wise
    const int total_size = f.get_header().get_identifier().get_layout().size();

    // Get raw views regardless of rank
    const auto f_data = f.get_internal_view_data<const Real>();
    auto d_data = d.get_internal_view_data<Real>();

    const int op_code = m_unary_op_code;
    Kokkos::parallel_for("UnaryOpsDiag",
                         Kokkos::RangePolicy<>(0, total_size),
                         KOKKOS_LAMBDA(const int& i) {
      const auto val = f_data[i];
      switch (op_code) {
        case OP_SQRT:   d_data[i] = sqrt(val);   break;
        case OP_ABS:    d_data[i] = fabs(val);    break;
        case OP_LOG:    d_data[i] = log(val);     break;
        case OP_EXP:    d_data[i] = exp(val);     break;
        case OP_SQUARE: d_data[i] = val * val;    break;
      }
    });
  } else {
    const Real c = dict.at(m_arg_name).value;
    Real result;
    switch (m_unary_op_code) {
      case OP_SQRT:   result = std::sqrt(c);  break;
      case OP_ABS:    result = std::abs(c);    break;
      case OP_LOG:    result = std::log(c);    break;
      case OP_EXP:    result = std::exp(c);    break;
      case OP_SQUARE: result = c * c;          break;
      default:
        EKAT_ERROR_MSG("Error! Unrecognized unary op code.\n");
    }
    m_diagnostic_output.deep_copy(result);
  }
}

}  // namespace scream
