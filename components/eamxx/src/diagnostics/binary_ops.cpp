#include "diagnostics/binary_ops.hpp"
#include "share/physics/physics_constants.hpp"

namespace scream {

// parse string to get the operator code
int get_binary_operator_code(const std::string& op) {
  if (op == "plus") return 0; // addition
  if (op == "minus") return 1; // subtraction
  if (op == "times") return 2; // multiplication
  if (op == "over") return 3; // division
  if (op == "times_rho_h2o") return 4; // multiply by water density
  if (op == "over_rho_h2o")  return 5; // divide by water density
  if (op == "times_gravit")  return 6; // multiply by gravity
  if (op == "over_gravit")   return 7; // divide by gravity
  return -1; // invalid operator
}
// apply binary operation on two input units
ekat::units::Units apply_binary_op(const ekat::units::Units& a, const ekat::units::Units& b, const int op_code) {
  using namespace ekat::units;
  switch (op_code) {
    case 0: // addition - units must be compatible
    case 1: // subtraction - units must be compatible
      EKAT_REQUIRE_MSG(a == b, "Error! Addition/subtraction requires compatible units.\n");
      return a;
    case 2: return a * b; // multiplication
    case 3: return a / b; // division
    case 4: {
      const auto rho = kg / (m*m*m);
      return a * rho;
    }
    case 5: {
      const auto rho = kg / (m*m*m);
      return a / rho;
    }
    case 6: {
      const auto g = m / (s*s);
      return a * g;
    }
    case 7: {
      const auto g = m / (s*s);
      return a / g;
    }
    default: return a; // no operation, just return a
  }
}
// apply binary operation on two input fields
void apply_binary_op(Field& a_clone, const Field &b, const int& op_code){
  switch (op_code) {
    case 0: return a_clone.update(b, 1, 1); // addition
    case 1: return a_clone.update(b, -1, 1); // subtraction
    case 2: return a_clone.scale(b); // multiplication
    case 3: return a_clone.scale_inv(b); // division
    case 4: {
      using PC = scream::physics::Constants<Real>;
      return a_clone.scale(PC::RHO_H2O);
    }
    case 5: {
      using PC = scream::physics::Constants<Real>;
      return a_clone.scale(1.0/PC::RHO_H2O);
    }
    case 6: {
      using PC = scream::physics::Constants<Real>;
      return a_clone.scale(PC::gravit);
    }
    case 7: {
      using PC = scream::physics::Constants<Real>;
      return a_clone.scale(1.0/PC::gravit);
    }
    default: return;
  }
}

BinaryOpsDiag::
BinaryOpsDiag(const ekat::Comm &comm,
                const ekat::ParameterList &params)
 : AtmosphereDiagnostic(comm, params)
{
  m_field_1 = m_params.get<std::string>("field_1");
  m_field_2 = m_params.get<std::string>("field_2");
  m_binary_op = m_params.get<std::string>("binary_op");

  // Validate operator
  EKAT_REQUIRE_MSG(get_binary_operator_code(m_binary_op) >= 0,
                   "Error! Invalid binary operator: '" + m_binary_op + "'\n"
                   "Valid operators are: plus, minus, times, over, times_rho_h2o, over_rho_h2o, times_gravit, over_gravit\n");
}

void BinaryOpsDiag::
set_grids(const std::shared_ptr<const GridsManager> grids_manager)
{
  const auto &gname = m_params.get<std::string>("grid_name");
  add_field<Required>(m_field_1, gname);
  if (m_field_2 != "") {
    add_field<Required>(m_field_2, gname);
  }
}

void BinaryOpsDiag::initialize_impl(const RunType /*run_type*/) {
  // get the input fields
  const auto &f1   = get_field_in(m_field_1);
  const auto &l1 = f1.get_header().get_identifier().get_layout();

  auto f2 = f1;
  auto l2 = l1;
  if (m_field_2 != "") {
    f2   = get_field_in(m_field_2);
    l2 = f2.get_header().get_identifier().get_layout();
  }

  // Must be on same layout, same datatype
  EKAT_REQUIRE_MSG(
    l1 == l2,
    "Error! BinaryOpsDiag requires both input fields to have the same layout.\n"
    " - field 1 name: " + f1.get_header().get_identifier().name() + "\n"
    " - field 1 layout: " + l1.to_string() + "\n"
    " - field 2 name: " + f2.get_header().get_identifier().name() + "\n"
    " - field 2 layout: " + l2.to_string() + "\n");
  EKAT_REQUIRE_MSG(
    f1.data_type() == f2.data_type(),
    "Error! BinaryOpsDiag requires both input fields to have the same data type.\n"
    " - field 1 name: " + f1.get_header().get_identifier().name() + "\n"
    " - field 1 data type: " + e2str(f1.data_type()) + "\n"
    " - field 2 name: " + f2.get_header().get_identifier().name() + "\n"
    " - field 2 data type: " + e2str(f2.data_type()) + "\n");

  const auto &gn1  = f1.get_header().get_identifier().get_grid_name();
  const auto &gn2  = f2.get_header().get_identifier().get_grid_name();
  // Must be on same grid too
  EKAT_REQUIRE_MSG(
    gn1 == gn2,
    "Error! BinaryOpsDiag requires both input fields to be on the same grid.\n"
    " - field 1 name: " + f1.get_header().get_identifier().name() + "\n"
    " - field 1 grid name: " + gn1 + "\n"
    " - field 2 name: " + f2.get_header().get_identifier().name() + "\n"
    " - field 2 grid name: " + gn2 + "\n");
  
  const auto &u1 = f1.get_header().get_identifier().get_units();
  const auto &u2 = f2.get_header().get_identifier().get_units();

  // All good, create the diag output
  auto diag_units = apply_binary_op(u1, u2, get_binary_operator_code(m_binary_op));
  auto diag_name = m_field_2.empty() ? (m_field_1 + "_" + m_binary_op) : (m_field_1 + "_" + m_binary_op + "_" + m_field_2);
  FieldIdentifier d_fid(diag_name, l1.clone(), diag_units, gn1);
  m_diagnostic_output = Field(d_fid);
  m_diagnostic_output.allocate_view();
}

void BinaryOpsDiag::compute_diagnostic_impl() {

  const auto &f1 = get_field_in(m_field_1);
  auto f2 = f1;
  if (m_field_2 != "") {
    f2 = get_field_in(m_field_2);
  }
  m_diagnostic_output.deep_copy(f1);
  apply_binary_op(m_diagnostic_output, f2, get_binary_operator_code(m_binary_op));
}

}  // namespace scream
