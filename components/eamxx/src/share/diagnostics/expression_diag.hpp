#ifndef EAMXX_EXPRESSION_DIAG_HPP
#define EAMXX_EXPRESSION_DIAG_HPP

#include "share/atm_process/atmosphere_diagnostic.hpp"

#include <string>
#include <vector>

namespace scream {

// ---- RPN instruction set for GPU-safe expression evaluation ----
//
// Expressions are parsed on the host into a flat array of Instruction
// structs.  The evaluator runs a simple stack machine on the GPU,
// using a fixed-size stack (max depth = MAX_EXPR_STACK).
//
// Supported operations:
//   Binary:  +  -  *  /  ** (power)
//   Unary:   sqrt  abs  log  exp  square  neg (unary minus)
//   Operands: field values (by index), numeric literals, physics constants

constexpr int MAX_EXPR_STACK = 16;  // max operand stack depth
constexpr int MAX_EXPR_INSTRUCTIONS = 64;  // max instructions per expression

// Instruction opcodes
enum class ExprOp : int {
  // Operand push
  PushField    = 0,   // push field_ptrs[operand_idx][i]
  PushConst    = 1,   // push const_val

  // Binary ops (pop 2, push 1)
  Add          = 10,
  Sub          = 11,
  Mul          = 12,
  Div          = 13,
  Pow          = 14,

  // Unary ops (pop 1, push 1)
  Sqrt         = 20,
  Abs          = 21,
  Log          = 22,
  Exp          = 23,
  Square       = 24,
  Neg          = 25,
};

struct Instruction {
  ExprOp op;
  int    operand_idx;  // index into field_ptrs array (for PushField)
  Real   const_val;    // literal value (for PushConst)
};

// ---- ExpressionDiag diagnostic class ----
//
// Evaluates an arbitrary arithmetic expression over fields.
//
// YAML naming convention:  expr_{expression}
//   e.g.  expr_qc*pseudo_density/gravit
//         expr_sqrt(T_mid)
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

  // The raw expression string
  std::string m_expr_str;

  // Parsed RPN program
  std::vector<Instruction> m_program;

  // Field names referenced by the expression (in order of first appearance)
  std::vector<std::string> m_field_names;
};

// ---- Expression parser (host-side only) ----
//
// Parses an infix expression string into RPN instructions.
// Returns the program and populates field_names with referenced fields.
//
// Grammar:
//   expr     = term (('+' | '-') term)*
//   term     = power (('*' | '/') power)*
//   power    = unary ('**' unary)*
//   unary    = ('-' unary) | func '(' expr ')' | atom
//   atom     = NUMBER | IDENTIFIER | '(' expr ')'
//   func     = 'sqrt' | 'abs' | 'log' | 'exp' | 'square'

struct ParseResult {
  std::vector<Instruction> program;
  std::vector<std::string> field_names;
};

ParseResult parse_expression(const std::string& expr);

}  // namespace scream

#endif  // EAMXX_EXPRESSION_DIAG_HPP
