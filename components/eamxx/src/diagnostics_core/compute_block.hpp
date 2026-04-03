#ifndef KOKKOS_DIAG_COMPUTE_BLOCK_HPP
#define KOKKOS_DIAG_COMPUTE_BLOCK_HPP

// ====================================================================
// Multi-statement compute block parser and evaluator
//
// PROTOTYPE: Extends the expression DSL with multi-line blocks
// that support intermediate variables, conditionals, and reductions.
//
// This is a design exploration for what an "extended DSL" could look
// like within the existing RPN framework. The key insight is that
// multi-statement blocks are compiled into a sequence of RPN programs
// that execute in order, with named intermediate results stored in
// a local variable table.
//
// Syntax (in YAML):
//
//   compute_fields:
//     - name: lwp
//       inputs: [qc, pseudo_density]
//       compute: |
//         integrand = qc * pseudo_density / gravit
//         result = col_sum(integrand)
//
//     - name: cloud_T
//       inputs: [T_mid, cldfrac_liq]
//       compute: |
//         mask = gt(cldfrac_liq, 0.5)
//         result = where(mask, T_mid, -9999.0)
//
//     - name: tropo_pressure
//       inputs: [T_mid, p_mid]
//       compute: |
//         cold_mask = lt(T_mid, 220.0)
//         result = where(cold_mask, p_mid, 0.0)
//         result = col_avg(result)
//
// Supported statement types:
//   variable = expression          (element-wise assignment)
//   variable = col_sum(expr)       (vertical sum reduction)
//   variable = col_avg(expr)       (vertical average reduction)
//   variable = horiz_avg(expr)     (horizontal average reduction)
//
// The special variable name "result" is the output of the compute block.
//
// Reduction operations change the field dimensions:
//   col_sum/col_avg: (ncol, nlev) -> (ncol,)
//   horiz_avg:       (ncol, ...) -> (...)
//
// This header provides the parser. Evaluation requires model-specific
// infrastructure (TeamPolicy for reductions) and is implemented in
// the EAMxx adapter.
// ====================================================================

#include "expression_parser.hpp"

#include <string>
#include <vector>

namespace diag_utils {

// Types of statements in a compute block
enum class StmtType {
  Assign,     // var = expression (element-wise)
  ColSum,     // var = col_sum(expression)
  ColAvg,     // var = col_avg(expression)
  HorizAvg,   // var = horiz_avg(expression)
};

// A single statement in a compute block
struct ComputeStatement {
  StmtType    type;
  std::string target_var;   // left-hand side variable name
  std::string expression;   // the expression string (right side)
  ParseResult parsed;       // parsed expression (filled after parsing)
};

// A complete compute block
struct ComputeBlock {
  std::string              name;        // output field name
  std::vector<std::string> input_names; // declared input fields
  std::vector<ComputeStatement> statements;
};

// Parse a multi-line compute block string into structured statements.
//
// Each line has the form:
//   target = expression
//   target = col_sum(expression)
//   target = col_avg(expression)
//   target = horiz_avg(expression)
//
// Blank lines and lines starting with # are ignored.
inline ComputeBlock parse_compute_block(
    const std::string& name,
    const std::vector<std::string>& inputs,
    const std::string& body,
    const ConstResolver& resolver = nullptr)
{
  ComputeBlock block;
  block.name = name;
  block.input_names = inputs;

  // Split body into lines
  std::istringstream stream(body);
  std::string line;

  while (std::getline(stream, line)) {
    // Strip whitespace
    auto start = line.find_first_not_of(" \t");
    if (start == std::string::npos) continue;
    line = line.substr(start);

    // Skip comments and blank lines
    if (line.empty() || line[0] == '#') continue;

    // Find the '=' assignment
    auto eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      throw std::runtime_error(
        "Compute block parse error: expected 'var = expr' but got: " + line);
    }

    // Don't confuse == with =
    if (eq_pos + 1 < line.size() && line[eq_pos + 1] == '=') {
      throw std::runtime_error(
        "Compute block parse error: use eq() instead of == in: " + line);
    }
    // Also check for !=, <=, >= before the =
    if (eq_pos > 0 && (line[eq_pos-1] == '!' || line[eq_pos-1] == '<' || line[eq_pos-1] == '>')) {
      throw std::runtime_error(
        "Compute block parse error: use ne()/le()/ge() instead of operators in: " + line);
    }

    std::string target = line.substr(0, eq_pos);
    std::string rhs = line.substr(eq_pos + 1);

    // Strip whitespace from target and rhs
    auto strip = [](const std::string& s) -> std::string {
      auto a = s.find_first_not_of(" \t");
      if (a == std::string::npos) return "";
      auto b = s.find_last_not_of(" \t");
      return s.substr(a, b - a + 1);
    };
    target = strip(target);
    rhs = strip(rhs);

    if (target.empty() || rhs.empty()) {
      throw std::runtime_error(
        "Compute block parse error: empty target or expression in: " + line);
    }

    // Check for reduction wrappers
    ComputeStatement stmt;
    stmt.target_var = target;

    if (rhs.substr(0, 8) == "col_sum(" && rhs.back() == ')') {
      stmt.type = StmtType::ColSum;
      stmt.expression = rhs.substr(8, rhs.size() - 9);
    } else if (rhs.substr(0, 8) == "col_avg(" && rhs.back() == ')') {
      stmt.type = StmtType::ColAvg;
      stmt.expression = rhs.substr(8, rhs.size() - 9);
    } else if (rhs.substr(0, 10) == "horiz_avg(" && rhs.back() == ')') {
      stmt.type = StmtType::HorizAvg;
      stmt.expression = rhs.substr(10, rhs.size() - 11);
    } else {
      stmt.type = StmtType::Assign;
      stmt.expression = rhs;
    }

    // Parse the expression (field names resolved against inputs + prior targets)
    stmt.parsed = resolver
                  ? parse_expression(stmt.expression, resolver)
                  : parse_expression(stmt.expression);

    block.statements.push_back(std::move(stmt));
  }

  // Verify that the last statement assigns to "result"
  if (block.statements.empty()) {
    throw std::runtime_error(
      "Compute block '" + name + "' has no statements.");
  }
  if (block.statements.back().target_var != "result") {
    throw std::runtime_error(
      "Compute block '" + name + "': last statement must assign to 'result'. "
      "Got '" + block.statements.back().target_var + "' instead.");
  }

  return block;
}

} // namespace diag_utils

#endif // KOKKOS_DIAG_COMPUTE_BLOCK_HPP
