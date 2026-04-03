#ifndef KOKKOS_DIAG_PY_TRANSLATOR_HPP
#define KOKKOS_DIAG_PY_TRANSLATOR_HPP

// ====================================================================
// Python-to-ComputeBlock translator
//
// Reads a Python source file containing a single `compute()` function
// and translates it into a ComputeBlock (sequence of expression
// statements and reductions) that can be evaluated by the Kokkos
// expression engine.
//
// The Python function must follow these constraints:
//   - Named `compute`
//   - Parameters are input field names
//   - Body is a sequence of assignments: var = expression
//   - Supported: arithmetic (+,-,*,/,**), math functions, comparisons,
//     where(), col_sum(), col_avg(), horiz_avg()
//   - Must end with `return <variable>`
//   - No loops, no if/else blocks, no imports, no function calls
//     other than recognized builtins
//
// Example Python file:
//
//   def compute(qc, pseudo_density):
//       integrand = qc * pseudo_density / 9.80616
//       return col_sum(integrand)
//
// Translates to ComputeBlock:
//   statements:
//     [0] Assign: integrand = qc * pseudo_density / 9.80616
//     [1] ColSum: result = integrand
//
// This is a TEXT-BASED translator (not using Python's ast module).
// It works by parsing the indented function body line-by-line.
// For a full AST-based translator, use the Python-side pyeamxx_diag.py.
// ====================================================================

#include "compute_block.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace diag_utils {

struct PyTranslateResult {
  ComputeBlock          block;
  std::vector<std::string> warnings;
};

inline PyTranslateResult translate_py_file(
    const std::string& filepath,
    const std::string& output_name,
    const ConstResolver& resolver = nullptr)
{
  PyTranslateResult result;
  result.block.name = output_name;

  // Read the file
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error(
      "Cannot open Python diagnostic file: " + filepath);
  }

  std::string full_source;
  {
    std::ostringstream ss;
    ss << file.rdbuf();
    full_source = ss.str();
  }

  // --- Phase 1: Find the compute() function and extract parameters ---

  std::istringstream stream(full_source);
  std::string line;
  bool in_function = false;
  int func_indent = -1;
  std::vector<std::string> body_lines;

  while (std::getline(stream, line)) {
    // Strip trailing whitespace/CR
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
      line.pop_back();

    if (!in_function) {
      // Look for "def compute("
      auto def_pos = line.find("def compute(");
      if (def_pos != std::string::npos) {
        in_function = true;
        func_indent = static_cast<int>(def_pos);

        // Extract parameter names from "def compute(param1, param2, ...):"
        auto paren_open = line.find('(', def_pos);
        auto paren_close = line.find(')', paren_open);
        if (paren_open == std::string::npos || paren_close == std::string::npos) {
          throw std::runtime_error(
            "Python translator: malformed function signature in " + filepath);
        }
        std::string params_str = line.substr(paren_open + 1, paren_close - paren_open - 1);

        // Split by comma, strip whitespace and type hints
        std::istringstream pstream(params_str);
        std::string param;
        while (std::getline(pstream, param, ',')) {
          // Strip whitespace
          auto a = param.find_first_not_of(" \t");
          if (a == std::string::npos) continue;
          auto b = param.find_last_not_of(" \t");
          param = param.substr(a, b - a + 1);
          // Strip type hint (": Type")
          auto colon = param.find(':');
          if (colon != std::string::npos) {
            param = param.substr(0, colon);
            // Strip trailing whitespace after removing hint
            b = param.find_last_not_of(" \t");
            if (b != std::string::npos) param = param.substr(0, b + 1);
          }
          if (!param.empty()) {
            result.block.input_names.push_back(param);
          }
        }
      }
    } else {
      // Inside the function body
      if (line.empty()) continue;

      // Check indentation - if we've dedented back to function level or less, we're done
      auto first_char = line.find_first_not_of(" \t");
      if (first_char != std::string::npos && static_cast<int>(first_char) <= func_indent) {
        // We've left the function body (unless it's a blank line)
        if (line.substr(first_char, 3) != "def" && line.substr(first_char, 1) != "#") {
          break;  // New top-level code, stop
        }
        break;
      }

      // Skip comments and docstrings
      auto stripped = line.substr(first_char);
      if (stripped[0] == '#') continue;
      if (stripped.substr(0, 3) == "\"\"\"" || stripped.substr(0, 3) == "'''") {
        // Skip docstrings (single-line only for now)
        continue;
      }

      body_lines.push_back(stripped);
    }
  }

  if (!in_function) {
    throw std::runtime_error(
      "Python translator: no 'def compute(...)' function found in " + filepath);
  }
  if (body_lines.empty()) {
    throw std::runtime_error(
      "Python translator: empty function body in " + filepath);
  }

  // --- Phase 2: Translate each body line into a ComputeStatement ---

  for (size_t i = 0; i < body_lines.size(); ++i) {
    const auto& bline = body_lines[i];

    // Handle return statement (must be last)
    if (bline.substr(0, 7) == "return ") {
      if (i != body_lines.size() - 1) {
        result.warnings.push_back("return is not the last statement; ignoring remaining lines");
      }
      std::string ret_expr = bline.substr(7);
      // Strip whitespace
      auto a = ret_expr.find_first_not_of(" \t");
      if (a != std::string::npos) ret_expr = ret_expr.substr(a);

      ComputeStatement stmt;
      stmt.target_var = "result";

      // Check for reduction wrappers
      if (ret_expr.substr(0, 8) == "col_sum(" && ret_expr.back() == ')') {
        stmt.type = StmtType::ColSum;
        stmt.expression = ret_expr.substr(8, ret_expr.size() - 9);
      } else if (ret_expr.substr(0, 8) == "col_avg(" && ret_expr.back() == ')') {
        stmt.type = StmtType::ColAvg;
        stmt.expression = ret_expr.substr(8, ret_expr.size() - 9);
      } else if (ret_expr.substr(0, 10) == "horiz_avg(" && ret_expr.back() == ')') {
        stmt.type = StmtType::HorizAvg;
        stmt.expression = ret_expr.substr(10, ret_expr.size() - 11);
      } else {
        stmt.type = StmtType::Assign;
        stmt.expression = ret_expr;
      }

      stmt.parsed = resolver
                    ? parse_expression(stmt.expression, resolver)
                    : parse_expression(stmt.expression);
      result.block.statements.push_back(std::move(stmt));
      break;  // return ends the function
    }

    // Handle assignment: var = expression
    auto eq_pos = bline.find('=');
    if (eq_pos == std::string::npos) {
      throw std::runtime_error(
        "Python translator: unsupported statement (expected assignment): " + bline);
    }
    // Skip ==, !=, <=, >=
    if (eq_pos + 1 < bline.size() && bline[eq_pos + 1] == '=') {
      throw std::runtime_error(
        "Python translator: use eq()/ne() instead of ==/!= in: " + bline);
    }
    if (eq_pos > 0 && (bline[eq_pos-1] == '!' || bline[eq_pos-1] == '<' || bline[eq_pos-1] == '>')) {
      throw std::runtime_error(
        "Python translator: use comparison functions instead of operators in: " + bline);
    }

    // Translate Python operators to expression syntax
    std::string target = bline.substr(0, eq_pos);
    std::string rhs = bline.substr(eq_pos + 1);

    // Strip whitespace
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
        "Python translator: empty target or expression in: " + bline);
    }

    ComputeStatement stmt;
    stmt.target_var = target;

    // Check for reduction wrappers
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

    stmt.parsed = resolver
                  ? parse_expression(stmt.expression, resolver)
                  : parse_expression(stmt.expression);
    result.block.statements.push_back(std::move(stmt));
  }

  // Verify the last statement assigns to "result"
  if (result.block.statements.empty() ||
      result.block.statements.back().target_var != "result") {
    throw std::runtime_error(
      "Python translator: function must end with 'return <expr>' in " + filepath);
  }

  return result;
}

} // namespace diag_utils

#endif // KOKKOS_DIAG_PY_TRANSLATOR_HPP
