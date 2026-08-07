#include <edp/ast.hpp>
#include <edp/tokens.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>

/**
 * @file ast_print.cpp
 * @brief Implementation of Vistors for printing AST nodes
 */
namespace edp::ast {

namespace {

struct ToStringVisitor {

  std::string operator()(const Identifier& expr) const;
  std::string operator()(const PrefixExpression& expr) const;
  std::string operator()(const InfixExpression& expr) const;
  std::string operator()(const FuncExpression& expr) const;
  std::string operator()(const SliceExpression& expr) const;
  std::string operator()(const ArrayExpression& expr) const;
  std::string operator()(const StringLiteral& expr) const;
  std::string operator()(const FloatLiteral& expr) const;
  std::string operator()(const IntegerLiteral& expr) const;
};

// A slice prints *without* enclosing parentheses ("1:10"), because that is the
// spelling the parser accepts and the one users write. That is only safe as
// long as nothing around it can steal the colon back when the text is
// re-parsed -- and `to_string` output is re-parsed, it is the canonical
// identity string for an expression. So any position where a colon would
// re-associate (an operand of a unary/binary operator, or a component of
// another slice) parenthesizes a slice child. Contexts that are already
// delimited -- call arguments, array elements -- do not need to.
struct IsSliceVisitor {
  template <typename T> bool operator()(const T&) const { return false; }
  bool operator()(const SliceExpression&) const { return true; }
};

std::string operand_to_string(const Expression& expr) {
  if (expr.visit(IsSliceVisitor{})) {
    return "(" + to_string(expr) + ")";
  }
  return to_string(expr);
}

std::string expr_list_to_string(std::span<const ExprPtr> vals) {

  std::string result;
  bool first = true;

  std::ranges::for_each(vals, [&](const ExprPtr& val) {
    if (!first) {
      result += ", ";
    }
    first = false;
    result += to_string(*val);
  });
  return result;
}

std::string ToStringVisitor::operator()(const Identifier& expr) const {
  return expr.value;
};

std::string ToStringVisitor::operator()(const PrefixExpression& expr) const {
  return "(" + unary_op_to_string(expr.op) + operand_to_string(*expr.right) +
         ")";
};

std::string ToStringVisitor::operator()(const InfixExpression& expr) const {
  return "(" + operand_to_string(*expr.left) + binary_op_to_string(expr.op) +
         operand_to_string(*expr.right) + ")";
};
std::string ToStringVisitor::operator()(const FuncExpression& expr) const {
  return operand_to_string(*expr.function) + "(" +
         expr_list_to_string(expr.args) + ")";
};

// Python spelling: "1:10", ":10", "1:", "::2", "1:10:2", and a bare ":" for a
// full slice. An omitted component contributes nothing but its colon.
std::string ToStringVisitor::operator()(const SliceExpression& expr) const {
  std::string result;
  if (expr.start) {
    result += operand_to_string(*expr.start);
  }
  result += ":";
  if (expr.stop) {
    result += operand_to_string(*expr.stop);
  }
  if (expr.step) {
    result += ":";
    result += operand_to_string(*expr.step);
  }
  return result;
};

std::string ToStringVisitor::operator()(const ArrayExpression& expr) const {
  return +"[" + expr_list_to_string(expr.elements) + "]";
};

std::string ToStringVisitor::operator()(const StringLiteral& expr) const {
  return "'" + expr.value + "'";
};
std::string ToStringVisitor::operator()(const IntegerLiteral& expr) const {
  return std::to_string(expr.value);
};
std::string ToStringVisitor::operator()(const FloatLiteral& expr) const {
  // Print the shortest of these formats that reads back as the same double.
  // NOTE: neither std::format nor floating-point std::to_chars can be used
  //       here -- both are missing from stdlib versions EAMxx must build
  //       against -- hence snprintf + strtod.
  // NOTE: upstream printed "%e" from a float, so "500.0" round-tripped as
  //       "5.000000e+02".
  char buf[64];
  for (const char* fmt : {"%g", "%.9g", "%.16g", "%.17g"}) {
    std::snprintf(buf, sizeof(buf), fmt, expr.value);
    if (std::strtod(buf, nullptr) == expr.value) {
      break;
    }
  }
  std::string result{buf};
  // Keep the output lexable as a Float rather than an Integer ("500" -> "500").
  if (result.find_first_of(".eEnif") == std::string::npos) {
    result += ".0";
  }
  return result;
};

} // namespace

std::string to_string(const Expression& expr) {
  return expr.visit(ToStringVisitor{});
}

} // namespace edp::ast
