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
  std::string operator()(const ArrayExpression& expr) const;
  std::string operator()(const StringLiteral& expr) const;
  std::string operator()(const FloatLiteral& expr) const;
  std::string operator()(const IntegerLiteral& expr) const;
};

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
  return "(" + unary_op_to_string(expr.op) + to_string(*expr.right) + ")";
};

std::string ToStringVisitor::operator()(const InfixExpression& expr) const {
  return "(" + to_string(*expr.left) + binary_op_to_string(expr.op) +
         to_string(*expr.right) + ")";
};
std::string ToStringVisitor::operator()(const FuncExpression& expr) const {
  return to_string(*expr.function) + "(" + expr_list_to_string(expr.args) + ")";
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
