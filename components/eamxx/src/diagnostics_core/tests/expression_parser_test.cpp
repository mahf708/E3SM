#include <catch2/catch.hpp>

#include "expression_parser.hpp"
#include "rpn_evaluator.hpp"

// ====================================================================
// Tests for the standalone expression parser + evaluator.
// These tests have NO EAMxx dependencies.
// ====================================================================

using namespace diag_utils;

TEST_CASE("standalone_parser_basics") {
  SECTION("simple_add") {
    auto r = parse_expression("a + b");
    REQUIRE(r.field_names.size() == 2);
    REQUIRE(r.field_names[0] == "a");
    REQUIRE(r.field_names[1] == "b");
    REQUIRE(r.program.size() == 3);
    REQUIRE(r.program[2].op == ExprOp::Add);
  }

  SECTION("precedence") {
    // a + b * c  →  RPN: a b c * +
    auto r = parse_expression("a + b * c");
    REQUIRE(r.program.size() == 5);
    REQUIRE(r.program[3].op == ExprOp::Mul);
    REQUIRE(r.program[4].op == ExprOp::Add);
  }

  SECTION("parentheses_override") {
    // (a + b) * c  →  RPN: a b + c *
    auto r = parse_expression("(a + b) * c");
    REQUIRE(r.program[2].op == ExprOp::Add);
    REQUIRE(r.program[4].op == ExprOp::Mul);
  }

  SECTION("power_right_assoc") {
    // a ** b ** c  →  a (b**c) **  (right-associative)
    auto r = parse_expression("a ** b ** c");
    // RPN: a b c ** **
    REQUIRE(r.program.size() == 5);
    REQUIRE(r.program[3].op == ExprOp::Pow);
    REQUIRE(r.program[4].op == ExprOp::Pow);
  }

  SECTION("unary_minus") {
    auto r = parse_expression("-a");
    REQUIRE(r.program.size() == 2);
    REQUIRE(r.program[1].op == ExprOp::Neg);
  }

  SECTION("double_neg") {
    auto r = parse_expression("--a");
    REQUIRE(r.program.size() == 3);
    REQUIRE(r.program[1].op == ExprOp::Neg);
    REQUIRE(r.program[2].op == ExprOp::Neg);
  }

  SECTION("function_sqrt") {
    auto r = parse_expression("sqrt(x)");
    REQUIRE(r.program.size() == 2);
    REQUIRE(r.program[1].op == ExprOp::Sqrt);
  }

  SECTION("function_abs") {
    auto r = parse_expression("abs(x)");
    REQUIRE(r.program[1].op == ExprOp::Abs);
  }

  SECTION("nested_functions") {
    auto r = parse_expression("sqrt(abs(x))");
    REQUIRE(r.program.size() == 3);
    REQUIRE(r.program[1].op == ExprOp::Abs);
    REQUIRE(r.program[2].op == ExprOp::Sqrt);
  }

  SECTION("binary_function_min") {
    auto r = parse_expression("min(a, b)");
    REQUIRE(r.program.size() == 3);
    REQUIRE(r.program[2].op == ExprOp::Min);
  }

  SECTION("binary_function_max") {
    auto r = parse_expression("max(a, b)");
    REQUIRE(r.program.size() == 3);
    REQUIRE(r.program[2].op == ExprOp::Max);
  }

  SECTION("numeric_literal") {
    auto r = parse_expression("a + 3.14");
    REQUIRE(r.field_names.size() == 1);
    REQUIRE(r.program[1].op == ExprOp::PushConst);
    REQUIRE(r.program[1].const_val == Approx(3.14));
  }

  SECTION("scientific_notation") {
    auto r = parse_expression("a * 1.5e-3");
    REQUIRE(r.program[1].op == ExprOp::PushConst);
    REQUIRE(r.program[1].const_val == Approx(1.5e-3));
  }

  SECTION("field_reuse") {
    auto r = parse_expression("a * a + a");
    REQUIRE(r.field_names.size() == 1);
  }

  SECTION("complex_wind_speed") {
    auto r = parse_expression("sqrt(u**2 + v**2)");
    REQUIRE(r.field_names.size() == 2);
    // RPN: u 2 ** v 2 ** + sqrt = 8 instructions
    REQUIRE(r.program.size() == 8);
  }

  SECTION("invalid_throws") {
    REQUIRE_THROWS(parse_expression("a + + b"));
    REQUIRE_THROWS(parse_expression("a @@ b"));
    REQUIRE_THROWS(parse_expression(""));
  }
}

TEST_CASE("standalone_parser_with_constants") {
  // Use a constant resolver
  ConstResolver resolver = [](const std::string& name, double& val) -> bool {
    if (name == "pi")    { val = 3.14159265358979; return true; }
    if (name == "g")     { val = 9.80616;          return true; }
    if (name == "two")   { val = 2.0;              return true; }
    return false;
  };

  SECTION("constant_resolved") {
    auto r = parse_expression("a / g", resolver);
    REQUIRE(r.field_names.size() == 1);
    REQUIRE(r.field_names[0] == "a");
    REQUIRE(r.program[1].op == ExprOp::PushConst);
    REQUIRE(r.program[1].const_val == Approx(9.80616));
  }

  SECTION("mixed_fields_and_constants") {
    auto r = parse_expression("two * pi * a", resolver);
    REQUIRE(r.field_names.size() == 1);
    // two and pi are constants, only 'a' is a field
    REQUIRE(r.program[0].op == ExprOp::PushConst);
    REQUIRE(r.program[0].const_val == Approx(2.0));
    REQUIRE(r.program[1].op == ExprOp::PushConst);
  }
}

TEST_CASE("standalone_rpn_evaluator") {
  // Host-side evaluation test
  // Expression: a * b + 3.0
  auto r = parse_expression("a * b + 3.0");

  // Create mock field data
  const int N = 10;
  double a_data[N], b_data[N], out_data[N];
  for (int i = 0; i < N; ++i) {
    a_data[i] = static_cast<double>(i + 1);
    b_data[i] = static_cast<double>(i + 1) * 0.5;
  }

  const double* field_ptrs[2] = {a_data, b_data};

  // Evaluate on host
  for (int i = 0; i < N; ++i) {
    out_data[i] = evaluate_rpn(r.program.data(),
                               static_cast<int>(r.program.size()),
                               field_ptrs, i);
  }

  // Verify
  for (int i = 0; i < N; ++i) {
    double expected = a_data[i] * b_data[i] + 3.0;
    REQUIRE(out_data[i] == Approx(expected));
  }
}

TEST_CASE("standalone_rpn_functions") {
  const int N = 5;
  double x_data[N] = {1.0, 4.0, 9.0, 16.0, 25.0};
  const double* ptrs[1] = {x_data};

  SECTION("sqrt") {
    auto r = parse_expression("sqrt(x)");
    for (int i = 0; i < N; ++i) {
      double result = evaluate_rpn(r.program.data(),
                                   static_cast<int>(r.program.size()), ptrs, i);
      REQUIRE(result == Approx(std::sqrt(x_data[i])));
    }
  }

  SECTION("square") {
    auto r = parse_expression("square(x)");
    for (int i = 0; i < N; ++i) {
      double result = evaluate_rpn(r.program.data(),
                                   static_cast<int>(r.program.size()), ptrs, i);
      REQUIRE(result == Approx(x_data[i] * x_data[i]));
    }
  }

  SECTION("power") {
    auto r = parse_expression("x ** 3");
    for (int i = 0; i < N; ++i) {
      double result = evaluate_rpn(r.program.data(),
                                   static_cast<int>(r.program.size()), ptrs, i);
      REQUIRE(result == Approx(std::pow(x_data[i], 3.0)));
    }
  }

  SECTION("negation") {
    auto r = parse_expression("-x");
    for (int i = 0; i < N; ++i) {
      double result = evaluate_rpn(r.program.data(),
                                   static_cast<int>(r.program.size()), ptrs, i);
      REQUIRE(result == Approx(-x_data[i]));
    }
  }

  SECTION("min_max") {
    double y_data[N] = {5.0, 2.0, 10.0, 8.0, 3.0};
    const double* ptrs2[2] = {x_data, y_data};

    auto r_min = parse_expression("min(x, y)");
    auto r_max = parse_expression("max(x, y)");
    for (int i = 0; i < N; ++i) {
      double rmin = evaluate_rpn(r_min.program.data(),
                                 static_cast<int>(r_min.program.size()), ptrs2, i);
      double rmax = evaluate_rpn(r_max.program.data(),
                                 static_cast<int>(r_max.program.size()), ptrs2, i);
      REQUIRE(rmin == Approx(std::min(x_data[i], y_data[i])));
      REQUIRE(rmax == Approx(std::max(x_data[i], y_data[i])));
    }
  }
}
