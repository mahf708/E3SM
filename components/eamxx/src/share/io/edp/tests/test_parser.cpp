// NOTE: upstream uses Catch2 v3 headers (<catch2/catch_test_macros.hpp> and
//       "catch2/catch_message.hpp"); EAMxx vendors Catch2 v2, whose single
//       header provides the same macros.
#include "catch2/catch.hpp"

#include <edp/ast.hpp>
#include <edp/lexer.hpp>
#include <edp/parser.hpp>
#include <edp/tokens.hpp>

#include <cstdlib>
#include <string>

namespace edp {

namespace {

std::string parse_to_string(const std::string& input) {
  parser::Parser parser{Lexer{input}};
  auto expr = parser.parse();
  REQUIRE(expr != nullptr);
  return ast::to_string(*expr);
}

} // namespace

TEST_CASE("Test Parse expressions") {
  std::string input = "x*y.derivative(dx=dy,['col']).where(x>0)";

  parser::Parser parser{Lexer{input}};

  auto expr = parser.parse();
  auto str_ = to_string(*expr);
  INFO("Parsed Expression: \n" << str_);
  CHECK(str_ == "(x*((y.derivative((dx=dy), ['col'])).where((x>0))))");
}

TEST_CASE("Parse errors are catchable") {
  // A bare infix operator has no prefix parse fn, so parsing must fail.
  // The failure must surface as a std::exception, not a bare std::string.
  parser::Parser parser{Lexer{"* x"}};
  CHECK_THROWS_AS(parser.parse(), std::exception);
}

// Regression: parse_prefix_expression never advanced past the operator, so it
// re-dispatched the same token and recursed until the stack overflowed.
TEST_CASE("prefix expressions terminate") {
  CHECK(parse_to_string("-T") == "(-T)");
  CHECK(parse_to_string("not x") == "(!x)");
  CHECK(parse_to_string("!x") == "(!x)");
  CHECK(parse_to_string("-1") == "(-1)");
  CHECK(parse_to_string("a * -b") == "(a*(-b))");
}

// Regression: parse_grouped_expression existed but was never registered as a
// prefix parse fn, so no parenthesized expression could parse at all.
TEST_CASE("parentheses group") {
  CHECK(parse_to_string("(T + q) * 2") == "((T+q)*2)");
  CHECK(parse_to_string("T + q * 2") == "(T+(q*2))");
  CHECK(parse_to_string("(T + q) * 2") != parse_to_string("T + q * 2"));
  CHECK(parse_to_string("(A + B) / C") == "((A+B)/C)");
  // A left paren in infix position is still a function call.
  CHECK(parse_to_string("abs(X)") == "abs(X)");
  CHECK(parse_to_string("abs(A - B) / C") == "(abs((A-B))/C)");
}

// Regression: the lexer lower-cased its input, so every field name was
// mangled.
TEST_CASE("field names keep their case and digits") {
  CHECK(parse_to_string("T_mid") == "T_mid");
  CHECK(parse_to_string("LiqWaterPath") == "LiqWaterPath");
  CHECK(parse_to_string("AeroComCldTop") == "AeroComCldTop");
  CHECK(parse_to_string("bc_a1 + so4_a2 + num_a3") ==
        "((bc_a1+so4_a2)+num_a3)");
  CHECK(parse_to_string("O3") == "O3");
}

// Regression: '!' was not lexed, so "!=" was impossible.
TEST_CASE("comparison operators parse and print") {
  CHECK(parse_to_string("qc != 0") == "(qc!=0)");
  CHECK(parse_to_string("qc == 0") == "(qc==0)");
  // These four were registered as infix operators but missing from
  // binary_op_to_string, so printing a valid AST threw.
  CHECK(parse_to_string("qc >= 1") == "(qc>=1)");
  CHECK(parse_to_string("qc <= 1") == "(qc<=1)");
  CHECK(parse_to_string("a and b") == "(a and b)");
  CHECK(parse_to_string("a or b") == "(a or b)");
}

// Regression: trailing/illegal input was silently dropped -- "bc_a1@" parsed
// to "bc_a", "T @ x" to "T" -- with only a message on std::cout.
TEST_CASE("garbage input throws instead of truncating") {
  for (const std::string input :
       {"bc_a1@", "T @ x", "T_mid #", "T_mid q", "T_mid 500", "T_mid.",
        "T_mid.mean(dim='lev", "(T + q", "1.2.3", "1e", "9999999999999"}) {
    INFO("Input: " << input);
    parser::Parser parser{Lexer{input}};
    CHECK_THROWS_AS(parser.parse(), parser::ParserError);
  }
}

// Regression: FloatLiteral held a float and was printed with "%e", so "500.0"
// round-tripped as "5.000000e+02".
TEST_CASE("float literals round-trip") {
  CHECK(parse_to_string("500.0") == "500.0");
  CHECK(parse_to_string("0.5") == "0.5");
  CHECK(parse_to_string("1e-5") == "1e-05");
  CHECK(parse_to_string("3.141592653589793") == "3.141592653589793");
  CHECK(parse_to_string("500") == "500"); // still an Integer

  // Whatever the format, the printed text must read back as the same double.
  for (const std::string input :
       {"0.1", "1e-5", "500.0", "2.5e-8", "1e300", "3.141592653589793",
        "1.0000000000000002"}) {
    INFO("Input: " << input);
    const auto printed = parse_to_string(input);
    CHECK(std::strtod(printed.c_str(), nullptr) ==
          std::strtod(input.c_str(), nullptr));
  }
}

// The expressions this parser exists to handle.
TEST_CASE("realistic diagnostic expressions") {
  CHECK(parse_to_string("T_mid.weighted('dp').mean(dim='lev')") ==
        "((T_mid.weighted('dp')).mean((dim='lev')))");
  CHECK(parse_to_string("T_mid.isel(lev=-1)") == "(T_mid.isel((lev=(-1))))");
  CHECK(parse_to_string("T_mid.where(qc > 1e-5)") ==
        "(T_mid.where((qc>1e-05)))");
  CHECK(parse_to_string("(A + B) / C") == "((A+B)/C)");
  CHECK(parse_to_string("T_mid.where(qc > 1e-5 and qc < 1)") ==
        "(T_mid.where(((qc>1e-05) and (qc<1))))");
}

} // namespace edp
