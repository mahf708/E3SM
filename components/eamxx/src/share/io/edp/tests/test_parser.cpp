// NOTE: upstream uses Catch2 v3 headers (<catch2/catch_test_macros.hpp> and
//       "catch2/catch_message.hpp"); EAMxx vendors Catch2 v2, whose single
//       header provides the same macros.
#include "catch2/catch.hpp"

#include <edp/ast.hpp>
#include <edp/lexer.hpp>
#include <edp/parser.hpp>
#include <edp/tokens.hpp>

#include <string>

namespace edp {

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

} // namespace edp
