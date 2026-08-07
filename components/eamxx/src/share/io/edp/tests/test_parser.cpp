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

// The message of the ParserError a failing input produces.
std::string parse_error(const std::string& input) {
  parser::Parser parser{Lexer{input}};
  try {
    auto expr = parser.parse();
  } catch (const parser::ParserError& e) {
    return e.what();
  }
  FAIL("Expected a ParserError for input: " << input);
  return {};
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
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

// Regression: errors used to name the offending token but not where it was.
TEST_CASE("errors carry line, column and a caret snippet") {
  {
    const auto msg = parse_error("T_mid + @foo");
    INFO(msg);
    // '@' is the 9th character.
    CHECK(contains(msg, "line 1, column 9: Illegal token in input: '@'"));
    // ... and the caret sits under it, on the line below the source snippet.
    CHECK(contains(msg, "\n      T_mid + @foo\n              ^"));
  }
  {
    // Multi-line input reports the offending line only, with a reset column.
    const auto msg = parse_error("T_mid +\n  q @ 3");
    INFO(msg);
    CHECK(contains(msg, "line 2, column 5:"));
    CHECK(contains(msg, "\n        q @ 3\n          ^"));
    CHECK(!contains(msg, "T_mid +\n"));
  }
  {
    // An unterminated string points at the quote that opened it.
    const auto msg = parse_error("T_mid.mean(dim='lev");
    INFO(msg);
    CHECK(contains(msg, "line 1, column 16:"));
  }
  {
    // An empty input has no line to show; the prefix must still be there and
    // the caret rendering must not read past the end of the string.
    const auto msg = parse_error("");
    INFO(msg);
    CHECK(contains(msg, "line 1, column 1:"));
  }
  {
    // A token at end-of-input parks the caret just past the last character.
    const auto msg = parse_error("(T + q");
    INFO(msg);
    CHECK(contains(msg, "line 1, column 7:"));
  }
}

// Regression: `=` is registered as an ordinary infix operator so that it can
// spell a keyword argument, which also made "a = b = c" parse happily.
TEST_CASE("chained assignment is rejected") {
  // The forms that must keep working.
  CHECK(parse_to_string("dim='lev'") == "(dim='lev')");
  CHECK(parse_to_string("f(a=1, b=2)") == "f((a=1), (b=2))");
  CHECK(parse_to_string("T_mid.mean(dim='lev', skipna=1)") ==
        "(T_mid.mean((dim='lev'), (skipna=1)))");

  for (const std::string input :
       {"a = b = c", "a = (b = c)", "a = b = c = d", "f(a=1, b=2=3)"}) {
    INFO("Input: " << input);
    parser::Parser parser{Lexer{input}};
    CHECK_THROWS_AS(parser.parse(), parser::ParserError);
  }
  CHECK(contains(parse_error("a = b = c"), "Chained assignment is not allowed"));

  // Equal (==) is a different operator and is deliberately left alone.
  CHECK_NOTHROW(parse_to_string("a == b == c"));
}

// Regression: TokenTypes::Colon had a precedence but no parse function, so
// "0:10" could not parse at all.
TEST_CASE("colon slices parse and print in Python form") {
  CHECK(parse_to_string("1:10") == "1:10");
  CHECK(parse_to_string(":10") == ":10");
  CHECK(parse_to_string("1:") == "1:");
  CHECK(parse_to_string("::2") == "::2");
  CHECK(parse_to_string("1:10:2") == "1:10:2");
  CHECK(parse_to_string(":10:2") == ":10:2");
  CHECK(parse_to_string(":") == ":");

  // An omitted component is omitted, not zero: ":" and "::" are the same
  // slice, and so are "1:" and "1::".
  CHECK(parse_to_string("::") == ":");
  CHECK(parse_to_string("1::") == "1:");
  CHECK(parse_to_string("1:10:") == "1:10");

  // The colon binds looser than every arithmetic/comparison operator (Python
  // semantics), so a leading minus belongs to the bound, not to the slice.
  CHECK(parse_to_string("-1:2") == "(-1):2");
  CHECK(parse_to_string("1:-1") == "1:(-1)");
  CHECK(parse_to_string("1+2:3") == "(1+2):3");
  CHECK(parse_to_string("1:2+3") == "1:(2+3)");

  // ... but tighter than '=', so a slice-valued keyword argument works.
  CHECK(parse_to_string("T_mid.isel(lev=0:10)") ==
        "(T_mid.isel((lev=(0:10))))");
  CHECK(parse_to_string("T_mid.isel(lev=::2)") == "(T_mid.isel((lev=(::2))))");
  CHECK(parse_to_string("T_mid.isel(lev=:)") == "(T_mid.isel((lev=(:))))");
  CHECK(parse_to_string("[1:2, 3, ::2]") == "[1:2, 3, ::2]");

  // slice(1,10) is an ordinary function call and stays valid.
  CHECK(parse_to_string("slice(1,10)") == "slice(1, 10)");

  // A fourth component is a syntax error, not a silently dropped one.
  parser::Parser parser{Lexer{"1:2:3:4"}};
  CHECK_THROWS_AS(parser.parse(), parser::ParserError);
  CHECK(contains(parse_error("1:2:3:4"), "Too many ':' in slice"));

  // A stray colon must produce a clean error, not a crash or a truncation.
  for (const std::string input : {"f(1, :, )", "1: @", "(1:2"}) {
    INFO("Input: " << input);
    parser::Parser bad{Lexer{input}};
    CHECK_THROWS_AS(bad.parse(), parser::ParserError);
  }
}

// CRITICAL: to_string() output is the canonical identity string for an
// expression and is re-parsed, so canonical(canonical(s)) must equal
// canonical(s) for everything the parser accepts.
TEST_CASE("to_string round-trips (canonical form is a fixed point)") {
  for (const std::string input :
       {"T_mid", "T_mid.weighted('dp').mean(dim='lev')", "T_mid.isel(lev=-1)",
        "(A - B) / C", "A - B / C", "T.where(qc > 0.5 and ni >= 1e-5)",
        "T.histogram(bins=[0,1,2])", "T.interp(plev=500.0, units='hPa')",
        "abs(X)", "(A + B) / C", "-T", "not x", "bc_a1 + so4_a2",
        // every slice form
        "1:10", ":10", "1:", "::2", "1:10:2", ":10:2", ":", "::", "1::",
        "1:10:", "-1:2", "1:-1", "1+2:3", "1:2+3", "1:(2:3)",
        "T_mid.isel(lev=0:10)", "T_mid.isel(lev=:10)", "T_mid.isel(lev=1:)",
        "T_mid.isel(lev=::2)", "T_mid.isel(lev=:)", "[1:2, 3, ::2]",
        "(1:2)*3", "-(1:2)", "f(1:)", "f(:)", "slice(1,10)",
        "T_mid.isel(lev=1:10:2).mean(dim='ncol')"}) {
    INFO("Input: " << input);
    const auto once = parse_to_string(input);
    INFO("canonical: " << once);
    const auto twice = parse_to_string(once);
    CHECK(once == twice);
    CHECK(parse_to_string(twice) == once);
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
