// NOTE: upstream uses Catch2 v3 headers (<catch2/catch_test_macros.hpp>);
//       EAMxx vendors Catch2 v2, whose single header provides the same macros.
#include "catch2/catch.hpp"

#include <edp/tokens.hpp>
#include <edp/lexer.hpp>

#include <string>
#include <vector>

namespace edp {

namespace {

// Lex an entire input. The token cap is a guard against a lexer that fails to
// make progress (see the "Illegal tokens make progress" test).
std::vector<Token> lex_all(const std::string& input, int max_tokens = 64) {
  Lexer lexer{input};
  std::vector<Token> tokens;
  for (int i = 0; i < max_tokens; ++i) {
    auto tok = lexer.next_token();
    const bool eof = tok.type == TokenTypes::EndofFile;
    tokens.push_back(std::move(tok));
    if (eof) {
      break;
    }
  }
  return tokens;
}

void check_tokens(const std::string& input,
                  const std::vector<Token>& expected) {
  const auto actual = lex_all(input);
  INFO("Input: " << input);
  REQUIRE(actual.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    INFO("Token " << i << "\nExpected: " << to_string(expected[i])
                  << "\nReceived: " << to_string(actual[i]));
    CHECK(expected[i].type == actual[i].type);
    CHECK(expected[i].literal == actual[i].literal);
  }
}

} // namespace

TEST_CASE("lexer token stream") {
  Lexer lexer{" not x <= 1.0e-4 and y + 5=1"};

  const std::vector<Token> expected{
      {TokenTypes::Bang, "!"},     {TokenTypes::Identifier, "x"},
      {TokenTypes::LessEq, "<="},  {TokenTypes::Float, "1.0e-4"},
      {TokenTypes::And, "and"},    {TokenTypes::Identifier, "y"},
      {TokenTypes::Plus, "+"},     {TokenTypes::Integer, "5"},
      {TokenTypes::Assign, "="},   {TokenTypes::Integer, "1"},
      {TokenTypes::EndofFile, ""},
  };
  for (const auto& expected_token : expected) {
    auto my_token = lexer.next_token();
    INFO("Expected: " << to_string(expected_token)
                      << "\nReceived: " << to_string(my_token));
    CHECK(expected_token.type == my_token.type);
    CHECK(expected_token.literal == my_token.literal);
  }
}

// Regression: the lexer used to lower-case its whole input, which destroyed
// EAMxx field names ("T_mid" -> "t_mid").
TEST_CASE("identifiers keep their case") {
  for (const std::string name :
       {"T_mid", "LiqWaterPath", "AeroComCldTop", "qv", "SW_flux_dn"}) {
    check_tokens(name, {{TokenTypes::Identifier, name},
                        {TokenTypes::EndofFile, ""}});
  }
}

// Regression: keywords are lower-case only (Python semantics). Now that the
// input is no longer lower-cased, "AND" must lex as an identifier.
TEST_CASE("keywords are lower-case only") {
  check_tokens("and", {{TokenTypes::And, "and"}, {TokenTypes::EndofFile, ""}});
  check_tokens("not", {{TokenTypes::Bang, "!"}, {TokenTypes::EndofFile, ""}});
  check_tokens("or", {{TokenTypes::Or, "or"}, {TokenTypes::EndofFile, ""}});
  check_tokens("AND",
               {{TokenTypes::Identifier, "AND"}, {TokenTypes::EndofFile, ""}});
  check_tokens("Not",
               {{TokenTypes::Identifier, "Not"}, {TokenTypes::EndofFile, ""}});
}

// Regression: identifiers used to exclude digits, so "bc_a1" lexed as the two
// tokens `bc_a` and `1`, and "so4_a2" as four tokens.
TEST_CASE("identifiers may contain digits") {
  for (const std::string name :
       {"bc_a1", "so4_a2", "num_a3", "O3", "H2O2", "dst_a1", "x1"}) {
    check_tokens(name, {{TokenTypes::Identifier, name},
                        {TokenTypes::EndofFile, ""}});
  }
}

// ... but an identifier may not *start* with a digit: numbers still lex as
// numbers.
TEST_CASE("numbers are not identifiers") {
  check_tokens("500",
               {{TokenTypes::Integer, "500"}, {TokenTypes::EndofFile, ""}});
  check_tokens("1e-5",
               {{TokenTypes::Float, "1e-5"}, {TokenTypes::EndofFile, ""}});
  check_tokens("1E-5",
               {{TokenTypes::Float, "1E-5"}, {TokenTypes::EndofFile, ""}});
  check_tokens("0.5",
               {{TokenTypes::Float, "0.5"}, {TokenTypes::EndofFile, ""}});
  check_tokens(".5",
               {{TokenTypes::Float, "0.5"}, {TokenTypes::EndofFile, ""}});
  // Regression: "500.0" used to lex as an *Integer* token, and std::stoi then
  // silently truncated the fractional part.
  check_tokens("500.0",
               {{TokenTypes::Float, "500.0"}, {TokenTypes::EndofFile, ""}});
  // A digit-led run followed by letters is a number then an identifier, never
  // one identifier.
  check_tokens("500hPa", {{TokenTypes::Integer, "500"},
                          {TokenTypes::Identifier, "hPa"},
                          {TokenTypes::EndofFile, ""}});
  // A dangling exponent marker is not part of the number: upstream consumed
  // it (and one character past the end of the input) and threw out of substr.
  check_tokens("1e", {{TokenTypes::Integer, "1"},
                      {TokenTypes::Identifier, "e"},
                      {TokenTypes::EndofFile, ""}});
  check_tokens("1e+", {{TokenTypes::Integer, "1"},
                       {TokenTypes::Identifier, "e"},
                       {TokenTypes::Plus, "+"},
                       {TokenTypes::EndofFile, ""}});
  check_tokens("1.5e-8",
               {{TokenTypes::Float, "1.5e-8"}, {TokenTypes::EndofFile, ""}});
  check_tokens("2e3 + 1", {{TokenTypes::Float, "2e3"},
                           {TokenTypes::Plus, "+"},
                           {TokenTypes::Integer, "1"},
                           {TokenTypes::EndofFile, ""}});
}

// Regression: '!' had no case in the lexer, so NotEqual could never be
// produced and "qc != 0" yielded an Illegal token.
TEST_CASE("bang and not-equal") {
  check_tokens("qc != 0", {{TokenTypes::Identifier, "qc"},
                           {TokenTypes::NotEqual, "!="},
                           {TokenTypes::Integer, "0"},
                           {TokenTypes::EndofFile, ""}});
  check_tokens("!x", {{TokenTypes::Bang, "!"},
                      {TokenTypes::Identifier, "x"},
                      {TokenTypes::EndofFile, ""}});
}

// Regression: an Illegal token did not consume the offending character, so a
// "lex until EndofFile" loop never terminated.
TEST_CASE("Illegal tokens make progress") {
  check_tokens("T @ x", {{TokenTypes::Identifier, "T"},
                         {TokenTypes::Illegal, "@"},
                         {TokenTypes::Identifier, "x"},
                         {TokenTypes::EndofFile, ""}});
}

// Regression: an unterminated string literal looped forever.
TEST_CASE("unterminated string literal terminates") {
  check_tokens("'lev", {{TokenTypes::Illegal, "lev"},
                        {TokenTypes::EndofFile, ""}});
  check_tokens("dim='lev", {{TokenTypes::Identifier, "dim"},
                            {TokenTypes::Assign, "="},
                            {TokenTypes::Illegal, "lev"},
                            {TokenTypes::EndofFile, ""}});
  check_tokens("'lev'",
               {{TokenTypes::String, "lev"}, {TokenTypes::EndofFile, ""}});
}

TEST_CASE("realistic field expressions lex") {
  check_tokens("T_mid.isel(lev=-1)", {{TokenTypes::Identifier, "T_mid"},
                                      {TokenTypes::Dot, "."},
                                      {TokenTypes::Identifier, "isel"},
                                      {TokenTypes::LeftParen, "("},
                                      {TokenTypes::Identifier, "lev"},
                                      {TokenTypes::Assign, "="},
                                      {TokenTypes::Minus, "-"},
                                      {TokenTypes::Integer, "1"},
                                      {TokenTypes::RightParen, ")"},
                                      {TokenTypes::EndofFile, ""}});
}
} // namespace edp
