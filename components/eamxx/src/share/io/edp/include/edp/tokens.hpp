#ifndef EDP_TOKENS_HPP
#define EDP_TOKENS_HPP
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
namespace edp {

enum class TokenTypes {
  EndofFile,
  Illegal,
  Newline,

  Identifier,
  Integer,
  Float,
  String,
  // Operators
  Assign,
  Plus,
  Minus,
  Asterisk,
  Bang,
  Slash,
  Exp,
  Equal,
  GreaterThan,
  GreaterEqual,
  LessThan,
  LessEq,
  NotEqual,
  Or,
  And,
  Concat,
  Dot,

  // DELIMITERS
  Comma,
  LeftParen,
  RightParen,
  Colon,
  Semicolon,
  Percent,
  DoubleColon,
  ArrayLeftBracket,
  ArrayRightBracket,
};

std::string_view to_string(TokenTypes type);

// NOTE: `line` and `column` are 1-based and record where the token *starts*.
//       They carry default member initializers on purpose, so that the
//       aggregate initializations used everywhere ({TokenTypes::X, "lit"}) keep
//       compiling; a Token built that way simply claims to sit at 1:1.
struct Token {
  TokenTypes type = TokenTypes::Illegal;
  std::string literal;
  int line = 1;
  int column = 1;
};
std::string to_string(const Token& tok);

// NOTE: keywords are matched case-sensitively (Python keywords are lower-case
//       only). The lexer no longer lower-cases its input, since EAMxx field
//       names are case sensitive, so "AND" is an identifier, not an operator.
const std::unordered_map<std::string, Token> keywords{
    {"or", {TokenTypes::Or, "or"}},
    {"and", {TokenTypes::And, "and"}},
    {"not", {TokenTypes::Bang, "!"}},
};
Token identifier_lookup(const Token& tok);

std::ostream& operator<<(std::ostream& os, const Token& tok);
std::string binary_op_to_string(const TokenTypes type);
std::string unary_op_to_string(const TokenTypes type) ;

} // namespace edp
#endif
