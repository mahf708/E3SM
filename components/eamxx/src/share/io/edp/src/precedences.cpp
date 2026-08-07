#include <edp/tokens.hpp>
#include <edp/precedences.hpp>
#include <stdexcept>

namespace edp::parser {


  // Lowest,
  // Equal,
  // Bounds,
  // LessGreater,
  // Sum,
  // Product,
  // Prefix,
  // Call,

Precedence token_precedence(TokenTypes type) {
  switch (type) {

  case TokenTypes::Equal:
  case TokenTypes::NotEqual:
  case TokenTypes::Assign:
  case TokenTypes::And:
  case TokenTypes::Or:
    return Precedence::Equal;

  case TokenTypes::GreaterThan:
  case TokenTypes::GreaterEqual:
  case TokenTypes::LessThan:
  case TokenTypes::LessEq:
    return Precedence::LessGreater;

  case TokenTypes::Plus:
  case TokenTypes::Minus:
    return Precedence::Sum;

  case TokenTypes::Slash:
  case TokenTypes::Asterisk:
  case TokenTypes::Exp:
    return Precedence::Product;

  case TokenTypes::Bang:
  case TokenTypes::Dot:
    return Precedence::Prefix;

  case TokenTypes::Colon:
    return Precedence::Bounds;
  case TokenTypes::LeftParen:
    return Precedence::Call;

  // Every other token type is non-operator, and falls through to Lowest.
  // NOTE: explicit default (vs upstream) only to silence -Wswitch; behavior
  //       is unchanged.
  default:
    break;
  }
  return Precedence::Lowest;
}

}

