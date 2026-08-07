#pragma once // becoming lazy

#include <edp/tokens.hpp>
namespace edp::parser {

// NOTE: `Bounds` (the slice colon) used to sit between Prefix and Call, which
//       would have made ":" bind tighter than "-" or "+": "-1:2" would parse as
//       "-(1:2)" and "1+2:3" as "1+(2:3)". Python binds the slice colon looser
//       than every arithmetic/comparison operator, so Bounds moved down. It
//       still binds tighter than `Assign`, so that "lev=0:10" is the keyword
//       argument `lev` with a slice value rather than a slice of "(lev=0)".
enum class Precedence {
  Lowest,
  Equal,
  Bounds,
  LessGreater,
  Sum,
  Product,
  Prefix,
  Call,
};
Precedence token_precedence(TokenTypes type);
} // namespace edp::parser
