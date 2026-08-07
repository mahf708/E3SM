#include <edp/lexer.hpp>
#include <edp/tokens.hpp>
#include <cctype>
#include <cstddef>
#include <string>
#include <utility>

namespace {

// Python identifier rules: the first character may be a letter or an
// underscore, but *not* a digit; every subsequent character may additionally
// be a digit. EAMxx field names rely on this (`bc_a1`, `so4_a2`, `O3`).
bool is_identifier_start(const char ch) {
  return std::isalpha(static_cast<unsigned char>(ch)) || ch == '_';
}

bool is_identifier_char(const char ch) {
  return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
}

bool is_numeric(const char ch) {
  return std::isdigit(static_cast<unsigned char>(ch));
}

bool is_exponent_marker(const char ch) { return ch == 'e' || ch == 'E'; }

} // namespace

namespace edp {

// NOTE: the input is deliberately *not* lower-cased (upstream did). EAMxx
//       field names are case sensitive (`T_mid`, `LiqWaterPath`). Keywords are
//       therefore lower-case only, which matches Python.
Lexer::Lexer(std::string input)
    : input_{std::move(input)}, position_{0}, read_position_{0},
      current_char_{'\0'}, line_{1}, column_{1} {
  read_char();
}

void Lexer::read_char() {
  // Advance the 1-based source position *before* moving on, i.e. account for
  // the character we are about to leave behind. On the very first call there
  // is no such character yet, so 1:1 (the position of input_[0]) stands.
  if (read_position_ > 0) {
    if (current_char_ == '\n') {
      line_ += 1;
      column_ = 1;
    } else if (current_char_ != '\0') {
      column_ += 1;
    }
  }
  if (static_cast<std::size_t>(read_position_) >= input_.length()) {
    current_char_ = '\0';
  } else {
    current_char_ = input_.at(read_position_);
  }
  position_ = read_position_;
  read_position_ += 1;
}

char Lexer::peek_char() const {
  if (static_cast<std::size_t>(read_position_) >= input_.length()) {
    return '\0';
  } else {
    return input_[read_position_];
  }
}

void Lexer::skip_whitespace() {
  while (std::isspace(static_cast<unsigned char>(current_char_))) {
    read_char();
  }
}

Token Lexer::make_token(TokenTypes kind) const {
  return {kind, std::string(1, current_char_)};
}

// Reads up to (and consumes) the closing delimiter `ch`. If the input runs out
// first the scan stops at end-of-input rather than looping forever; the caller
// detects that case by checking that `current_char_` is the delimiter.
std::string Lexer::read_to_delim(char ch) {
  auto start_pos = position_+1;
  while (peek_char() != ch && peek_char() != '\0') {
    read_char();
  }
  read_char();
  auto count = position_ - start_pos;
  return input_.substr(start_pos, count);
}

std::string Lexer::read_number() {
  auto start_pos = position_;
  // At most one '.' belongs to the literal, so that "1.2.3" does not lex as a
  // single (silently truncated) number.
  bool seen_dot = false;
  while (is_numeric(current_char_) || (current_char_ == '.' && !seen_dot)) {
    if (current_char_ == '.') {
      seen_dot = true;
    }
    read_char();
  }
  check_precision();
  return input_.substr(start_pos, position_ - start_pos);
}

void Lexer::check_precision() {
  if (!is_exponent_marker(current_char_)) {
    return;
  }
  // An 'e'/'E' only belongs to the number when an optionally signed digit
  // sequence follows it. Upstream consumed it unconditionally (and then read
  // one character too many), so "1e" walked `position_` past the end of the
  // input and the substr in read_number threw std::out_of_range.
  auto idx = static_cast<std::size_t>(read_position_);
  if (idx < input_.length() && (input_[idx] == '+' || input_[idx] == '-')) {
    ++idx;
  }
  if (idx >= input_.length() || !is_numeric(input_[idx])) {
    return;
  }

  read_char(); // the exponent marker
  if (current_char_ == '+' || current_char_ == '-') {
    read_char();
  }
  while (is_numeric(current_char_)) {
    read_char();
  }
}

std::string Lexer::read_identifier() {
  auto start_pos = position_;
  // The caller only enters here on an identifier-start character; digits are
  // legal from the second character on.
  if (is_identifier_start(current_char_)) {
    read_char();
  }
  while (is_identifier_char(current_char_)) {
    read_char();
  }
  auto length = position_ - start_pos;
  return input_.substr(start_pos, length);
}

// Every token is stamped with the position of its *first* character, which is
// where the lexer sits after skipping whitespace and before scan_token()
// consumes anything. Doing the stamping in one place here keeps scan_token()'s
// many early returns from each having to remember to do it.
Token Lexer::next_token() {
  skip_whitespace();
  const int start_line = line_;
  const int start_column = column_;

  Token tok = scan_token();
  tok.line = start_line;
  tok.column = start_column;
  return tok;
}

Token Lexer::scan_token() {

  Token tok;

  switch (current_char_) {
  case '=':
    if (peek_char() == '=') {
      tok = {TokenTypes::Equal, "=="};
      read_char();
    } else {
      tok = make_token(TokenTypes::Assign);
    }
    break;
  case '(':
    tok = make_token(TokenTypes::LeftParen);
    break;
  case ')':
    tok = make_token(TokenTypes::RightParen);
    break;
  case ',':
    tok = make_token(TokenTypes::Comma);
    break;
  case '+':
    tok = make_token(TokenTypes::Plus);
    break;
  case '-':
    tok = make_token(TokenTypes::Minus);
    break;
  case '\n':
    tok = make_token(TokenTypes::Newline);
    break;
  case '[':
    tok = make_token(TokenTypes::ArrayLeftBracket);
    break;
  case ']':
    tok = make_token(TokenTypes::ArrayRightBracket);
    break;
  case '/':
    tok = make_token(TokenTypes::Slash);
    break;
  case '*':
    if (peek_char() == '*') {
      read_char();
      tok = {TokenTypes::Exp, "**"};
    } else {
      tok = make_token(TokenTypes::Asterisk);
    }
    break;
  case '!':
    if (peek_char() == '=') {
      read_char();
      tok = {TokenTypes::NotEqual, "!="};
    } else {
      tok = make_token(TokenTypes::Bang);
    }
    break;
  case '<':
    if (peek_char() == '=') {
      read_char();
      tok = {TokenTypes::LessEq, "<="};
    } else {
      tok = make_token(TokenTypes::LessThan);
    }
    break;

  case '>':
    if (peek_char() == '=') {
      read_char();
      tok = {TokenTypes::GreaterEqual, ">="};
    } else {
      tok = make_token(TokenTypes::GreaterThan);
    }
    break;
  case '\0':
    return {TokenTypes::EndofFile, ""};
  case ':':
    tok = make_token(TokenTypes::Colon);
    break;
  case '"':
  case '\'': {
    const char delim = current_char_;
    auto value = read_to_delim(delim);
    if (current_char_ != delim) {
      // Unterminated string literal: the lexer is now at end-of-input.
      return {TokenTypes::Illegal, value};
    }
    tok = {TokenTypes::String, std::move(value)};
    break;
  }
  case '.': {
    if (is_numeric(peek_char())) {
      read_char();
      auto number = read_number();
      number.insert(0, "0.");
      return {TokenTypes::Float, number};
    } else {
        tok = make_token(TokenTypes::Dot);
        break;
      }
  }
  default: {

    if (is_identifier_start(current_char_)) {
      return identifier_lookup({TokenTypes::Identifier, read_identifier()});
    } else if (is_numeric(current_char_)) {
      auto number = read_number();

      // A '.' or an exponent marker makes it a Float; anything else is an
      // Integer. (Upstream only looked for 'e', so "0.5" became an Integer
      // token and std::stoi silently truncated it to 0.)
      if (number.find_first_of(".eE") != std::string::npos) {
        return {TokenTypes::Float, number};
      } else {
        return {TokenTypes::Integer, number};
      }
    } else {
      // NOTE: consume the offending character, otherwise the lexer is stuck
      //       returning the same Illegal token forever.
      auto illegal = make_token(TokenTypes::Illegal);
      read_char();
      return illegal;
    }

  } // default
  } // end switch
  read_char();
  return tok;
}

} // namespace edp
