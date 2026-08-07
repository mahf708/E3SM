#ifndef EDP_LEXER_HPP
#define EDP_LEXER_HPP

#include <edp/tokens.hpp>
#include <string>

namespace edp {

class Lexer {

public:
  explicit Lexer(std::string input);
  ~Lexer() = default;

  Token next_token();

  // The text this lexer was constructed with. The parser needs it to render
  // the offending source line under an error message.
  const std::string& input() const noexcept { return input_; }

private:
  std::string input_;
  int position_;
  int read_position_;
  char current_char_;
  // 1-based position of `current_char_` within `input_`.
  int line_;
  int column_;

  // functions
  void skip_whitespace();
  Token scan_token();

  std::string read_identifier();
  std::string read_number();
  std::string read_to_delim(char ch);
  void check_precision();

  char peek_char() const;
  void read_char();
  Token make_token(TokenTypes kind) const;

};

}

#endif
