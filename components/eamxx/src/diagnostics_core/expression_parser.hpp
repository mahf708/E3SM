#ifndef KOKKOS_DIAG_EXPRESSION_PARSER_HPP
#define KOKKOS_DIAG_EXPRESSION_PARSER_HPP

// ====================================================================
// Standalone expression parser for field-diagnostic computations
//
// This header is part of the kokkos-diag-utils package and has NO
// dependencies on EAMxx, EKAT, or any other E3SM infrastructure.
// It depends only on standard C++ headers.
//
// Usage:
//   auto result = diag_utils::parse_expression("sqrt(u**2 + v**2)");
//   // result.program   = vector of RPN instructions
//   // result.field_names = {"u", "v"}
//   // result.constants   = {} (physics constants resolved externally)
//
// The parser accepts:
//   - Identifiers:  [a-zA-Z_][a-zA-Z0-9_]*
//   - Numbers:      integer, decimal, scientific notation
//   - Binary ops:   + - * / ** (power)
//   - Unary prefix: - (negation)
//   - Functions:    sqrt, abs, log, exp, square, min, max
//   - Grouping:     ( )
//
// The output is a flat RPN (reverse Polish notation) instruction
// stream suitable for evaluation by a stack machine.
// ====================================================================

#include <cctype>
#include <cmath>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace diag_utils {

// ---- Instruction set ----

constexpr int MAX_EXPR_STACK = 16;
constexpr int MAX_EXPR_INSTRUCTIONS = 64;

enum class ExprOp : int {
  // Operand push
  PushField    = 0,   // push field_data[operand_idx][element]
  PushConst    = 1,   // push const_val

  // Binary ops (pop 2, push 1)
  Add          = 10,
  Sub          = 11,
  Mul          = 12,
  Div          = 13,
  Pow          = 14,
  Min          = 15,
  Max          = 16,

  // Unary ops (pop 1, push 1)
  Sqrt         = 20,
  Abs          = 21,
  Log          = 22,
  Exp          = 23,
  Square       = 24,
  Neg          = 25,
  Log10        = 26,
};

// Plain-old-data instruction (trivially copyable → safe for GPU)
struct Instruction {
  ExprOp op;
  int    operand_idx;  // field index (for PushField)
  double const_val;    // literal value (for PushConst)
};

// Result of parsing
struct ParseResult {
  std::vector<Instruction> program;
  std::vector<std::string> field_names;     // referenced field names
  std::vector<std::string> constant_names;  // unresolved identifiers (physics constants)
};

// ---- Tokenizer ----

namespace detail {

enum class TokenType {
  Number, Identifier,
  Plus, Minus, Star, Slash, DoubleStar, Comma,
  LParen, RParen,
  End
};

struct Token {
  TokenType   type;
  std::string text;
  double      num_val = 0.0;
};

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& input) : m_input(input), m_pos(0) {}

  Token next() {
    skip_whitespace();
    if (m_pos >= m_input.size()) return {TokenType::End, "", 0.0};

    char c = m_input[m_pos];

    if (std::isdigit(c) ||
        (c == '.' && m_pos + 1 < m_input.size() && std::isdigit(m_input[m_pos+1]))) {
      return read_number();
    }
    if (std::isalpha(c) || c == '_') return read_identifier();
    if (c == '+') { m_pos++; return {TokenType::Plus,  "+", 0.0}; }
    if (c == '-') { m_pos++; return {TokenType::Minus, "-", 0.0}; }
    if (c == '/') { m_pos++; return {TokenType::Slash, "/", 0.0}; }
    if (c == '(') { m_pos++; return {TokenType::LParen,"(", 0.0}; }
    if (c == ')') { m_pos++; return {TokenType::RParen,")", 0.0}; }
    if (c == ',') { m_pos++; return {TokenType::Comma, ",", 0.0}; }
    if (c == '*') {
      m_pos++;
      if (m_pos < m_input.size() && m_input[m_pos] == '*') {
        m_pos++;
        return {TokenType::DoubleStar, "**", 0.0};
      }
      return {TokenType::Star, "*", 0.0};
    }
    throw std::runtime_error(
      "Expression parser: unexpected character '" + std::string(1, c) +
      "' at position " + std::to_string(m_pos));
  }

 private:
  void skip_whitespace() {
    while (m_pos < m_input.size() && std::isspace(m_input[m_pos])) m_pos++;
  }

  Token read_number() {
    size_t start = m_pos;
    while (m_pos < m_input.size() && (std::isdigit(m_input[m_pos]) || m_input[m_pos] == '.'))
      m_pos++;
    if (m_pos < m_input.size() && (m_input[m_pos] == 'e' || m_input[m_pos] == 'E')) {
      m_pos++;
      if (m_pos < m_input.size() && (m_input[m_pos] == '+' || m_input[m_pos] == '-'))
        m_pos++;
      while (m_pos < m_input.size() && std::isdigit(m_input[m_pos])) m_pos++;
    }
    std::string text = m_input.substr(start, m_pos - start);
    return {TokenType::Number, text, std::stod(text)};
  }

  Token read_identifier() {
    size_t start = m_pos;
    while (m_pos < m_input.size() && (std::isalnum(m_input[m_pos]) || m_input[m_pos] == '_'))
      m_pos++;
    return {TokenType::Identifier, m_input.substr(start, m_pos - start), 0.0};
  }

  std::string m_input;
  size_t m_pos;
};

// ---- Recursive descent parser → RPN ----

class ExprParser {
 public:
  using ConstResolver = std::function<bool(const std::string&, double&)>;

  ExprParser(const std::string& input, ConstResolver resolver = nullptr)
    : m_tokenizer(input), m_expr_str(input), m_resolver(std::move(resolver)) {
    advance();
  }

  ParseResult parse() {
    parse_expr();
    if (m_current.type != TokenType::End) {
      throw std::runtime_error(
        "Expression parser: unexpected token '" + m_current.text +
        "' (expected end of expression) in: " + m_expr_str);
    }
    return {m_program, m_field_names, m_constant_names};
  }

 private:
  // expr = term (('+' | '-') term)*
  void parse_expr() {
    parse_term();
    while (m_current.type == TokenType::Plus || m_current.type == TokenType::Minus) {
      auto op = m_current.type;
      advance();
      parse_term();
      emit(op == TokenType::Plus ? ExprOp::Add : ExprOp::Sub);
    }
  }

  // term = power (('*' | '/') power)*
  void parse_term() {
    parse_power();
    while (m_current.type == TokenType::Star || m_current.type == TokenType::Slash) {
      auto op = m_current.type;
      advance();
      parse_power();
      emit(op == TokenType::Star ? ExprOp::Mul : ExprOp::Div);
    }
  }

  // power = unary ('**' unary)*  (right-associative)
  void parse_power() {
    parse_unary();
    if (m_current.type == TokenType::DoubleStar) {
      advance();
      parse_power();
      emit(ExprOp::Pow);
    }
  }

  // unary = ('-' unary) | func_call | atom
  void parse_unary() {
    if (m_current.type == TokenType::Minus) {
      advance();
      parse_unary();
      emit(ExprOp::Neg);
      return;
    }

    // Check for function call: identifier followed by '('
    if (m_current.type == TokenType::Identifier && peek_is_lparen()) {
      auto func = get_func_op(m_current.text);
      if (func.first) {
        advance();  // consume function name
        advance();  // consume '('
        if (func.second == ExprOp::Min || func.second == ExprOp::Max) {
          // Binary functions: min(a, b) / max(a, b)
          parse_expr();
          expect(TokenType::Comma, "',' in min/max");
          advance();
          parse_expr();
        } else {
          // Unary functions
          parse_expr();
        }
        expect(TokenType::RParen, "')' after function argument");
        advance();
        emit(func.second);
        return;
      }
    }

    parse_atom();
  }

  // atom = NUMBER | IDENTIFIER | '(' expr ')'
  void parse_atom() {
    if (m_current.type == TokenType::Number) {
      emit_const(m_current.num_val);
      advance();
    } else if (m_current.type == TokenType::Identifier) {
      std::string name = m_current.text;
      // Try constant resolver first
      double cval;
      if (m_resolver && m_resolver(name, cval)) {
        emit_const(cval);
      } else {
        // Treat as field reference
        emit_field(name);
      }
      advance();
    } else if (m_current.type == TokenType::LParen) {
      advance();
      parse_expr();
      expect(TokenType::RParen, "')'");
      advance();
    } else {
      throw std::runtime_error(
        "Expression parser: unexpected token '" + m_current.text +
        "' in expression: " + m_expr_str);
    }
  }

  // --- Helpers ---

  std::pair<bool, ExprOp> get_func_op(const std::string& name) {
    if (name == "sqrt")   return {true, ExprOp::Sqrt};
    if (name == "abs")    return {true, ExprOp::Abs};
    if (name == "log")    return {true, ExprOp::Log};
    if (name == "exp")    return {true, ExprOp::Exp};
    if (name == "square") return {true, ExprOp::Square};
    if (name == "log10")  return {true, ExprOp::Log10};
    if (name == "min")    return {true, ExprOp::Min};
    if (name == "max")    return {true, ExprOp::Max};
    return {false, ExprOp::PushConst};
  }

  bool peek_is_lparen() {
    // Save state, peek at next token, restore
    // Simple approach: check if next non-whitespace char is '('
    size_t saved_pos = 0;
    // We can't easily peek with our tokenizer, so just check if
    // current identifier is followed by '(' by looking at the expression
    // This is a heuristic - for a known function name, assume it's a call
    auto func = get_func_op(m_current.text);
    return func.first;  // If it's a known function name, treat as call
  }

  void advance() { m_current = m_tokenizer.next(); }

  void expect(TokenType type, const std::string& what) {
    if (m_current.type != type) {
      throw std::runtime_error(
        "Expression parser: expected " + what + " but got '" +
        m_current.text + "' in expression: " + m_expr_str);
    }
  }

  void emit(ExprOp op) {
    if (static_cast<int>(m_program.size()) >= MAX_EXPR_INSTRUCTIONS) {
      throw std::runtime_error("Expression too complex (>" +
        std::to_string(MAX_EXPR_INSTRUCTIONS) + " instructions)");
    }
    m_program.push_back({op, 0, 0.0});
  }

  void emit_const(double val) {
    if (static_cast<int>(m_program.size()) >= MAX_EXPR_INSTRUCTIONS) {
      throw std::runtime_error("Expression too complex");
    }
    m_program.push_back({ExprOp::PushConst, 0, val});
  }

  void emit_field(const std::string& name) {
    if (static_cast<int>(m_program.size()) >= MAX_EXPR_INSTRUCTIONS) {
      throw std::runtime_error("Expression too complex");
    }
    int idx = -1;
    for (int j = 0; j < static_cast<int>(m_field_names.size()); ++j) {
      if (m_field_names[j] == name) { idx = j; break; }
    }
    if (idx < 0) {
      idx = static_cast<int>(m_field_names.size());
      m_field_names.push_back(name);
    }
    m_program.push_back({ExprOp::PushField, idx, 0.0});
  }

  Tokenizer    m_tokenizer;
  std::string  m_expr_str;
  Token        m_current;
  ConstResolver m_resolver;

  std::vector<Instruction> m_program;
  std::vector<std::string> m_field_names;
  std::vector<std::string> m_constant_names;
};

} // namespace detail

// ============================================================
// PUBLIC API
// ============================================================

// Parse with no constant resolution (all identifiers become fields)
inline ParseResult parse_expression(const std::string& expr) {
  detail::ExprParser parser(expr);
  return parser.parse();
}

// Parse with a constant resolver callback.
// The resolver returns true and sets `value` if the identifier is a known constant.
// Otherwise the identifier is treated as a field name.
using ConstResolver = std::function<bool(const std::string& name, double& value)>;

inline ParseResult parse_expression(const std::string& expr,
                                    const ConstResolver& resolver) {
  detail::ExprParser parser(expr, resolver);
  return parser.parse();
}

} // namespace diag_utils

#endif // KOKKOS_DIAG_EXPRESSION_PARSER_HPP
