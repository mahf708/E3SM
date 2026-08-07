#include <edp/parser.hpp>
#include <edp/ast.hpp>
#include <edp/precedences.hpp>
#include <edp/tokens.hpp>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace edp::parser {

namespace {

// A tiny classifier over the AST variant. Two of the parse functions below need
// to know what kind of node they were handed (an assignment? a slice?) in order
// to reject a malformed nesting, and the variant itself is private to
// ast::Expression -- `visit` is the only way in.
enum class NodeKind { Other, Assign, Slice };

struct NodeKindVisitor {
  template <typename T> NodeKind operator()(const T&) const {
    return NodeKind::Other;
  }
  NodeKind operator()(const ast::InfixExpression& expr) const {
    return expr.op == TokenTypes::Assign ? NodeKind::Assign : NodeKind::Other;
  }
  NodeKind operator()(const ast::SliceExpression&) const {
    return NodeKind::Slice;
  }
};

NodeKind node_kind(const ast::ExprPtr& expr) {
  return expr ? expr->visit(NodeKindVisitor{}) : NodeKind::Other;
}

} // namespace

bool Parser::cur_token_is(TokenTypes expected_type) {
  return cur_token_.type == expected_type;
};
bool Parser::peek_token_is(TokenTypes expected_type) {
  return peek_token_.type == expected_type;
};

bool Parser::expect_peek_and_advance(TokenTypes expected_type) {
  if (peek_token_is(expected_type)) {
    next_token();
    return true;
  } else {
    add_error_at(peek_token_, "Expected " +
                                  std::string(to_string(expected_type)) +
                                  ", got " + to_string(peek_token_));
    return false;
  }
}

Precedence Parser::cur_precedence() {
  return token_precedence(cur_token_.type);
}
Precedence Parser::peek_precedence() {
  return token_precedence(peek_token_.type);
}

bool Parser::has_errors() { return !errors_.empty(); }

void Parser::add_error(std::string msg) { errors_.push_back(std::move(msg)); }

std::string Parser::error_at(const Token& tok, const std::string& msg) const {
  std::string out = "line " + std::to_string(tok.line) + ", column " +
                    std::to_string(tok.column) + ": " + msg;

  // Walk to the start of the reported line.
  const std::string& src = lexer_.input();
  std::size_t start = 0;
  for (int line = 1; line < tok.line; ++line) {
    const auto newline = src.find('\n', start);
    if (newline == std::string::npos) {
      return out; // position does not name a line of this input
    }
    start = newline + 1;
  }
  auto stop = src.find('\n', start);
  if (stop == std::string::npos) {
    stop = src.size();
  }

  std::string text = src.substr(start, stop - start);
  if (!text.empty() && text.back() == '\r') {
    text.pop_back();
  }
  if (text.empty()) {
    return out;
  }
  // Tabs would put the caret under the wrong character; one tab, one column.
  std::replace(text.begin(), text.end(), '\t', ' ');

  const int column = tok.column < 1 ? 1 : tok.column;
  auto caret = static_cast<std::size_t>(column - 1);
  if (caret > text.size()) {
    caret = text.size(); // end-of-input tokens point just past the last char
  }

  // The indent lines the snippet up under the "  - " that ParserError adds.
  out += "\n      " + text;
  out += "\n      " + std::string(caret, ' ') + "^";
  return out;
}

void Parser::add_error_at(const Token& tok, const std::string& msg) {
  errors_.push_back(error_at(tok, msg));
}

void Parser::next_token() {
  cur_token_ = peek_token_;
  peek_token_ = lexer_.next_token();
  if (peek_token_is(TokenTypes::Illegal)) {
    // NOTE: upstream printed this to std::cout and carried on, so garbage
    //       input silently produced a valid-looking (but wrong) expression.
    //       This is a library: record the error instead, so parse() throws.
    add_error_at(peek_token_,
                 "Illegal token in input: '" + peek_token_.literal + "'");
  }
}

ast::ExprPtr Parser::parse_expression(Precedence prec) {
  const auto prefix = prefix_parse_fns_.find(cur_token_.type);
  if (prefix == prefix_parse_fns_.end()) {
    // NOTE: upstream throws a bare std::string here, which no std::exception
    //       handler can catch. Throw a ParserError instead, so callers can
    //       report the failure rather than std::terminate.
    // NOTE: report everything recorded so far, not just this one message --
    //       the interesting error is usually the earlier one ("Illegal token
    //       in input: '@'"), and this is merely its consequence.
    add_error_at(cur_token_,
                 "Unexpected Prefix Token " + to_string(cur_token_));
    throw ParserError(errors_);
  }
  const auto fn = prefix->second;
  auto left_expr = (this->*fn)();

  while (!peek_token_is(TokenTypes::EndofFile) && prec < peek_precedence()) {
    const auto infix_it = infix_parse_fns_.find(peek_token_.type);
    if (infix_it == infix_parse_fns_.end()) {
      return left_expr;
    }
    const auto infix_fn = infix_it->second;
    next_token();
    left_expr = (this->*infix_fn)(std::move(left_expr));
  }
  return left_expr;
}

ast::ExprPtr Parser::parse_identifier() {
  return ast::make_expression<ast::Identifier>(cur_token_.literal);
}
ast::ExprPtr Parser::parse_string_literal() {
  return ast::make_expression<ast::StringLiteral>(cur_token_.literal);
}
ast::ExprPtr Parser::parse_integer_literal() {
  int value = 0;
  try {
    std::size_t pos = 0;
    value = std::stoi(cur_token_.literal, &pos);
    if (pos != cur_token_.literal.size()) {
      throw std::invalid_argument("trailing characters");
    }
  } catch (const std::exception&) {
    // Never let a bad literal turn into a silently different number.
    add_error_at(cur_token_,
                 "Invalid integer literal '" + cur_token_.literal + "'");
  }
  return ast::make_expression<ast::IntegerLiteral>(value);
}
ast::ExprPtr Parser::parse_float_literal() {
  double value = 0.0;
  try {
    std::size_t pos = 0;
    value = std::stod(cur_token_.literal, &pos);
    if (pos != cur_token_.literal.size()) {
      throw std::invalid_argument("trailing characters");
    }
  } catch (const std::exception&) {
    add_error_at(cur_token_,
                 "Invalid float literal '" + cur_token_.literal + "'");
  }
  return ast::make_expression<ast::FloatLiteral>(value);
}
ast::ExprPtr Parser::parse_prefix_expression() {
  auto op = cur_token_.type;
  // NOTE: upstream never advanced past the operator, so parse_expression
  //       re-dispatched the same prefix token and recursed until the stack
  //       overflowed ("-T" segfaulted).
  next_token();
  auto right_expr = parse_expression(Precedence::Prefix);
  return ast::make_expression<ast::PrefixExpression>(op, std::move(right_expr));
}

ast::ExprPtr Parser::parse_grouped_expression() {
  next_token();
  auto expr = parse_expression(Precedence::Lowest);
  if (!expect_peek_and_advance(TokenTypes::RightParen)) {
    return nullptr;
  }
  return expr;
}

ast::ExprPtr Parser::parse_infix_expression(ast::ExprPtr left_expr) {
  const auto op = cur_token_.type;
  const auto prec = cur_precedence();
  next_token();

  auto right_expr = parse_expression(prec);

  return ast::make_expression<ast::InfixExpression>(std::move(left_expr), op,
                                                    std::move(right_expr));
}

// `=` exists in this DSL only to spell a keyword argument ("dim='lev'"), so it
// is non-associative: "a = b = c" is meaningless and must not quietly produce a
// nested InfixExpression for a later layer to detect. Assign is left
// associative under the Pratt loop, so a chain shows up as an Assign *left*
// operand; parenthesizing it ("a = (b = c)") puts one on the right instead.
// Both are rejected here, in the parse path, rather than by walking the tree
// afterwards.
ast::ExprPtr Parser::parse_assign_expression(ast::ExprPtr left_expr) {
  const Token op_token = cur_token_;
  const auto prec = cur_precedence();
  const bool left_is_assign = node_kind(left_expr) == NodeKind::Assign;
  next_token();

  auto right_expr = parse_expression(prec);
  const bool right_is_assign = node_kind(right_expr) == NodeKind::Assign;

  if (left_is_assign || right_is_assign) {
    add_error_at(op_token, "Chained assignment is not allowed: '=' is "
                           "non-associative");
  }

  return ast::make_expression<ast::InfixExpression>(
      std::move(left_expr), op_token.type, std::move(right_expr));
}

// True when the token after the current ':' can begin an expression, i.e. when
// the slice component is present rather than omitted. Asking the prefix table
// rather than enumerating terminators means ')' , ']' , ',' and EndofFile all
// answer "omitted" for free, and so does an Illegal token (whose error has
// already been recorded by next_token()).
bool Parser::at_slice_component() const {
  return peek_token_.type != TokenTypes::Colon &&
         prefix_parse_fns_.find(peek_token_.type) != prefix_parse_fns_.end();
}

// Called with cur_token_ == ':' and `start` already parsed (possibly null for
// a leading colon). Consumes "stop" and an optional ":step" so that the whole
// slice becomes ONE SliceExpression -- letting the colon be plain
// left-associative infix would turn "1:2:3" into slice(slice(1,2),3).
ast::ExprPtr Parser::parse_slice_tail(ast::ExprPtr start) {
  ast::ExprPtr stop;
  ast::ExprPtr step;

  if (at_slice_component()) {
    next_token();
    stop = parse_expression(Precedence::Bounds);
  }

  if (peek_token_is(TokenTypes::Colon)) {
    next_token(); // cur_token_ is now the second ':'
    if (at_slice_component()) {
      next_token();
      step = parse_expression(Precedence::Bounds);
    } else {
      // "1::" -- an explicit but empty step. Keep it null; it prints as "1:".
      step = nullptr;
    }
  }

  return ast::make_expression<ast::SliceExpression>(
      std::move(start), std::move(stop), std::move(step));
}

ast::ExprPtr Parser::parse_slice_prefix() { return parse_slice_tail(nullptr); }

ast::ExprPtr Parser::parse_slice_expression(ast::ExprPtr left_expr) {
  if (node_kind(left_expr) == NodeKind::Slice) {
    // e.g. "1:2:3:4": parse_slice_tail already took the two colons it is
    // allowed, so a third one is a syntax error rather than a fourth field.
    add_error_at(cur_token_, "Too many ':' in slice");
  }
  return parse_slice_tail(std::move(left_expr));
}

std::vector<ast::ExprPtr>
Parser::parse_list_of_expressions(TokenTypes end_token) {
  // Should this consume the end-token or not?

  std::vector<ast::ExprPtr> expressions;
  if (peek_token_is(end_token)) {
    next_token();
    return expressions;
  }
  next_token();

  expressions.push_back(parse_expression(Precedence::Lowest));
  // should this be an input arg as well ...?
  while (peek_token_is(TokenTypes::Comma)) {
    next_token();
    next_token(); // Comma should be consumed
    expressions.push_back(parse_expression(Precedence::Lowest));
  }

  if (!expect_peek_and_advance(end_token)) {
    // NOTE: upstream threw a std::runtime_error here, bypassing the ParserError
    //       channel that every other failure uses. expect_peek_and_advance has
    //       already recorded the details.
    add_error_at(peek_token_,
                 "Unexpected token at end of list " + to_string(peek_token_));
    throw ParserError(errors_);
  }
  return expressions;
}

ast::ExprPtr Parser::parse_function_expression(ast::ExprPtr func) {
  auto args = parse_list_of_expressions(TokenTypes::RightParen);
  return ast::make_expression<ast::FuncExpression>(std::move(func),
                                                   std::move(args));
}

ast::ExprPtr Parser::parse_array_expression() {
  return ast::make_expression<ast::ArrayExpression>(
      parse_list_of_expressions(TokenTypes::ArrayRightBracket));
}

Parser::Parser(Lexer lexer)
    : lexer_{std::move(lexer)},
      prefix_parse_fns_{{
          {TokenTypes::Identifier, &Parser::parse_identifier},
          {TokenTypes::Integer, &Parser::parse_integer_literal},
          {TokenTypes::Float, &Parser::parse_float_literal},
          {TokenTypes::String, &Parser::parse_string_literal},
          {TokenTypes::Minus, &Parser::parse_prefix_expression},
          {TokenTypes::Bang, &Parser::parse_prefix_expression},
          // NOTE: upstream defined parse_grouped_expression but never
          //       registered it, so no parenthesized expression could parse.
          //       LeftParen is legitimately in both maps: prefix position is
          //       grouping, infix position is a function call.
          {TokenTypes::LeftParen, &Parser::parse_grouped_expression},
          {TokenTypes::ArrayLeftBracket, &Parser::parse_array_expression},
          // A leading colon is a slice with an omitted start (":10", "::2",
          // ":"). Colon is deliberately in both maps, like LeftParen.
          {TokenTypes::Colon, &Parser::parse_slice_prefix},
      }},
      infix_parse_fns_{{
          {TokenTypes::Plus, &Parser::parse_infix_expression},
          {TokenTypes::Minus, &Parser::parse_infix_expression},
          {TokenTypes::Asterisk, &Parser::parse_infix_expression},
          {TokenTypes::Exp, &Parser::parse_infix_expression},
          {TokenTypes::Assign, &Parser::parse_assign_expression},
          {TokenTypes::Slash, &Parser::parse_infix_expression},
          {TokenTypes::Equal, &Parser::parse_infix_expression},
          {TokenTypes::NotEqual, &Parser::parse_infix_expression},
          {TokenTypes::GreaterThan, &Parser::parse_infix_expression},
          {TokenTypes::GreaterEqual, &Parser::parse_infix_expression},
          {TokenTypes::LessThan, &Parser::parse_infix_expression},
          {TokenTypes::LessEq, &Parser::parse_infix_expression},
          {TokenTypes::Or, &Parser::parse_infix_expression},
          {TokenTypes::And, &Parser::parse_infix_expression},
          {TokenTypes::Dot, &Parser::parse_infix_expression},
          {TokenTypes::LeftParen, &Parser::parse_function_expression},
          // NOTE: Colon had a precedence (Bounds) but no parse function
          //       upstream, so "0:10" could not parse at all.
          {TokenTypes::Colon, &Parser::parse_slice_expression},
      }} {
  next_token();
  next_token();
}

ast::ExprPtr Parser::parse() {
  // For now i'll assume we're parsing one expression statement at a time
  // and nothing more complicated
  auto expr = parse_expression(Precedence::Lowest);

  // NOTE: upstream returned whatever it had parsed without checking that the
  //       input was exhausted, so trailing garbage was silently dropped
  //       ("T @ x" parsed as "T"). Require that we stopped at end-of-input.
  if (!cur_token_is(TokenTypes::EndofFile) &&
      !peek_token_is(TokenTypes::EndofFile)) {
    add_error_at(peek_token_, "Unexpected token after end of expression: " +
                                  to_string(peek_token_));
  }

  if (has_errors()) {
    throw ParserError(errors_);
  }
  return expr;
}

} // namespace edp::parser
