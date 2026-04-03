#include "expression_diag.hpp"

#include "share/physics/physics_constants.hpp"

#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace scream {

// ============================================================
// TOKENIZER
// ============================================================

enum class TokenType {
  Number,       // 3.14, 1e-5
  Identifier,   // field names, function names, constants
  Plus, Minus, Star, Slash, DoubleStar,  // operators
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

    // Number: digit or decimal point followed by digit
    if (std::isdigit(c) || (c == '.' && m_pos + 1 < m_input.size() && std::isdigit(m_input[m_pos+1]))) {
      return read_number();
    }

    // Identifier: letter or underscore
    if (std::isalpha(c) || c == '_') {
      return read_identifier();
    }

    // Operators
    if (c == '+') { m_pos++; return {TokenType::Plus, "+", 0.0}; }
    if (c == '-') { m_pos++; return {TokenType::Minus, "-", 0.0}; }
    if (c == '/') { m_pos++; return {TokenType::Slash, "/", 0.0}; }
    if (c == '(') { m_pos++; return {TokenType::LParen, "(", 0.0}; }
    if (c == ')') { m_pos++; return {TokenType::RParen, ")", 0.0}; }

    // * or **
    if (c == '*') {
      m_pos++;
      if (m_pos < m_input.size() && m_input[m_pos] == '*') {
        m_pos++;
        return {TokenType::DoubleStar, "**", 0.0};
      }
      return {TokenType::Star, "*", 0.0};
    }

    EKAT_ERROR_MSG("ExpressionDiag parser: unexpected character '" +
                   std::string(1, c) + "' at position " + std::to_string(m_pos) +
                   " in expression: " + m_input + "\n");
    return {TokenType::End, "", 0.0};
  }

 private:
  void skip_whitespace() {
    while (m_pos < m_input.size() && std::isspace(m_input[m_pos])) m_pos++;
  }

  Token read_number() {
    size_t start = m_pos;
    while (m_pos < m_input.size() && (std::isdigit(m_input[m_pos]) || m_input[m_pos] == '.'))
      m_pos++;
    // Scientific notation: e or E
    if (m_pos < m_input.size() && (m_input[m_pos] == 'e' || m_input[m_pos] == 'E')) {
      m_pos++;
      if (m_pos < m_input.size() && (m_input[m_pos] == '+' || m_input[m_pos] == '-'))
        m_pos++;
      while (m_pos < m_input.size() && std::isdigit(m_input[m_pos]))
        m_pos++;
    }
    std::string text = m_input.substr(start, m_pos - start);
    double val = std::stod(text);
    return {TokenType::Number, text, val};
  }

  Token read_identifier() {
    size_t start = m_pos;
    while (m_pos < m_input.size() &&
           (std::isalnum(m_input[m_pos]) || m_input[m_pos] == '_'))
      m_pos++;
    std::string text = m_input.substr(start, m_pos - start);
    return {TokenType::Identifier, text, 0.0};
  }

  std::string m_input;
  size_t m_pos;
};

// ============================================================
// RECURSIVE DESCENT PARSER → RPN
// ============================================================

class ExprParser {
 public:
  explicit ExprParser(const std::string& input)
    : m_tokenizer(input), m_expr_str(input) {
    advance();
  }

  ParseResult parse() {
    parse_expr();
    expect(TokenType::End, "end of expression");
    return {m_program, m_field_names};
  }

 private:
  // --- Grammar rules ---

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

  // power = unary ('**' unary)*   (right-associative)
  void parse_power() {
    parse_unary();
    if (m_current.type == TokenType::DoubleStar) {
      advance();
      parse_power();  // right-associative recursion
      emit(ExprOp::Pow);
    }
  }

  // unary = ('-' unary) | func '(' expr ')' | atom
  void parse_unary() {
    if (m_current.type == TokenType::Minus) {
      advance();
      parse_unary();
      emit(ExprOp::Neg);
      return;
    }

    // Check for function call: identifier followed by '('
    if (m_current.type == TokenType::Identifier) {
      ExprOp func_op = get_func_op(m_current.text);
      if (func_op != ExprOp::PushConst) {  // PushConst used as "not a function" sentinel
        advance();  // consume function name
        expect(TokenType::LParen, "'(' after function name");
        advance();  // consume '('
        parse_expr();
        expect(TokenType::RParen, "')' after function argument");
        advance();  // consume ')'
        emit(func_op);
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
      // Check if it's a physics constant
      const auto& dict = physics::Constants<Real>::dictionary();
      ekat::CaseInsensitiveString ci_name(m_current.text);
      if (dict.count(ci_name) > 0) {
        emit_const(dict.at(ci_name).value);
      } else {
        // It's a field name
        emit_field(m_current.text);
      }
      advance();
    } else if (m_current.type == TokenType::LParen) {
      advance();  // consume '('
      parse_expr();
      expect(TokenType::RParen, "')'");
      advance();  // consume ')'
    } else {
      EKAT_ERROR_MSG("ExpressionDiag parser: unexpected token '" +
                     m_current.text + "' in expression: " + m_expr_str + "\n");
    }
  }

  // --- Helpers ---

  ExprOp get_func_op(const std::string& name) {
    if (name == "sqrt")   return ExprOp::Sqrt;
    if (name == "abs")    return ExprOp::Abs;
    if (name == "log")    return ExprOp::Log;
    if (name == "exp")    return ExprOp::Exp;
    if (name == "square") return ExprOp::Square;
    return ExprOp::PushConst;  // sentinel: not a function
  }

  void advance() {
    m_current = m_tokenizer.next();
  }

  void expect(TokenType type, const std::string& what) {
    EKAT_REQUIRE_MSG(m_current.type == type,
      "ExpressionDiag parser: expected " + what +
      " but got '" + m_current.text + "' in expression: " + m_expr_str + "\n");
  }

  void emit(ExprOp op) {
    EKAT_REQUIRE_MSG(static_cast<int>(m_program.size()) < MAX_EXPR_INSTRUCTIONS,
      "ExpressionDiag: expression too complex (>" +
      std::to_string(MAX_EXPR_INSTRUCTIONS) + " instructions).\n");
    m_program.push_back({op, 0, 0.0});
  }

  void emit_const(Real val) {
    EKAT_REQUIRE_MSG(static_cast<int>(m_program.size()) < MAX_EXPR_INSTRUCTIONS,
      "ExpressionDiag: expression too complex.\n");
    m_program.push_back({ExprOp::PushConst, 0, val});
  }

  void emit_field(const std::string& name) {
    EKAT_REQUIRE_MSG(static_cast<int>(m_program.size()) < MAX_EXPR_INSTRUCTIONS,
      "ExpressionDiag: expression too complex.\n");
    // Find or add the field name
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

  Tokenizer   m_tokenizer;
  std::string m_expr_str;
  Token       m_current;

  std::vector<Instruction> m_program;
  std::vector<std::string> m_field_names;
};

// ============================================================
// PUBLIC PARSE FUNCTION
// ============================================================

ParseResult parse_expression(const std::string& expr) {
  ExprParser parser(expr);
  return parser.parse();
}

// ============================================================
// RPN EVALUATOR (device-safe inline function)
// ============================================================

KOKKOS_INLINE_FUNCTION
Real evaluate_rpn(const Instruction* program, const int n_instr,
                  const Real* const* field_ptrs, const int elem_idx) {
  Real stack[MAX_EXPR_STACK];
  int sp = 0;  // stack pointer (points to next free slot)

  for (int pc = 0; pc < n_instr; ++pc) {
    const auto& instr = program[pc];
    switch (instr.op) {
      case ExprOp::PushField:
        stack[sp++] = field_ptrs[instr.operand_idx][elem_idx];
        break;
      case ExprOp::PushConst:
        stack[sp++] = instr.const_val;
        break;

      // Binary ops: pop b, pop a, push result
      case ExprOp::Add: { auto b = stack[--sp]; auto a = stack[--sp]; stack[sp++] = a + b; break; }
      case ExprOp::Sub: { auto b = stack[--sp]; auto a = stack[--sp]; stack[sp++] = a - b; break; }
      case ExprOp::Mul: { auto b = stack[--sp]; auto a = stack[--sp]; stack[sp++] = a * b; break; }
      case ExprOp::Div: { auto b = stack[--sp]; auto a = stack[--sp]; stack[sp++] = a / b; break; }
      case ExprOp::Pow: { auto b = stack[--sp]; auto a = stack[--sp]; stack[sp++] = pow(a, b); break; }

      // Unary ops: pop a, push result
      case ExprOp::Sqrt:   { auto a = stack[--sp]; stack[sp++] = sqrt(a);  break; }
      case ExprOp::Abs:    { auto a = stack[--sp]; stack[sp++] = fabs(a);  break; }
      case ExprOp::Log:    { auto a = stack[--sp]; stack[sp++] = log(a);   break; }
      case ExprOp::Exp:    { auto a = stack[--sp]; stack[sp++] = exp(a);   break; }
      case ExprOp::Square: { auto a = stack[--sp]; stack[sp++] = a * a;    break; }
      case ExprOp::Neg:    { auto a = stack[--sp]; stack[sp++] = -a;       break; }
    }
  }
  return stack[0];
}

// ============================================================
// ExpressionDiag IMPLEMENTATION
// ============================================================

ExpressionDiag::
ExpressionDiag(const ekat::Comm &comm, const ekat::ParameterList &params)
 : AtmosphereDiagnostic(comm, params)
{
  m_expr_str = m_params.get<std::string>("expression");

  EKAT_REQUIRE_MSG(!m_expr_str.empty(),
    "Error! ExpressionDiag requires a non-empty 'expression' parameter.\n");

  // Parse the expression now (host-side) so errors are caught early
  auto result = parse_expression(m_expr_str);
  m_program     = std::move(result.program);
  m_field_names = std::move(result.field_names);

  EKAT_REQUIRE_MSG(!m_program.empty(),
    "Error! ExpressionDiag: expression '" + m_expr_str + "' produced no instructions.\n");
}

void ExpressionDiag::
create_requests()
{
  const auto &gname = m_params.get<std::string>("grid_name");

  // Request all fields referenced in the expression
  for (const auto& fname : m_field_names) {
    add_field<Required>(fname, gname);
  }
}

void ExpressionDiag::initialize_impl(const RunType /*run_type*/)
{
  // All fields must have the same layout
  EKAT_REQUIRE_MSG(!m_field_names.empty(),
    "Error! ExpressionDiag expression '" + m_expr_str +
    "' references no fields. Use physics constants with BinaryOpsDiag instead.\n");

  const auto& first_fid = get_field_in(m_field_names[0]).get_header().get_identifier();
  const auto& dl = first_fid.get_layout();

  for (size_t i = 1; i < m_field_names.size(); ++i) {
    const auto& fid = get_field_in(m_field_names[i]).get_header().get_identifier();
    EKAT_REQUIRE_MSG(fid.get_layout() == dl,
      "Error! ExpressionDiag requires all fields to have the same layout.\n"
      " - expression: " + m_expr_str + "\n"
      " - field '" + m_field_names[0] + "' layout: " + dl.to_string() + "\n"
      " - field '" + m_field_names[i] + "' layout: " + fid.get_layout().to_string() + "\n");
  }

  // Output units are nondimensional for general expressions
  // (unit tracking for arbitrary expressions would require a full unit inference engine)
  auto gn = m_params.get<std::string>("grid_name");
  auto diag_name = "expr_" + m_expr_str;
  FieldIdentifier d_fid(diag_name, dl, ekat::units::Units::nondimensional(), gn);
  m_diagnostic_output = Field(d_fid, true);
}

void ExpressionDiag::compute_diagnostic_impl()
{
  const int n_fields = static_cast<int>(m_field_names.size());
  const int n_instr  = static_cast<int>(m_program.size());
  const int total_size = m_diagnostic_output.get_header().get_identifier().get_layout().size();

  EKAT_REQUIRE_MSG(n_fields <= MAX_EXPR_STACK,
    "Error! ExpressionDiag: expression references too many fields (" +
    std::to_string(n_fields) + " > " + std::to_string(MAX_EXPR_STACK) + ").\n");

  // Gather device pointers to field data
  const Real* field_ptrs[MAX_EXPR_STACK] = {};
  for (int j = 0; j < n_fields; ++j) {
    field_ptrs[j] = get_field_in(m_field_names[j]).get_internal_view_data<const Real>();
  }

  // Copy RPN program to device-accessible memory
  Kokkos::View<Instruction*, Kokkos::DefaultExecutionSpace::memory_space>
    d_program("expr_program", n_instr);
  {
    auto h_program = Kokkos::create_mirror_view(d_program);
    for (int i = 0; i < n_instr; ++i) {
      h_program(i) = m_program[i];
    }
    Kokkos::deep_copy(d_program, h_program);
  }

  // Copy field pointers to device via uintptr_t View
  // (Kokkos Views cannot store raw pointers directly)
  Kokkos::View<uintptr_t*, Kokkos::DefaultExecutionSpace::memory_space>
    d_fptrs("fld_ptrs", n_fields);
  {
    auto h_fptrs = Kokkos::create_mirror_view(d_fptrs);
    for (int j = 0; j < n_fields; ++j) {
      h_fptrs(j) = reinterpret_cast<uintptr_t>(field_ptrs[j]);
    }
    Kokkos::deep_copy(d_fptrs, h_fptrs);
  }

  auto d_out = m_diagnostic_output.get_internal_view_data<Real>();
  auto prog  = d_program;
  auto fptrs = d_fptrs;
  const int ni = n_instr;
  const int nf = n_fields;

  Kokkos::parallel_for("ExpressionDiag",
    Kokkos::RangePolicy<>(0, total_size),
    KOKKOS_LAMBDA(const int& i) {
      // Reconstruct field pointer array from uintptr_t
      const Real* local_fptrs[MAX_EXPR_STACK];
      for (int j = 0; j < nf; ++j) {
        local_fptrs[j] = reinterpret_cast<const Real*>(fptrs(j));
      }
      d_out[i] = evaluate_rpn(prog.data(), ni, local_fptrs, i);
    });
}

}  // namespace scream
