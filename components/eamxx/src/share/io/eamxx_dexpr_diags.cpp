#include "share/io/eamxx_dexpr_diags.hpp"

#include <dexpr/ast.hpp>
#include <dexpr/lexer.hpp>
#include <dexpr/parser.hpp>
#include <dexpr/supported_functions.hpp>
#include <dexpr/tokens.hpp>

#include <ekat_assert.hpp>

#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace scream {

namespace {

using dexpr::TokenTypes;
namespace ast = dexpr::ast;

// ---------------------------------------------------------------------------
// The EAMxx vocabulary
// ---------------------------------------------------------------------------
// Spelled after xarray wherever there is an honest analogue, since that is what
// users already reach for when post-processing the output these diags replace.
// This lives here, not in share/dexpr: dexpr owns the grammar, we own the
// vocabulary, and adding a diag here touches nothing in that library.
const dexpr::FunctionRegistry& eamxx_registry ()
{
  static const dexpr::FunctionRegistry reg = [] {
    dexpr::FunctionRegistry r;
    using dexpr::CallForm;

    r.add({.name = "isel",
           .desc = "value at a vertical index: X.isel(lev=10), X.isel(lev=-1)",
           .min_positional = 0, .max_positional = 0,
           .keywords = {{"lev",true}},
           .form = CallForm::Method});
    r.add({.name = "interp",
           .desc = "value interpolated to a pressure or height: "
                   "X.interp(plev=500,units='hPa'), X.interp(z=10,reference='surface')",
           .min_positional = 0, .max_positional = 0,
           .keywords = {{"plev",false},{"z",false},{"units",false},{"reference",false}},
           .form = CallForm::Method});
    r.add({.name = "mean",
           .desc = "average over a dimension: X.mean('col'), X.mean('lev',weights='dp')",
           .min_positional = 1, .max_positional = 1,
           .keywords = {{"weights",false}},
           .form = CallForm::Method});
    r.add({.name = "sum",
           .desc = "sum over a dimension: X.sum('lev'), X.sum('lev',weights='dz')",
           .min_positional = 1, .max_positional = 1,
           .keywords = {{"weights",false}},
           .form = CallForm::Method});
    r.add({.name = "where",
           .desc = "keep values where a condition holds: X.where(qv>0.01)",
           .min_positional = 1, .max_positional = 1,
           .keywords = {},
           .form = CallForm::Method});
    r.add({.name = "differentiate",
           .desc = "vertical derivative w.r.t. pressure or height: X.differentiate('p')",
           .min_positional = 1, .max_positional = 1,
           .keywords = {},
           .form = CallForm::Method});
    r.add({.name = "histogram",
           .desc = "counts per bin, given the bin edges: X.histogram([0,1,2])",
           .min_positional = 1, .max_positional = 1,
           .keywords = {},
           .form = CallForm::Method});
    r.add({.name = "zonal_mean",
           .desc = "average within latitude bands: X.zonal_mean(bins=20)",
           .min_positional = 0, .max_positional = 0,
           .keywords = {{"bins",true}},
           .form = CallForm::Method});
    r.add({.name = "prev",
           .desc = "value at the previous time step: X.prev()",
           .min_positional = 0, .max_positional = 0,
           .keywords = {},
           .form = CallForm::Method});
    r.add({.name = "over_dt",
           .desc = "value divided by the time step: X.over_dt()",
           .min_positional = 0, .max_positional = 0,
           .keywords = {},
           .form = CallForm::Method});
    r.add({.name = "tend",
           .desc = "tendency over the time step, i.e. (X-X.prev()).over_dt()",
           .min_positional = 0, .max_positional = 0,
           .keywords = {},
           .form = CallForm::Method});
    return r;
  }();
  return reg;
}

// ---------------------------------------------------------------------------
// Small helpers over the AST
// ---------------------------------------------------------------------------

template<typename Node>
const Node* as (const ast::Expression& e) {
  return e.visit([](const auto& node) -> const Node* {
    if constexpr (std::is_same_v<std::decay_t<decltype(node)>,Node>) {
      return &node;
    } else {
      return nullptr;
    }
  });
}

[[noreturn]] void unsupported (const std::string& what, const ast::Expression& e)
{
  EKAT_ERROR_MSG (
      "Error! Unsupported expression: " + what + ".\n"
      " - subexpression: " + ast::to_string(e) + "\n"
      " - for the diagnostics EAMxx can build from an expression, see\n"
      "   components/eamxx/docs/user/diags/expressions.md\n");
  throw std::logic_error("unreachable"); // EKAT_ERROR_MSG already threw
}

// A call, split into the pieces the translation needs. Method calls parse as a
// FuncExpression whose 'function' is a Dot binary expression, so the operand
// and the method name have to be dug back out.
struct Call {
  const ast::Expression* receiver = nullptr;
  std::string name;
  std::vector<const ast::Expression*> positional;
  std::map<std::string,const ast::Expression*> keywords;
};

// Returns false if 'e' is not a method call at all.
bool as_method_call (const ast::Expression& e, Call& call)
{
  const auto* fn = as<ast::FuncExpression>(e);
  if (fn==nullptr) {
    return false;
  }
  const auto* dot = as<ast::BinaryExpression>(*fn->function);
  if (dot==nullptr or dot->op!=TokenTypes::Dot) {
    return false;
  }
  const auto* name = as<ast::Identifier>(*dot->right);
  if (name==nullptr) {
    return false;
  }

  call.receiver = dot->left.get();
  call.name     = name->value;
  for (const auto& arg : fn->args) {
    // A keyword argument parses as an Assign whose lhs is a name. Anything else
    // counts as positional -- including 'f(1=2)', which validate_calls() sees
    // the same way, and rejects on arity.
    const auto* assign = as<ast::BinaryExpression>(*arg);
    const auto* kw = assign!=nullptr && assign->op==TokenTypes::Assign
                   ? as<ast::Identifier>(*assign->left)
                   : nullptr;
    if (kw!=nullptr) {
      call.keywords[kw->value] = assign->right.get();
    } else {
      call.positional.push_back(arg.get());
    }
  }
  return true;
}

// The name a subexpression is known by. Anything that is not a plain field name
// comes back fully parenthesized, e.g. "(qc+qv)", which is both what
// create_diagnostic() will be asked to resolve later and what the resulting
// field is named. Rendering is dexpr's, so it round-trips through the parser.
std::string name_of (const ast::Expression& e) {
  return ast::to_string(e);
}

// Literals are rendered by dexpr too: a FloatLiteral has kept a double and lost
// its lexeme, and dexpr already prints the shortest form that reads back.
std::string literal_str (const ast::Expression& e)
{
  if (as<ast::IntegerLiteral>(e) or as<ast::FloatLiteral>(e)) {
    return ast::to_string(e);
  }
  if (const auto* s = as<ast::StringLiteral>(e)) {
    return s->value;
  }
  unsupported("expected a literal",e);
}

int int_arg (const ast::Expression& e, const std::string& ctx)
{
  const auto* i = as<ast::IntegerLiteral>(e);
  if (i==nullptr) {
    // Unary minus is how a negative index parses, and it is the only place we
    // accept one, since there is no diagnostic that negates a field.
    const auto* u = as<ast::UnaryExpression>(e);
    if (u!=nullptr and u->op==TokenTypes::Minus) {
      const auto* inner = as<ast::IntegerLiteral>(*u->right);
      if (inner!=nullptr) {
        return -inner->value;
      }
    }
    unsupported(ctx + " must be an integer",e);
  }
  return i->value;
}

std::string string_arg (const ast::Expression& e, const std::string& ctx)
{
  const auto* s = as<ast::StringLiteral>(e);
  if (s==nullptr) {
    unsupported(ctx + " must be a quoted string",e);
  }
  return s->value;
}

const ast::Expression* keyword (const Call& call, const std::string& name)
{
  auto it = call.keywords.find(name);
  return it==call.keywords.end() ? nullptr : it->second;
}

// ---------------------------------------------------------------------------
// Translation: one AST node -> one diagnostic
// ---------------------------------------------------------------------------
// Only the root node is translated. Operands are handed down by name, and the
// customer resolving diag dependencies brings them back here one at a time.

std::string comparison_to_cmp (TokenTypes op)
{
  switch (op) {
    case TokenTypes::GreaterThan:  return "gt";
    case TokenTypes::GreaterEqual: return "ge";
    case TokenTypes::LessThan:     return "lt";
    case TokenTypes::LessEq:       return "le";
    case TokenTypes::Equal:        return "eq";
    case TokenTypes::NotEqual:     return "ne";
    default:                       return "";
  }
}

std::string binary_op_to_diag_op (TokenTypes op)
{
  switch (op) {
    case TokenTypes::Plus:     return "plus";
    case TokenTypes::Minus:    return "minus";
    case TokenTypes::Asterisk: return "times";
    case TokenTypes::Slash:    return "over";
    default:                   return "";
  }
}

void translate_binary (const ast::BinaryExpression& e, const ast::Expression& self,
                       std::string& diag_name, ekat::ParameterList& params)
{
  const auto op = binary_op_to_diag_op(e.op);
  if (op=="") {
    if (comparison_to_cmp(e.op)!="") {
      unsupported("a comparison is only meaningful inside where(..)",self);
    }
    if (e.op==TokenTypes::Dot) {
      unsupported("attribute access with no call",self);
    }
    unsupported("no diagnostic implements the operator '" +
                dexpr::binary_op_to_string(e.op) + "'",self);
  }

  diag_name = "BinaryOp";
  params.set<std::string>("arg1",name_of(*e.left));
  params.set<std::string>("arg2",name_of(*e.right));
  params.set<std::string>("binary_op",op);
}

void translate_call (const Call& call, const ast::Expression& self,
                     std::string& diag_name, ekat::ParameterList& params)
{
  const auto operand = name_of(*call.receiver);
  params.set<std::string>("field_name",operand);

  if (call.name=="isel") {
    // xarray indexing, so a negative index counts from the end. Only -1 can be
    // resolved without knowing the number of levels, and it is the only one
    // anybody wants ("the bottom level").
    const auto* lev = keyword(call,"lev");
    std::string location;
    if (const auto* s = as<ast::StringLiteral>(*lev)) {
      location = s->value;
      EKAT_REQUIRE_MSG (location=="model_top" or location=="model_bot",
          "Error! Invalid vertical index in isel().\n"
          " - expression: " + ast::to_string(self) + "\n"
          " - input: '" + location + "'\n"
          " - expected an integer, or 'model_top'/'model_bot'\n");
    } else {
      const auto idx = int_arg(*lev,"the 'lev' index");
      if (idx==-1) {
        location = "model_bot";
      } else {
        EKAT_REQUIRE_MSG (idx>=0,
            "Error! Invalid vertical index in isel().\n"
            " - expression: " + ast::to_string(self) + "\n"
            " - input: " + std::to_string(idx) + "\n"
            " - only -1 (the bottom level) is supported as a negative index,\n"
            "   since the others cannot be resolved without the level count.\n");
        location = "lev_" + std::to_string(idx);
      }
    }
    diag_name = "FieldAtLevel";
    params.set<std::string>("vertical_location",location);
    return;
  }

  if (call.name=="interp") {
    const auto* plev = keyword(call,"plev");
    const auto* z    = keyword(call,"z");
    EKAT_REQUIRE_MSG ((plev!=nullptr) != (z!=nullptr),
        "Error! interp() takes exactly one of 'plev' or 'z'.\n"
        " - expression: " + ast::to_string(self) + "\n");

    if (plev!=nullptr) {
      const auto* units = keyword(call,"units");
      EKAT_REQUIRE_MSG (keyword(call,"reference")==nullptr,
          "Error! 'reference' applies to interp(z=..), not interp(plev=..).\n"
          " - expression: " + ast::to_string(self) + "\n");
      diag_name = "FieldAtPressureLevel";
      params.set<std::string>("pressure_value",literal_str(*plev));
      params.set<std::string>("pressure_units",
          units==nullptr ? std::string("Pa") : string_arg(*units,"'units'"));
    } else {
      const auto* units = keyword(call,"units");
      const auto* ref   = keyword(call,"reference");
      diag_name = "FieldAtHeight";
      params.set<std::string>("height_value",literal_str(*z));
      params.set<std::string>("height_units",
          units==nullptr ? std::string("m") : string_arg(*units,"'units'"));
      params.set<std::string>("surface_reference",
          ref==nullptr ? std::string("sealevel") : string_arg(*ref,"'reference'"));
    }
    return;
  }

  if (call.name=="mean" or call.name=="sum") {
    const auto dim = string_arg(*call.positional[0],"the dimension");
    const auto* weights = keyword(call,"weights");

    if (dim=="col") {
      EKAT_REQUIRE_MSG (call.name=="mean",
          "Error! Only an average is available over 'col'.\n"
          " - expression: " + ast::to_string(self) + "\n");
      EKAT_REQUIRE_MSG (weights==nullptr,
          "Error! Averaging over 'col' is always area weighted; 'weights' does not apply.\n"
          " - expression: " + ast::to_string(self) + "\n");
      diag_name = "HorizAvg";
      return;
    }

    EKAT_REQUIRE_MSG (dim=="lev",
        "Error! Unknown dimension in " + call.name + "().\n"
        " - expression: " + ast::to_string(self) + "\n"
        " - input: '" + dim + "'\n"
        " - valid dimensions: 'col', 'lev'\n");

    diag_name = "VertContract";
    params.set<std::string>("contract_method",call.name=="mean" ? "avg" : "sum");
    if (weights!=nullptr) {
      params.set<std::string>("weighting_method",string_arg(*weights,"'weights'"));
    }
    return;
  }

  if (call.name=="where") {
    const auto& cond = *call.positional[0];
    const auto* cmp = as<ast::BinaryExpression>(cond);
    EKAT_REQUIRE_MSG (cmp!=nullptr and comparison_to_cmp(cmp->op)!="",
        "Error! where() takes a single comparison.\n"
        " - expression: " + ast::to_string(self) + "\n"
        " - condition: " + ast::to_string(cond) + "\n"
        " - valid comparisons: >, >=, <, <=, ==, !=\n"
        " - note: 'and'/'or' are not supported; chain where(..) calls instead.\n");

    diag_name = "ConditionalSampling";
    params.set<std::string>("condition_lhs",name_of(*cmp->left));
    params.set<std::string>("condition_cmp",comparison_to_cmp(cmp->op));
    params.set<std::string>("condition_rhs",name_of(*cmp->right));
    return;
  }

  if (call.name=="differentiate") {
    const auto wrt = string_arg(*call.positional[0],"the coordinate");
    EKAT_REQUIRE_MSG (wrt=="p" or wrt=="z",
        "Error! Unknown coordinate in differentiate().\n"
        " - expression: " + ast::to_string(self) + "\n"
        " - input: '" + wrt + "'\n"
        " - valid coordinates: 'p', 'z'\n");
    diag_name = "VertDerivative";
    params.set<std::string>("derivative_method",wrt);
    return;
  }

  if (call.name=="histogram") {
    const auto* bins = as<ast::ArrayExpression>(*call.positional[0]);
    EKAT_REQUIRE_MSG (bins!=nullptr and bins->elements.size()>=2,
        "Error! histogram() takes an array of at least two bin edges.\n"
        " - expression: " + ast::to_string(self) + "\n");
    // The diag re-splits this on '_', so the edges must survive that round trip:
    // no exponents, and no minus signs.
    std::string config;
    for (size_t i=0; i<bins->elements.size(); ++i) {
      const auto edge = literal_str(*bins->elements[i]);
      EKAT_REQUIRE_MSG (edge.find_first_not_of("0123456789.")==std::string::npos,
          "Error! Histogram bin edges must be non-negative decimals.\n"
          " - expression: " + ast::to_string(self) + "\n"
          " - edge: " + edge + "\n");
      config += (i==0 ? "" : "_") + edge;
    }
    diag_name = "Histogram";
    params.set<std::string>("bin_configuration",config);
    return;
  }

  if (call.name=="zonal_mean") {
    const auto bins = int_arg(*keyword(call,"bins"),"'bins'");
    EKAT_REQUIRE_MSG (bins>0,
        "Error! zonal_mean() needs a positive number of bins.\n"
        " - expression: " + ast::to_string(self) + "\n"
        " - bins: " + std::to_string(bins) + "\n");
    diag_name = "ZonalAvg";
    params.set<std::string>("number_of_zonal_bins",std::to_string(bins));
    return;
  }

  if (call.name=="prev") {
    diag_name = "FieldPrev";
    return;
  }

  if (call.name=="over_dt") {
    diag_name = "FieldOverDt";
    return;
  }

  if (call.name=="tend") {
    // Shorthand, expanded here rather than in the grammar: the operand of the
    // FieldOverDt is the difference expression, which comes straight back
    // through create_diagnostic() and is built as a BinaryOp.
    diag_name = "FieldOverDt";
    params.set<std::string>("field_name","(" + operand + "-" + operand + ".prev())");
    return;
  }

  // validate_calls() has already rejected unknown names, so reaching here means
  // the registry and this switch have drifted apart.
  EKAT_ERROR_MSG (
      "Error! Internal error: '" + call.name + "' is registered but not translated.\n"
      " - expression: " + ast::to_string(self) + "\n");
}

} // anonymous namespace

std::shared_ptr<AbstractDiagnostic>
dexpr_create_diagnostic (const std::string& expr,
                         const std::shared_ptr<const AbstractGrid>& grid)
{
  ast::ExprPtr root;
  try {
    dexpr::parser::Parser parser {dexpr::Lexer{expr}};
    root = parser.parse();
  } catch (const dexpr::parser::ParserError& e) {
    // Getting here means the name did not match any of the legacy patterns AND
    // is not a legal identifier, so it can only have been meant as an
    // expression. Report where it went wrong rather than the much less useful
    // "no such diagnostic".
    EKAT_ERROR_MSG (
        "Error! '" + expr + "' is neither a registered diagnostic nor a valid expression.\n" +
        std::string(e.what()));
  }

  // A bare identifier is not an expression: it is a diagnostic class name, or a
  // typo. Either way it is the caller's to resolve, exactly as before.
  if (as<ast::Identifier>(*root)) {
    return nullptr;
  }

  dexpr::validate_calls(*root,eamxx_registry());

  std::string diag_name;
  ekat::ParameterList params(expr);
  params.set("grid_name",grid->name());

  if (Call call; as_method_call(*root,call)) {
    translate_call(call,*root,diag_name,params);
  } else if (const auto* bin = as<ast::BinaryExpression>(*root)) {
    translate_binary(*bin,*root,diag_name,params);
  } else if (const auto* un = as<ast::UnaryExpression>(*root)) {
    unsupported("no diagnostic implements the unary operator '" +
                dexpr::unary_op_to_string(un->op) + "'",*root);
  } else if (as<ast::FuncExpression>(*root)) {
    unsupported("EAMxx functions are written as methods, X.f(..), not f(X)",*root);
  } else {
    unsupported("an expression must produce a field",*root);
  }

  // The field is named after the request, not after the diag's own param
  // concatenation, so that customers find it under the name they asked for.
  params.set<std::string>("output_field_name",expr);
  // ...and mark how it was resolved, since an expression is not a usable NetCDF
  // variable name and so must be given an output name by whoever writes it.
  params.set<bool>("from_expression",true);

  return DiagnosticFactory::instance().create(diag_name,grid->get_comm(),params,grid);
}

} // namespace scream
