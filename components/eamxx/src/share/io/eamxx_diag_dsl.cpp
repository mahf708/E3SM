#include "share/io/eamxx_diag_dsl.hpp"

#include <edp/lexer.hpp>
#include <edp/parser.hpp>
#include <edp/tokens.hpp>

#include <map>
#include <type_traits>

namespace scream {
namespace diag_dsl {

using namespace edp::ast;

namespace {

// ---------------------------------------------------------------------------
// Node inspection
//
// Expression stores its variant privately and only exposes visit(), so testing
// "is this node a T" goes through a generic visitor. The generic lambda covers
// every alternative, so adding a new one to ExpressionVariant does not break
// these (unlike an exhaustive overload set).
// ---------------------------------------------------------------------------

template <typename T>
const T* node_as (const Expression& e) {
  return e.visit([](const auto& n) -> const T* {
    if constexpr (std::is_same_v<std::decay_t<decltype(n)>,T>) {
      return &n;
    } else {
      return nullptr;
    }
  });
}

// ---------------------------------------------------------------------------
// Call arguments
// ---------------------------------------------------------------------------

// A keyword argument is parsed as an ordinary infix '=' expression whose left
// side is an identifier, so args have to be sorted out after the fact.
struct CallArgs {
  std::vector<const Expression*> positional;
  std::vector<std::pair<std::string,const Expression*>> keyword;

  const Expression* kw (const std::string& name) const {
    for (const auto& [k,v] : keyword) {
      if (k==name) return v;
    }
    return nullptr;
  }
  bool has_kw (const std::string& name) const { return kw(name)!=nullptr; }
};

CallArgs split_args (const std::vector<ExprPtr>& args, const std::string& what)
{
  CallArgs out;
  for (const auto& a : args) {
    const auto* inf = node_as<InfixExpression>(*a);
    if (inf and inf->op==edp::TokenTypes::Assign) {
      const auto* key = node_as<Identifier>(*inf->left);
      if (not key) {
        throw DslError("In '" + what + "': the left side of '=' must be a "
                       "keyword name.\n");
      }
      out.keyword.emplace_back(key->value,inf->right.get());
    } else {
      out.positional.push_back(a.get());
    }
  }
  return out;
}

void require_no_extra_kwargs (const CallArgs& args,
                              const std::vector<std::string>& allowed,
                              const std::string& what)
{
  for (const auto& [k,v] : args.keyword) {
    bool ok = false;
    for (const auto& a : allowed) {
      ok = ok or a==k;
    }
    if (not ok) {
      std::string msg = "In '" + what + "': unrecognized keyword argument '" + k + "'.\n";
      msg += " - accepted here: ";
      for (size_t i=0; i<allowed.size(); ++i) {
        msg += (i?", ":"") + allowed[i];
      }
      msg += "\n";
      throw DslError(msg);
    }
  }
}

// ---------------------------------------------------------------------------
// Operand naming
// ---------------------------------------------------------------------------

// The name by which an operand is referenced as an input to its parent. A bare
// identifier is used as-is (it is a model field, a constant, or a named diag);
// anything else is named by its canonical form and created on demand.
std::string operand_name (const Expression& e)
{
  if (const auto* id = node_as<Identifier>(e)) {
    return id->value;
  }
  return canonical(e);
}

// Text of a literal, for params that take a number or a string verbatim.
std::string literal_text (const Expression& e, const std::string& what)
{
  if (const auto* s = node_as<StringLiteral>(e)) return s->value;
  if (const auto* i = node_as<IntegerLiteral>(e)) return std::to_string(i->value);
  if (node_as<FloatLiteral>(e)) return canonical(e);
  throw DslError("In '" + what + "': expected a literal value.\n");
}

int int_literal (const Expression& e, const std::string& what)
{
  if (const auto* i = node_as<IntegerLiteral>(e)) return i->value;
  // A negative literal arrives as a prefix '-' applied to a positive one.
  if (const auto* p = node_as<PrefixExpression>(e)) {
    if (p->op==edp::TokenTypes::Minus) {
      if (const auto* i = node_as<IntegerLiteral>(*p->right)) return -i->value;
    }
  }
  throw DslError("In '" + what + "': expected an integer.\n");
}

// ---------------------------------------------------------------------------
// Comparison operators, for where()
// ---------------------------------------------------------------------------

const char* cmp_keyword (edp::TokenTypes op)
{
  switch (op) {
    case edp::TokenTypes::GreaterThan:  return "gt";
    case edp::TokenTypes::GreaterEqual: return "ge";
    case edp::TokenTypes::Equal:        return "eq";
    case edp::TokenTypes::NotEqual:     return "ne";
    case edp::TokenTypes::LessEq:       return "le";
    case edp::TokenTypes::LessThan:     return "lt";
    default:                            return nullptr;
  }
}

const char* binary_op_keyword (edp::TokenTypes op)
{
  switch (op) {
    case edp::TokenTypes::Plus:     return "plus";
    case edp::TokenTypes::Minus:    return "minus";
    case edp::TokenTypes::Asterisk: return "times";
    case edp::TokenTypes::Slash:    return "over";
    default:                        return nullptr;
  }
}

// ---------------------------------------------------------------------------
// Method-call handling
// ---------------------------------------------------------------------------

// A `.weighted(W)` receiver is not a diagnostic on its own -- mirroring xarray,
// it only has meaning as a modifier on the reduction that follows. If `recv` is
// such a call, strip it, hand back the real receiver and the weighting name.
struct Weighting {
  const Expression* receiver;
  std::string       method;   // "dp", "dz", or "" when unweighted
};

// Forward decl: defined below, since unwrapping needs method parsing.
Weighting peel_weighting (const Expression& recv);

DiagSpec method_call (const Expression& recv,
                      const std::string& method,
                      const std::vector<ExprPtr>& raw_args,
                      const std::string& grid_name,
                      const std::string& whole);

// ---------------------------------------------------------------------------

DiagSpec reduction (const Expression& recv,
                    const std::string& method,
                    const CallArgs& args,
                    const std::string& grid_name,
                    const std::string& whole)
{
  require_no_extra_kwargs(args,{"dim"},whole);
  const auto* dim_e = args.kw("dim");
  if (not dim_e) {
    throw DslError("In '" + whole + "': ." + method + "() needs a dimension, "
                   "e.g. ." + method + "(dim='lev') or ." + method + "(dim='col').\n");
  }
  const auto* dim_s = node_as<StringLiteral>(*dim_e);
  if (not dim_s) {
    throw DslError("In '" + whole + "': dim must be a quoted string, e.g. dim='lev'.\n");
  }
  const auto& dim = dim_s->value;

  auto w = peel_weighting(recv);

  DiagSpec spec;
  spec.set("grid_name",grid_name);

  if (dim=="col" or dim=="ncol") {
    if (method!="mean") {
      throw DslError("In '" + whole + "': only .mean(dim='col') is available "
                     "over the column dimension.\n");
    }
    if (not w.method.empty()) {
      throw DslError("In '" + whole + "': .weighted() is not supported for "
                     "horizontal averages.\n");
    }
    spec.diag_name = "HorizAvg";
    spec.set("field_name",operand_name(*w.receiver));
    return spec;
  }

  if (dim!="lev") {
    throw DslError("In '" + whole + "': unrecognized dimension '" + dim +
                   "'. Expected 'lev' (vertical) or 'col' (horizontal).\n");
  }

  std::string contract;
  if      (method=="mean") contract = "avg";
  else if (method=="sum")  contract = "sum";
  else {
    // min/max/std/var exist on a branch but are not on master yet.
    throw DslError("In '" + whole + "': ." + method + "(dim='lev') is not "
                   "available yet. VertContract currently supports only "
                   "'mean' and 'sum'.\n");
  }

  spec.diag_name = "VertContract";
  spec.set("field_name",operand_name(*w.receiver));
  spec.set("contract_method",contract);
  if (not w.method.empty()) {
    spec.set("weighting_method",w.method);
  }
  return spec;
}

DiagSpec select (const Expression& recv,
                 const CallArgs& args,
                 const std::string& grid_name,
                 const std::string& whole)
{
  require_no_extra_kwargs(args,{"lev"},whole);
  const auto* lev = args.kw("lev");
  if (not lev) {
    throw DslError("In '" + whole + "': .isel() needs a level, e.g. "
                   ".isel(lev=10) or .isel(lev=-1) for the bottom level.\n");
  }
  const int n = int_literal(*lev,whole);

  std::string location;
  if (n>=0) {
    location = "lev_" + std::to_string(n);
  } else if (n==-1) {
    // Only the last level can be named without knowing the layout, which the
    // translation layer does not have. FieldAtLevel resolves 'model_bot'.
    location = "model_bot";
  } else {
    throw DslError("In '" + whole + "': only lev=-1 is supported among negative "
                   "indices (it selects the bottom level). Use a non-negative "
                   "index for anything else.\n");
  }

  DiagSpec spec;
  spec.diag_name = "FieldAtLevel";
  spec.set("grid_name",grid_name);
  spec.set("field_name",operand_name(recv));
  spec.set("vertical_location",location);
  return spec;
}

DiagSpec interp (const Expression& recv,
                 const CallArgs& args,
                 const std::string& grid_name,
                 const std::string& whole)
{
  const bool by_p = args.has_kw("plev");
  const bool by_z = args.has_kw("z");
  if (by_p==by_z) {
    throw DslError("In '" + whole + "': .interp() needs exactly one of "
                   "plev= (pressure) or z= (height).\n");
  }

  DiagSpec spec;
  spec.set("grid_name",grid_name);
  spec.set("field_name",operand_name(recv));

  if (by_p) {
    require_no_extra_kwargs(args,{"plev","units"},whole);
    spec.diag_name = "FieldAtPressureLevel";
    spec.set("pressure_value",literal_text(*args.kw("plev"),whole));
    std::string units = "Pa";
    if (const auto* u = args.kw("units")) {
      units = literal_text(*u,whole);
      if (units!="Pa" and units!="hPa" and units!="mb") {
        throw DslError("In '" + whole + "': unrecognized pressure units '" +
                       units + "'. Expected 'Pa', 'hPa', or 'mb'.\n");
      }
    }
    spec.set("pressure_units",units);
    return spec;
  }

  require_no_extra_kwargs(args,{"z","units","reference"},whole);
  spec.diag_name = "FieldAtHeight";
  spec.set("height_value",literal_text(*args.kw("z"),whole));
  std::string units = "m";
  if (const auto* u = args.kw("units")) {
    units = literal_text(*u,whole);
    if (units!="m") {
      throw DslError("In '" + whole + "': unrecognized height units '" + units +
                     "'. Only 'm' is supported.\n");
    }
  }
  spec.set("height_units",units);
  std::string ref = "surface";
  if (const auto* r = args.kw("reference")) {
    ref = literal_text(*r,whole);
    if (ref!="surface" and ref!="sealevel") {
      throw DslError("In '" + whole + "': unrecognized reference '" + ref +
                     "'. Expected 'surface' or 'sealevel'.\n");
    }
  }
  spec.set("surface_reference",ref);
  return spec;
}

DiagSpec where (const Expression& recv,
                const CallArgs& args,
                const std::string& grid_name,
                const std::string& whole)
{
  if (args.positional.size()!=1 or not args.keyword.empty()) {
    throw DslError("In '" + whole + "': .where() takes exactly one condition, "
                   "e.g. .where(qc > 1e-5).\n");
  }
  const auto* cond = node_as<InfixExpression>(*args.positional.front());
  if (not cond) {
    throw DslError("In '" + whole + "': .where() needs a comparison, "
                   "e.g. .where(qc > 1e-5).\n");
  }
  if (cond->op==edp::TokenTypes::And or cond->op==edp::TokenTypes::Or) {
    throw DslError("In '" + whole + "': compound conditions are not supported "
                   "yet. ConditionalSampling takes a single comparison; chain "
                   "two .where() calls instead.\n");
  }
  const char* cmp = cmp_keyword(cond->op);
  if (not cmp) {
    throw DslError("In '" + whole + "': .where() needs a comparison operator "
                   "(>, >=, ==, !=, <=, <).\n");
  }

  DiagSpec spec;
  spec.diag_name = "ConditionalSampling";
  spec.set("grid_name",grid_name);
  spec.set("field_name",operand_name(recv));
  spec.set("condition_lhs",operand_name(*cond->left));
  spec.set("condition_cmp",cmp);
  spec.set("condition_rhs",operand_name(*cond->right));
  return spec;
}

DiagSpec shift (const Expression& recv,
                const CallArgs& args,
                const std::string& grid_name,
                const std::string& whole)
{
  require_no_extra_kwargs(args,{"time"},whole);
  const auto* t = args.kw("time");
  if (not t or int_literal(*t,whole)!=1) {
    throw DslError("In '" + whole + "': only .shift(time=1) is supported "
                   "(the value at the previous timestep).\n");
  }
  DiagSpec spec;
  spec.diag_name = "FieldPrev";
  spec.set("grid_name",grid_name);
  spec.set("field_name",operand_name(recv));
  return spec;
}

DiagSpec differentiate (const Expression& recv,
                        const CallArgs& args,
                        const std::string& grid_name,
                        const std::string& whole)
{
  if (args.positional.size()!=1 or not args.keyword.empty()) {
    throw DslError("In '" + whole + "': .differentiate() takes one coordinate, "
                   "e.g. .differentiate('p') or .differentiate('z').\n");
  }
  const auto coord = literal_text(*args.positional.front(),whole);
  if (coord!="p" and coord!="z") {
    throw DslError("In '" + whole + "': unrecognized coordinate '" + coord +
                   "'. Expected 'p' or 'z'.\n");
  }
  DiagSpec spec;
  spec.diag_name = "VertDerivative";
  spec.set("grid_name",grid_name);
  spec.set("field_name",operand_name(recv));
  spec.set("derivative_method",coord);
  return spec;
}

DiagSpec histogram (const Expression& recv,
                    const CallArgs& args,
                    const std::string& grid_name,
                    const std::string& whole)
{
  require_no_extra_kwargs(args,{"bins"},whole);
  const auto* bins = args.kw("bins");
  if (not bins) {
    throw DslError("In '" + whole + "': .histogram() needs bin edges, "
                   "e.g. .histogram(bins=[0,1,2]).\n");
  }
  const auto* arr = node_as<ArrayExpression>(*bins);
  if (not arr or arr->elements.size()<2) {
    throw DslError("In '" + whole + "': bins must be a list of at least two "
                   "edges, e.g. bins=[0,1,2].\n");
  }
  // Histogram takes the edges as an underscore-joined string.
  std::string cfg;
  for (size_t i=0; i<arr->elements.size(); ++i) {
    cfg += (i?"_":"") + literal_text(*arr->elements[i],whole);
  }

  DiagSpec spec;
  spec.diag_name = "Histogram";
  spec.set("grid_name",grid_name);
  spec.set("field_name",operand_name(recv));
  spec.set("bin_configuration",cfg);
  return spec;
}

DiagSpec zonal_mean (const Expression& recv,
                     const CallArgs& args,
                     const std::string& grid_name,
                     const std::string& whole)
{
  require_no_extra_kwargs(args,{"bins"},whole);
  const auto* bins = args.kw("bins");
  if (not bins) {
    throw DslError("In '" + whole + "': .zonal_mean() needs a bin count, "
                   "e.g. .zonal_mean(bins=20).\n");
  }
  const int n = int_literal(*bins,whole);
  if (n<1) {
    throw DslError("In '" + whole + "': bins must be a positive integer.\n");
  }
  DiagSpec spec;
  spec.diag_name = "ZonalAvg";
  spec.set("grid_name",grid_name);
  spec.set("field_name",operand_name(recv));
  spec.set("number_of_zonal_bins",std::to_string(n));
  return spec;
}

// ---------------------------------------------------------------------------

Weighting peel_weighting (const Expression& recv)
{
  const auto* inf = node_as<InfixExpression>(recv);
  if (not inf or inf->op!=edp::TokenTypes::Dot) {
    return {&recv,""};
  }
  const auto* call = node_as<FuncExpression>(*inf->right);
  if (not call) {
    return {&recv,""};
  }
  const auto* name = node_as<Identifier>(*call->function);
  if (not name or name->value!="weighted") {
    return {&recv,""};
  }
  const auto args = split_args(call->args,"weighted");
  if (args.positional.size()!=1 or not args.keyword.empty()) {
    throw DslError("'.weighted()' takes one weight name, e.g. "
                   ".weighted('dp').\n");
  }
  const auto w = literal_text(*args.positional.front(),"weighted");
  if (w!="dp" and w!="dz") {
    throw DslError("Unrecognized weighting '" + w + "'. Expected 'dp' or 'dz'.\n");
  }
  return {inf->left.get(),w};
}

DiagSpec method_call (const Expression& recv,
                      const std::string& method,
                      const std::vector<ExprPtr>& raw_args,
                      const std::string& grid_name,
                      const std::string& whole)
{
  const auto args = split_args(raw_args,whole);

  if (method=="mean" or method=="sum" or method=="min" or method=="max" or
      method=="std" or method=="var") {
    return reduction(recv,method,args,grid_name,whole);
  }
  if (method=="isel")          return select(recv,args,grid_name,whole);
  if (method=="interp")        return interp(recv,args,grid_name,whole);
  if (method=="where")         return where(recv,args,grid_name,whole);
  if (method=="shift")         return shift(recv,args,grid_name,whole);
  if (method=="differentiate") return differentiate(recv,args,grid_name,whole);
  if (method=="histogram")     return histogram(recv,args,grid_name,whole);
  if (method=="zonal_mean")    return zonal_mean(recv,args,grid_name,whole);

  if (method=="tend") {
    if (not raw_args.empty()) {
      throw DslError("In '" + whole + "': .tend() takes no arguments.\n");
    }
    // Shorthand for the backward tendency. Expanded rather than special-cased
    // so it composes with everything else for free.
    const auto x = canonical(recv);
    DiagSpec spec;
    spec.rewrite_to = "(" + x + " - " + x + ".shift(time=1)) / dt";
    return spec;
  }

  if (method=="weighted") {
    throw DslError("In '" + whole + "': .weighted() only has meaning attached "
                   "to a reduction, e.g. .weighted('dp').mean(dim='lev').\n");
  }

  throw DslError("In '" + whole + "': unrecognized method '." + method + "()'.\n"
                 " - available: .mean, .sum, .isel, .interp, .where, .shift,\n"
                 "              .differentiate, .histogram, .zonal_mean,\n"
                 "              .weighted, .tend\n");
}

// ---------------------------------------------------------------------------

DiagSpec infix (const InfixExpression& e,
                const std::string& grid_name,
                const std::string& whole)
{
  // Method call: `recv . name(args)`. The call binds tighter than the dot, so
  // the right operand is the FuncExpression rather than a bare name.
  if (e.op==edp::TokenTypes::Dot) {
    const auto* call = node_as<FuncExpression>(*e.right);
    if (not call) {
      const auto* name = node_as<Identifier>(*e.right);
      if (name) {
        throw DslError("In '" + whole + "': '." + name->value + "' is not a "
                       "diagnostic. Did you mean '." + name->value + "()'?\n");
      }
      throw DslError("In '" + whole + "': expected a method call after '.'.\n");
    }
    const auto* name = node_as<Identifier>(*call->function);
    if (not name) {
      throw DslError("In '" + whole + "': expected a method name after '.'.\n");
    }
    return method_call(*e.left,name->value,call->args,grid_name,whole);
  }

  // Division by the reserved identifier `dt` is FieldOverDt rather than a
  // generic binary op.
  //
  // NOTE: this must be tested before the generic arithmetic case below, the
  //       same ordering constraint the old regexes had. The difference is that
  //       it is now one explicit branch keyed on an identifier, rather than an
  //       implicit dependency between the order of two patterns.
  if (e.op==edp::TokenTypes::Slash) {
    const auto* rhs = node_as<Identifier>(*e.right);
    if (rhs and rhs->value=="dt") {
      DiagSpec spec;
      spec.diag_name = "FieldOverDt";
      spec.set("grid_name",grid_name);
      spec.set("field_name",operand_name(*e.left));
      return spec;
    }
  }

  if (const char* op = binary_op_keyword(e.op)) {
    DiagSpec spec;
    spec.diag_name = "BinaryOp";
    spec.set("grid_name",grid_name);
    spec.set("arg1",operand_name(*e.left));
    spec.set("arg2",operand_name(*e.right));
    spec.set("binary_op",op);
    // NOTE: an operand may name a physical constant rather than a field.
    //       BinaryOp sorts that out itself against the physics-constants
    //       dictionary, which is exactly why the spec carries no input list.
    return spec;
  }

  if (cmp_keyword(e.op)) {
    throw DslError("In '" + whole + "': a comparison is not a diagnostic by "
                   "itself. Use it inside .where(), e.g. X.where(" + whole + ").\n");
  }
  if (e.op==edp::TokenTypes::Assign) {
    throw DslError("In '" + whole + "': '=' is only valid as a keyword argument "
                   "inside a call.\n");
  }
  if (e.op==edp::TokenTypes::Exp) {
    throw DslError("In '" + whole + "': '**' is not available yet; it needs the "
                   "UnaryOps diagnostic.\n");
  }
  throw DslError("In '" + whole + "': unsupported operator.\n");
}

} // anonymous namespace

// ---------------------------------------------------------------------------

std::string canonical (const Expression& e)
{
  return to_string(e);
}

std::optional<std::string> bare_name (const Expression& e)
{
  if (const auto* id = node_as<Identifier>(e)) {
    return id->value;
  }
  return std::nullopt;
}

std::string canonical (const std::string& expr)
{
  edp::parser::Parser p{edp::Lexer{expr}};
  return to_string(*p.parse());
}

DiagSpec spec_from_ast (const Expression& e, const std::string& grid_name)
{
  const auto whole = canonical(e);

  if (const auto* inf = node_as<InfixExpression>(e)) {
    return infix(*inf,grid_name,whole);
  }

  if (const auto* id = node_as<Identifier>(e)) {
    // A bare name is not translated here: it is either a model field, or a
    // named diagnostic that the caller resolves against the factory and the
    // named-diagnostic table. Report it so the caller can do that.
    DiagSpec spec;
    spec.diag_name = id->value;
    spec.set("grid_name",grid_name);
    return spec;
  }

  if (const auto* call = node_as<FuncExpression>(e)) {
    const auto* name = node_as<Identifier>(*call->function);
    const std::string fn = name ? name->value : std::string("<expression>");
    if (fn=="log" or fn=="exp" or fn=="sqrt" or fn=="abs" or fn=="square" or
        fn=="inverse") {
      throw DslError("In '" + whole + "': '" + fn + "()' is not available yet; "
                     "it needs the UnaryOps diagnostic.\n");
    }
    throw DslError("In '" + whole + "': unrecognized function '" + fn + "()'.\n");
  }

  if (const auto* pre = node_as<PrefixExpression>(e)) {
    if (pre->op==edp::TokenTypes::Minus) {
      throw DslError("In '" + whole + "': negation is not available yet; it "
                     "needs the UnaryOps diagnostic.\n");
    }
    throw DslError("In '" + whole + "': 'not' is only valid inside .where().\n");
  }

  throw DslError("'" + whole + "' is not a diagnostic. A literal, list, or "
                 "slice cannot stand on its own.\n");
}

} // namespace diag_dsl
} // namespace scream
