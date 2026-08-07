#ifndef SCREAM_DIAG_DSL_HPP
#define SCREAM_DIAG_DSL_HPP

#include <edp/ast.hpp>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace scream {
namespace diag_dsl {

/*
 * Translation from a parsed diagnostics-DSL expression to the information
 * needed to build a diagnostic: a factory key plus a set of parameters.
 *
 * NOTE: this header and its implementation deliberately depend on nothing but
 *       the vendored parser (edp) and the C++ standard library -- no EKAT, no
 *       Kokkos, no EAMxx types. Every diagnostic parameter is a string (see
 *       the m_params.get<std::string> call sites in share/diagnostics), so
 *       there is nothing here that needs ekat::ParameterList. Keeping it free
 *       of EAMxx lets the whole translation be unit tested without a model
 *       build, which is most of its value. The thin conversion of a DiagSpec
 *       into an ekat::ParameterList lives in eamxx_io_utils.cpp.
 */

// Thrown when an expression parses but does not describe a diagnostic we can
// build. The message is user-facing: it ends up in front of whoever wrote the
// output YAML, so it should say what is wrong and what to write instead.
class DslError : public std::runtime_error {
public:
  explicit DslError (const std::string& msg) : std::runtime_error(msg) {}
};

struct DiagSpec {
  // Key the diagnostic factory is registered under, e.g. "VertContract".
  std::string diag_name;

  // Parameters to hand the diagnostic, in insertion order. All values are
  // strings; the diags parse them themselves.
  std::vector<std::pair<std::string,std::string>> params;

  // NOTE: there is deliberately no list of input fields here. Every diagnostic
  //       already derives its own m_field_in_names from these params, and some
  //       add more than the expression mentions (VertContract pulls in
  //       pseudo_density, qv, p_mid and T_mid when weighted). Duplicating that
  //       here would be a partial copy that drifts.

  // When non-empty, this expression is shorthand: the caller should parse this
  // DSL string instead and translate that. Used for forms that expand to a
  // composition of other diagnostics (e.g. `X.tend()`).
  std::string rewrite_to;

  void set (const std::string& k, const std::string& v) {
    params.emplace_back(k,v);
  }
};

// Canonical string form of an expression. This is the identity of a
// diagnostic: it is what a parent names its children, and what a diag is told
// to call its output field, so that the two match when the IO layer resolves
// dependencies by name.
//
// It must be a fixed point -- canonical(parse(canonical(parse(s)))) ==
// canonical(parse(s)) -- or that resolution never converges.
std::string canonical (const edp::ast::Expression& e);

// Convenience: parse and canonicalize in one step. Throws edp ParserError.
std::string canonical (const std::string& expr);

// Translate the outermost operation of an expression into a DiagSpec.
//
// This does NOT recurse into sub-expressions to build them: a composite
// operand is referred to by its canonical name and left for the IO layer to
// create on demand, which is how diagnostics already compose.
//
// `grid_name` is stored in the "grid_name" param, as the existing diags expect.
DiagSpec spec_from_ast (const edp::ast::Expression& e,
                        const std::string& grid_name);

} // namespace diag_dsl
} // namespace scream

#endif // SCREAM_DIAG_DSL_HPP
