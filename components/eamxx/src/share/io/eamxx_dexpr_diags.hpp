#ifndef SCREAM_EAMXX_DEXPR_DIAGS_HPP
#define SCREAM_EAMXX_DEXPR_DIAGS_HPP

#include "share/diagnostics/abstract_diagnostic.hpp"
#include "share/grid/abstract_grid.hpp"

#include <memory>
#include <string>
#include <vector>

namespace scream {

/*
 * The expression front end for diagnostics.
 *
 * This is an alternative to the regex patterns in create_diagnostic(), not a
 * layer on top of them: it parses the request with share/dexpr and drives the
 * diagnostic factory straight from the resulting AST. Nothing here produces or
 * consumes the legacy underscore names, so the two front ends can evolve (and
 * the legacy one can eventually be removed) independently.
 *
 * Composition works the same way it always has, by field name: each diag built
 * here declares its operands using their canonical expression strings, and the
 * customer resolving diag dependencies feeds those back through
 * create_diagnostic(). E.g. '(qc+qv)*p_mid' produces
 * BinaryOp{arg1="(qc+qv)", arg2="p_mid"}, and '(qc+qv)' comes back around to be
 * built here as well. Those canonical names carry parentheses, quotes and
 * commas, which the legacy patterns cannot match, so a name minted here cannot
 * be intercepted by a regex on the way back in.
 *
 * Diags created here carry two extra params:
 *   - 'output_field_name': the expression, so the diag's output field is named
 *     after the request rather than after the diag's own param concatenation.
 *   - 'from_expression': a marker, so customers can tell that a name was
 *     resolved as an expression (expressions are not usable NetCDF variable
 *     names, so they must be given an output name via 'name := expr').
 *
 * NOTE: this header deliberately exposes no dexpr types, so that dexpr (and its
 *       C++20 requirement) stays private to the eamxx_io library.
 */

// Returns nullptr if 'expr' is a plain identifier, i.e. not an expression at
// all: that is a diagnostic class name (or a typo), and the caller should
// resolve it as it always has.
// Throws if 'expr' is an expression we cannot parse, validate, or translate.
std::shared_ptr<AbstractDiagnostic>
dexpr_create_diagnostic (const std::string& expr,
                         const std::shared_ptr<const AbstractGrid>& grid);

// One example call per supported function, e.g. "T_mid.isel(lev=10)". dexpr
// already checks each example against the spec that declared it; this exposes
// them so a test can go one step further and prove every registered function is
// actually buildable. A function registered without a matching case in the
// translator is otherwise only caught when a user happens to write that call.
std::vector<std::string> dexpr_diagnostic_examples ();

} // namespace scream

#endif // SCREAM_EAMXX_DEXPR_DIAGS_HPP
