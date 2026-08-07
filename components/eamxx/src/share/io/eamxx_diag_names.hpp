#ifndef SCREAM_DIAG_NAMES_HPP
#define SCREAM_DIAG_NAMES_HPP

#include "share/io/eamxx_diag_dsl.hpp"

#include <optional>
#include <string>

namespace scream {
namespace diag_dsl {

/*
 * Resolution of bare identifiers, which the DSL translation deliberately does
 * not attempt on its own.
 *
 * There are two very different kinds of name here, and they have different
 * futures:
 *
 *  - named_diagnostic() is PERMANENT. "LiqWaterPath", "z_mid", "dz",
 *    "MeridionalVapFlux" are the canonical, user-facing names of particular
 *    diagnostics. They are not syntax and the DSL does not replace them. They
 *    need a lookup only because the factory is keyed on the class name
 *    ("WaterPath") while the parameter that picks the variant ("water_kind")
 *    has to come from somewhere.
 *
 *  - legacy_to_dsl() is DEPRECATED, and exists to keep existing output YAML
 *    working. It rewrites the old composite name syntax ("T_mid_at_500hPa")
 *    into the equivalent DSL string, which the caller re-parses. It is a pure
 *    syntactic rewrite: it extracts no parameters and knows nothing about the
 *    diagnostic factory, so the whole thing can be deleted in one commit when
 *    the deprecation window closes.
 *
 * Like the rest of this library, neither depends on any EAMxx type.
 */

// Look up a canonical named diagnostic. Returns nothing if the name is not one.
std::optional<DiagSpec> named_diagnostic (const std::string& name,
                                          const std::string& grid_name);

// Rewrite a legacy composite name into the equivalent DSL expression.
// Returns nothing if the name is not in the legacy syntax.
//
// The result may itself contain legacy sub-names: "f_minus_f_prev_over_dt"
// rewrites to "f_minus_f_prev / dt", leaving "f_minus_f_prev" to be resolved
// when the IO layer asks for it. That recursion is what preserves the existing
// netCDF variable names for intermediate quantities.
std::optional<std::string> legacy_to_dsl (const std::string& name);

} // namespace diag_dsl
} // namespace scream

#endif // SCREAM_DIAG_NAMES_HPP
