#include "share/io/eamxx_diag_names.hpp"

#include <edp/lexer.hpp>
#include <edp/parser.hpp>

#include <map>
#include <regex>
#include <utility>

namespace scream {
namespace diag_dsl {

namespace {

DiagSpec make (const std::string& diag_name,
               const std::string& grid_name,
               std::initializer_list<std::pair<std::string,std::string>> params)
{
  DiagSpec s;
  s.diag_name = diag_name;
  s.set("grid_name",grid_name);
  for (const auto& [k,v] : params) {
    s.set(k,v);
  }
  return s;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Permanent: canonical names of specific diagnostics
// ---------------------------------------------------------------------------

std::optional<DiagSpec> named_diagnostic (const std::string& name,
                                          const std::string& grid_name)
{
  std::smatch m;

  static const std::regex water_path  ("^(Ice|Liq|Rain|Rime|Vap)WaterPath$");
  static const std::regex number_path ("^(Ice|Liq|Rain)NumberPath$");
  static const std::regex vap_flux    ("^(Meridional|Zonal)VapFlux$");
  static const std::regex pot_temp    ("^(Liq)?PotentialTemperature$");
  static const std::regex precip_flux ("^precip_(liq|ice|total)_surf_mass_flux$");
  static const std::regex aerocom_cld ("^AeroComCld(Top|Bot)$");
  static const std::regex vert_layer  ("^(z|geopotential|height)_(mid|int)$");

  if (std::regex_match(name,m,water_path)) {
    return make("WaterPath",grid_name,{{"water_kind",m[1].str()}});
  }
  if (std::regex_match(name,m,number_path)) {
    return make("NumberPath",grid_name,{{"number_kind",m[1].str()}});
  }
  if (std::regex_match(name,m,vap_flux)) {
    return make("VaporFlux",grid_name,{{"wind_component",m[1].str()}});
  }
  if (std::regex_match(name,m,pot_temp)) {
    // A bare "PotentialTemperature" is the total form.
    return make("PotentialTemperature",grid_name,
                {{"temperature_kind",m[1].matched ? m[1].str() : std::string("Tot")}});
  }
  if (std::regex_match(name,m,precip_flux)) {
    return make("precip_surf_mass_flux",grid_name,{{"precip_type",m[1].str()}});
  }
  if (std::regex_match(name,m,aerocom_cld)) {
    return make("AeroComCld",grid_name,{{"aero_com_cld_kind",m[1].str()}});
  }
  if (std::regex_match(name,m,vert_layer)) {
    return make("VerticalLayer",grid_name,
                {{"diag_name",m[1].str()},{"vert_location",m[2].str()}});
  }
  if (name=="dz") {
    return make("VerticalLayer",grid_name,
                {{"diag_name","dz"},{"vert_location","mid"}});
  }

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Deprecated: the old composite name syntax
// ---------------------------------------------------------------------------

std::optional<std::string> legacy_to_dsl (const std::string& name)
{
  // A field name may contain letters, digits, underscore, dot, and the
  // arithmetic characters that legacy composite names could produce.
  static const std::string F = R"(([A-Za-z0-9_.+\-\*\xC3\xB7]+))";

  static const std::regex backtend   ("^" + F + "_atm_backtend$");
  static const std::regex at_level   ("^" + F + R"(_at_(?:lev_(\d+)|model_(top|bot))$)");
  static const std::regex at_press   ("^" + F + R"(_at_(\d+(?:\.\d+)?)(hPa|mb|Pa)$)");
  static const std::regex at_height  ("^" + F + R"(_at_(\d+(?:\.\d+)?)m_above_(sealevel|surface)$)");
  static const std::regex over_dt    ("^" + F + "_over_dt$");
  static const std::regex horiz_avg  ("^" + F + "_horiz_avg$");
  static const std::regex vert_contr ("^" + F + R"(_vert_(avg|sum)(?:_(dp|dz)_weighted)?$)");
  static const std::regex zonal_avg  ("^" + F + R"(_zonal_avg_(\d+)_bins$)");
  static const std::regex cond_samp  ("^" + F + "_where_" + F + R"(_(gt|ge|eq|ne|le|lt)_)" + F + "$");
  static const std::regex binary_ops ("^" + F + "_(plus|minus|times|over)_" + F + "$");
  static const std::regex field_prev ("^" + F + "_prev$");
  static const std::regex histogram  ("^" + F + R"(_histogram_((?:\d+(?:\.\d+)?)(?:_\d+(?:\.\d+)?)+)$)");
  static const std::regex vert_deriv ("^" + F + "_(p|z)vert_derivative$");

  std::smatch m;

  // NOTE: the order below reproduces the precedence the regexes in
  //       create_diagnostic used to have, and it is load-bearing in three
  //       places. It is preserved here only to keep old names meaning exactly
  //       what they used to mean; new expressions say what they mean with
  //       parentheses instead, and none of this ordering applies to them.

  if (std::regex_match(name,m,backtend)) {
    return m[1].str() + ".tend()";
  }
  if (std::regex_match(name,m,at_level)) {
    if (m[2].matched) return m[1].str() + ".isel(lev=" + m[2].str() + ")";
    return m[1].str() + (m[3].str()=="top" ? ".isel(lev=0)" : ".isel(lev=-1)");
  }
  if (std::regex_match(name,m,at_press)) {
    return m[1].str() + ".interp(plev=" + m[2].str() + ", units='" + m[3].str() + "')";
  }
  if (std::regex_match(name,m,at_height)) {
    return m[1].str() + ".interp(z=" + m[2].str() + ", reference='" + m[3].str() + "')";
  }
  // NOTE (1/3): before binary ops, or "X_over_dt" reads as BinaryOp(X,over,dt).
  if (std::regex_match(name,m,over_dt)) {
    return m[1].str() + " / dt";
  }
  if (std::regex_match(name,m,horiz_avg)) {
    return m[1].str() + ".mean(dim='col')";
  }
  if (std::regex_match(name,m,vert_contr)) {
    const std::string method = m[2].str()=="avg" ? "mean" : "sum";
    std::string out = m[1].str();
    if (m[3].matched) out += ".weighted('" + m[3].str() + "')";
    return out + "." + method + "(dim='lev')";
  }
  if (std::regex_match(name,m,zonal_avg)) {
    return m[1].str() + ".zonal_mean(bins=" + m[2].str() + ")";
  }
  if (std::regex_match(name,m,cond_samp)) {
    static const std::map<std::string,std::string> sym{
      {"gt",">"},{"ge",">="},{"eq","=="},{"ne","!="},{"le","<="},{"lt","<"}};
    return m[1].str() + ".where(" + m[2].str() + " " +
           sym.at(m[3].str()) + " " + m[4].str() + ")";
  }
  // NOTE (2/3): the left operand is greedy, so for a name with several
  //             operator words the RIGHTMOST becomes the outermost operation:
  //             "A_minus_B_over_C" is (A-B)/C. That asymmetry is exactly why
  //             the DSL exists; it is reproduced here for old names only.
  if (std::regex_match(name,m,binary_ops)) {
    static const std::map<std::string,std::string> sym{
      {"plus","+"},{"minus","-"},{"times","*"},{"over","/"}};
    return m[1].str() + " " + sym.at(m[2].str()) + " " + m[3].str();
  }
  // NOTE (3/3): after binary ops, so "X_minus_X_prev" is BinaryOp(X,minus,
  //             X_prev) rather than FieldPrev(X_minus_X).
  if (std::regex_match(name,m,field_prev)) {
    return m[1].str() + ".shift(time=1)";
  }
  if (std::regex_match(name,m,histogram)) {
    std::string bins;
    std::string cfg = m[2].str();
    size_t pos = 0;
    while (pos<=cfg.size()) {
      const auto next = cfg.find('_',pos);
      const auto tok = cfg.substr(pos,next==std::string::npos ? std::string::npos : next-pos);
      bins += (bins.empty() ? "" : ",") + tok;
      if (next==std::string::npos) break;
      pos = next+1;
    }
    return m[1].str() + ".histogram(bins=[" + bins + "])";
  }
  if (std::regex_match(name,m,vert_deriv)) {
    return m[1].str() + ".differentiate('" + m[2].str() + "')";
  }

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

DiagSpec resolve (const std::string& request,
                  const std::string& grid_name,
                  const std::function<bool(const std::string&)>& is_registered)
{
  // Shorthand forms rewrite to another expression rather than resolving
  // directly, so this loops. The bound is a safety net against a rewrite rule
  // that cycles; the rules here nest at most two deep (a legacy backtend name
  // becomes X.tend(), which becomes the subtraction over dt).
  constexpr int max_rewrites = 8;

  std::string expr_str = request;

  for (int i=0; i<max_rewrites; ++i) {
    edp::parser::Parser parser {edp::Lexer{expr_str}};
    const auto expr = parser.parse();

    if (const auto name = bare_name(*expr)) {
      // A single name: a diagnostic in its own right, a canonical named
      // diagnostic, or an old composite name -- in that order, so a genuinely
      // registered name always wins.
      if (is_registered(*name)) {
        DiagSpec spec;
        spec.diag_name = *name;
        spec.set("grid_name",grid_name);
        return spec;
      }
      if (auto named = named_diagnostic(*name,grid_name)) {
        return *named;
      }
      if (auto legacy = legacy_to_dsl(*name)) {
        expr_str = *legacy;
        continue;
      }
      throw DslError(
          "Unknown field or diagnostic: '" + *name + "'.\n" +
          (expr_str==request ? std::string("")
                             : " - requested as: " + request + "\n") +
          " If this is meant to be a diagnostic, check the spelling against the\n"
          " registered diagnostics; if it is meant to be a model field, check\n"
          " that it is present on grid '" + grid_name + "'.\n");
    }

    auto spec = spec_from_ast(*expr,grid_name);
    if (spec.rewrite_to.empty()) {
      return spec;
    }
    expr_str = spec.rewrite_to;
  }

  throw DslError("Gave up expanding '" + request + "' after " +
                 std::to_string(max_rewrites) + " rewrites.\n"
                 " - last form: " + expr_str + "\n");
}

} // namespace diag_dsl
} // namespace scream
