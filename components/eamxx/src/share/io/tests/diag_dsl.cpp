#include "catch2/catch.hpp"

#include "share/io/eamxx_diag_dsl.hpp"

#include <edp/lexer.hpp>
#include <edp/parser.hpp>

#include <string>

namespace scream {
namespace diag_dsl {

namespace {

DiagSpec translate (const std::string& expr) {
  edp::parser::Parser p{edp::Lexer{expr}};
  return spec_from_ast(*p.parse(),"physics");
}

// Value of a param, or "" if the spec does not set it.
std::string param (const DiagSpec& s, const std::string& key) {
  for (const auto& [k,v] : s.params) {
    if (k==key) return v;
  }
  return "";
}

bool has_param (const DiagSpec& s, const std::string& key) {
  for (const auto& [k,v] : s.params) {
    if (k==key) return true;
  }
  return false;
}

} // anonymous namespace

TEST_CASE("dsl: vertical and horizontal reductions") {
  auto s = translate("T_mid.mean(dim='lev')");
  REQUIRE(s.diag_name=="VertContract");
  REQUIRE(param(s,"field_name")=="T_mid");
  REQUIRE(param(s,"contract_method")=="avg");
  // Unweighted reductions must not set weighting_method at all, so that
  // VertContract falls back to its own "none" default.
  REQUIRE_FALSE(has_param(s,"weighting_method"));

  REQUIRE(param(translate("T_mid.sum(dim='lev')"),"contract_method")=="sum");

  auto h = translate("T_mid.mean(dim='col')");
  REQUIRE(h.diag_name=="HorizAvg");
  REQUIRE(param(h,"field_name")=="T_mid");
}

TEST_CASE("dsl: .weighted() folds into the reduction that follows it") {
  auto s = translate("T_mid.weighted('dp').mean(dim='lev')");
  REQUIRE(s.diag_name=="VertContract");
  REQUIRE(param(s,"field_name")=="T_mid");
  REQUIRE(param(s,"contract_method")=="avg");
  REQUIRE(param(s,"weighting_method")=="dp");

  REQUIRE(param(translate("T_mid.weighted('dz').sum(dim='lev')"),
                "weighting_method")=="dz");

  // On its own it is not a diagnostic.
  REQUIRE_THROWS_AS(translate("T_mid.weighted('dp')"),DslError);
  // Nor is it meaningful on a horizontal average.
  REQUIRE_THROWS_AS(translate("T_mid.weighted('dp').mean(dim='col')"),DslError);
  REQUIRE_THROWS_AS(translate("T_mid.weighted('kg').mean(dim='lev')"),DslError);
}

TEST_CASE("dsl: level selection") {
  REQUIRE(param(translate("T_mid.isel(lev=10)"),"vertical_location")=="lev_10");
  REQUIRE(param(translate("T_mid.isel(lev=0)"),"vertical_location")=="lev_0");
  // Python's negative index for the last element; anything further back would
  // need the layout, which the translation layer does not have.
  REQUIRE(param(translate("T_mid.isel(lev=-1)"),"vertical_location")=="model_bot");
  REQUIRE_THROWS_AS(translate("T_mid.isel(lev=-3)"),DslError);
  REQUIRE_THROWS_AS(translate("T_mid.isel()"),DslError);
}

TEST_CASE("dsl: interpolation to pressure and height") {
  auto p = translate("T_mid.interp(plev=500, units='hPa')");
  REQUIRE(p.diag_name=="FieldAtPressureLevel");
  REQUIRE(param(p,"pressure_value")=="500");
  REQUIRE(param(p,"pressure_units")=="hPa");
  // Pa is the default, matching the SI-unit convention elsewhere.
  REQUIRE(param(translate("T_mid.interp(plev=50000)"),"pressure_units")=="Pa");

  auto z = translate("T_mid.interp(z=10, reference='surface')");
  REQUIRE(z.diag_name=="FieldAtHeight");
  REQUIRE(param(z,"height_value")=="10");
  REQUIRE(param(z,"height_units")=="m");
  REQUIRE(param(z,"surface_reference")=="surface");
  REQUIRE(param(translate("T_mid.interp(z=10, reference='sealevel')"),
                "surface_reference")=="sealevel");

  REQUIRE_THROWS_AS(translate("T_mid.interp(plev=500, units='atm')"),DslError);
  REQUIRE_THROWS_AS(translate("T_mid.interp(z=1, units='km')"),DslError);
  // Exactly one of plev=/z= is required.
  REQUIRE_THROWS_AS(translate("T_mid.interp()"),DslError);
  REQUIRE_THROWS_AS(translate("T_mid.interp(plev=500, z=10)"),DslError);
}

TEST_CASE("dsl: conditional sampling") {
  auto s = translate("T_mid.where(qc > 1e-5)");
  REQUIRE(s.diag_name=="ConditionalSampling");
  REQUIRE(param(s,"field_name")=="T_mid");
  REQUIRE(param(s,"condition_lhs")=="qc");
  REQUIRE(param(s,"condition_cmp")=="gt");

  REQUIRE(param(translate("T.where(q >= 1)"),"condition_cmp")=="ge");
  REQUIRE(param(translate("T.where(q == 1)"),"condition_cmp")=="eq");
  REQUIRE(param(translate("T.where(q != 1)"),"condition_cmp")=="ne");
  REQUIRE(param(translate("T.where(q <= 1)"),"condition_cmp")=="le");
  REQUIRE(param(translate("T.where(q < 1)"),"condition_cmp")=="lt");

  // ConditionalSampling takes a single comparison; say so rather than
  // silently dropping half the condition.
  REQUIRE_THROWS_AS(translate("T.where(q > 0 and r > 0)"),DslError);
  REQUIRE_THROWS_AS(translate("T.where(q)"),DslError);
}

TEST_CASE("dsl: remaining single-field operations") {
  REQUIRE(translate("T_mid.shift(time=1)").diag_name=="FieldPrev");
  REQUIRE_THROWS_AS(translate("T_mid.shift(time=2)"),DslError);

  auto d = translate("T_mid.differentiate('p')");
  REQUIRE(d.diag_name=="VertDerivative");
  REQUIRE(param(d,"derivative_method")=="p");
  REQUIRE(param(translate("T_mid.differentiate('z')"),"derivative_method")=="z");
  REQUIRE_THROWS_AS(translate("T_mid.differentiate('q')"),DslError);

  auto hi = translate("T_mid.histogram(bins=[0,1,2])");
  REQUIRE(hi.diag_name=="Histogram");
  // Histogram wants the edges underscore-joined.
  REQUIRE(param(hi,"bin_configuration")=="0_1_2");
  REQUIRE_THROWS_AS(translate("T_mid.histogram(bins=[0])"),DslError);

  auto z = translate("T_mid.zonal_mean(bins=20)");
  REQUIRE(z.diag_name=="ZonalAvg");
  REQUIRE(param(z,"number_of_zonal_bins")=="20");
  REQUIRE_THROWS_AS(translate("T_mid.zonal_mean(bins=0)"),DslError);
}

TEST_CASE("dsl: arithmetic") {
  auto s = translate("qc + qr");
  REQUIRE(s.diag_name=="BinaryOp");
  REQUIRE(param(s,"arg1")=="qc");
  REQUIRE(param(s,"arg2")=="qr");
  REQUIRE(param(s,"binary_op")=="plus");

  REQUIRE(param(translate("a - b"),"binary_op")=="minus");
  REQUIRE(param(translate("a * b"),"binary_op")=="times");
  REQUIRE(param(translate("a / b"),"binary_op")=="over");

  // A physical constant is just a name; BinaryOp resolves it itself.
  REQUIRE(param(translate("Rgas * T_mid"),"arg1")=="Rgas");

  // Grouping decides the shape, rather than a greedy leftmost-operand rule.
  // These two used to be indistinguishable.
  auto left  = translate("(A - B) / C");
  auto right = translate("A - (B / C)");
  REQUIRE(param(left,"binary_op")=="over");
  REQUIRE(param(right,"binary_op")=="minus");
}

TEST_CASE("dsl: division by dt is FieldOverDt, not a binary op") {
  auto s = translate("T_mid / dt");
  REQUIRE(s.diag_name=="FieldOverDt");
  REQUIRE(param(s,"field_name")=="T_mid");

  // Any other divisor is ordinary division.
  REQUIRE(translate("T_mid / p_mid").diag_name=="BinaryOp");
}

TEST_CASE("dsl: composite operands are named by canonical form") {
  // A nested expression is not built here: it is named, and the IO layer
  // creates it on demand. The name must be the canonical form so that the
  // diag created for it later reports the very same name.
  auto s = translate("T_mid.mean(dim='lev').isel(lev=0)");
  REQUIRE(s.diag_name=="FieldAtLevel");
  const auto child = param(s,"field_name");
  REQUIRE(child==canonical("T_mid.mean(dim='lev')"));

  // Translating that child yields the inner diagnostic, closing the loop.
  auto inner = translate(child);
  REQUIRE(inner.diag_name=="VertContract");
  REQUIRE(param(inner,"field_name")=="T_mid");
}

TEST_CASE("dsl: canonical form is a fixed point") {
  // The dependency resolution in scorpio_output walks name -> diag -> name,
  // and only terminates if canonicalizing is idempotent.
  for (const std::string e : {
        "T_mid",
        "T_mid.weighted('dp').mean(dim='lev')",
        "T_mid.isel(lev=-1)",
        "(A - B) / C",
        "A - B / C",
        "T.where(qc > 0.5)",
        "T.histogram(bins=[0,1,2])",
        "T.interp(plev=500.0, units='hPa')",
        "T_mid.mean(dim='lev').isel(lev=0)"}) {
    const auto c1 = canonical(e);
    INFO("expression: " << e << " -> " << c1);
    REQUIRE(canonical(c1)==c1);
  }
}

TEST_CASE("dsl: .tend() expands rather than being special-cased") {
  auto s = translate("T_mid.tend()");
  REQUIRE(s.diag_name.empty());
  REQUIRE_FALSE(s.rewrite_to.empty());

  // The expansion must itself be translatable, and must land on FieldOverDt.
  auto expanded = translate(s.rewrite_to);
  REQUIRE(expanded.diag_name=="FieldOverDt");

  REQUIRE_THROWS_AS(translate("T_mid.tend(1)"),DslError);
}

TEST_CASE("dsl: a bare name is handed back for the caller to resolve") {
  // Identifier resolution needs the diagnostic factory and the named-diagnostic
  // table, neither of which this layer knows about.
  auto s = translate("LiqWaterPath");
  REQUIRE(s.diag_name=="LiqWaterPath");
  REQUIRE(param(s,"grid_name")=="physics");
}

TEST_CASE("dsl: unsupported forms report what is missing") {
  // Operations whose diagnostic is not on master yet must say so, rather than
  // falling through to a confusing 'unknown diagnostic' from the factory.
  REQUIRE_THROWS_AS(translate("T_mid.min(dim='lev')"),DslError);
  REQUIRE_THROWS_AS(translate("T_mid.std(dim='lev')"),DslError);
  REQUIRE_THROWS_AS(translate("abs(T_mid)"),DslError);
  REQUIRE_THROWS_AS(translate("-T_mid"),DslError);
  REQUIRE_THROWS_AS(translate("T_mid ** 2"),DslError);

  // Typos, and things that are not diagnostics at all.
  REQUIRE_THROWS_AS(translate("T_mid.man(dim='lev')"),DslError);
  REQUIRE_THROWS_AS(translate("T_mid.mean(dims='lev')"),DslError);
  REQUIRE_THROWS_AS(translate("qc > 0"),DslError);
  REQUIRE_THROWS_AS(translate("1:10"),DslError);
  REQUIRE_THROWS_AS(translate("[1,2]"),DslError);
}

} // namespace diag_dsl
} // namespace scream
