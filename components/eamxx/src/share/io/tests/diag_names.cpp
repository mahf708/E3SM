#include "catch2/catch.hpp"

#include "share/io/eamxx_diag_names.hpp"

#include <edp/lexer.hpp>
#include <edp/parser.hpp>

#include <string>

namespace scream {
namespace diag_dsl {

namespace {

std::string param (const DiagSpec& s, const std::string& key) {
  for (const auto& [k,v] : s.params) {
    if (k==key) return v;
  }
  return "";
}

// Resolve a legacy name the way create_diagnostic will: rewrite it to DSL,
// parse that, and translate. Asserts the rewrite happened.
DiagSpec via_legacy (const std::string& name) {
  auto dsl = legacy_to_dsl(name);
  REQUIRE(dsl.has_value());
  edp::parser::Parser p{edp::Lexer{*dsl}};
  return spec_from_ast(*p.parse(),"physics");
}

DiagSpec named (const std::string& name) {
  auto s = named_diagnostic(name,"physics");
  REQUIRE(s.has_value());
  return *s;
}

} // anonymous namespace

TEST_CASE("names: canonical named diagnostics resolve to class + kind") {
  REQUIRE(named("IceWaterPath").diag_name=="WaterPath");
  REQUIRE(param(named("IceWaterPath"),"water_kind")=="Ice");
  REQUIRE(param(named("VapWaterPath"),"water_kind")=="Vap");
  REQUIRE(param(named("LiqNumberPath"),"number_kind")=="Liq");
  REQUIRE(param(named("MeridionalVapFlux"),"wind_component")=="Meridional");
  REQUIRE(param(named("precip_ice_surf_mass_flux"),"precip_type")=="ice");
  REQUIRE(param(named("AeroComCldTop"),"aero_com_cld_kind")=="Top");

  // A bare PotentialTemperature is the total form.
  REQUIRE(param(named("PotentialTemperature"),"temperature_kind")=="Tot");
  REQUIRE(param(named("LiqPotentialTemperature"),"temperature_kind")=="Liq");

  auto z = named("z_mid");
  REQUIRE(z.diag_name=="VerticalLayer");
  REQUIRE(param(z,"diag_name")=="z");
  REQUIRE(param(z,"vert_location")=="mid");
  REQUIRE(param(named("geopotential_int"),"vert_location")=="int");
  REQUIRE(param(named("dz"),"diag_name")=="dz");

  // Diagnostics with no variant are not in this table; they are looked up in
  // the factory directly.
  REQUIRE_FALSE(named_diagnostic("RelativeHumidity","physics").has_value());
  REQUIRE_FALSE(named_diagnostic("T_mid","physics").has_value());
}

TEST_CASE("names: legacy syntax rewrites to the same diagnostic as before") {
  REQUIRE(via_legacy("BlaH_123_at_lev_10").diag_name=="FieldAtLevel");
  REQUIRE(param(via_legacy("BlaH_123_at_lev_10"),"vertical_location")=="lev_10");
  // model_top is level 0 and model_bot is the last level. The generated
  // location differs in spelling from the old one for model_top, which is
  // invisible because create_diagnostic pins output_name to the requested
  // string for legacy inputs.
  REQUIRE(param(via_legacy("X_at_model_top"),"vertical_location")=="lev_0");
  REQUIRE(param(via_legacy("X_at_model_bot"),"vertical_location")=="model_bot");

  REQUIRE(param(via_legacy("X_at_10hPa"),"pressure_units")=="hPa");
  REQUIRE(param(via_legacy("X_at_10mb"),"pressure_units")=="mb");
  REQUIRE(param(via_legacy("X_at_10Pa"),"pressure_value")=="10");
  REQUIRE(param(via_legacy("X_at_10m_above_sealevel"),"surface_reference")=="sealevel");
  REQUIRE(param(via_legacy("X_at_10m_above_surface"),"surface_reference")=="surface");

  REQUIRE(via_legacy("X_over_dt").diag_name=="FieldOverDt");
  REQUIRE(via_legacy("X_prev").diag_name=="FieldPrev");
  REQUIRE(via_legacy("X_horiz_avg").diag_name=="HorizAvg");
  REQUIRE(param(via_legacy("X_vert_avg"),"contract_method")=="avg");
  REQUIRE(param(via_legacy("X_vert_sum"),"contract_method")=="sum");
  REQUIRE(param(via_legacy("X_vert_sum_dp_weighted"),"weighting_method")=="dp");
  REQUIRE(param(via_legacy("X_vert_avg_dz_weighted"),"weighting_method")=="dz");
  REQUIRE(param(via_legacy("X_zonal_avg_20_bins"),"number_of_zonal_bins")=="20");
  REQUIRE(param(via_legacy("X_pvert_derivative"),"derivative_method")=="p");
  REQUIRE(param(via_legacy("X_histogram_0_1_2"),"bin_configuration")=="0_1_2");

  auto cs = via_legacy("T_where_qc_gt_0");
  REQUIRE(cs.diag_name=="ConditionalSampling");
  REQUIRE(param(cs,"condition_lhs")=="qc");
  REQUIRE(param(cs,"condition_cmp")=="gt");
  REQUIRE(param(cs,"condition_rhs")=="0");
}

TEST_CASE("names: legacy precedence quirks are reproduced exactly") {
  // These three orderings were load-bearing in the old regex chain, and old
  // names have to keep meaning what they meant.

  // 1. _over_dt beats binary ops, so this is not BinaryOp(X, over, dt).
  REQUIRE(via_legacy("X_over_dt").diag_name=="FieldOverDt");

  // 2. The left operand is greedy, so the RIGHTMOST operator word is the
  //    outermost operation: A_minus_B_over_C is (A-B)/C, not A-(B/C).
  auto b = via_legacy("A_minus_B_over_C");
  REQUIRE(b.diag_name=="BinaryOp");
  REQUIRE(param(b,"binary_op")=="over");
  REQUIRE(param(b,"arg1")=="A_minus_B");
  REQUIRE(param(b,"arg2")=="C");

  // 3. Binary ops beat _prev, so this is BinaryOp(X, minus, X_prev).
  auto p = via_legacy("X_minus_X_prev");
  REQUIRE(p.diag_name=="BinaryOp");
  REQUIRE(param(p,"binary_op")=="minus");
  REQUIRE(param(p,"arg2")=="X_prev");

  // The worked example from the parsing-precedence docs: the outer operation
  // is the division, and the intermediate keeps its legacy name so that a
  // separately requested f_minus_f_prev still refers to the same field.
  auto bt = via_legacy("f_minus_f_prev_over_dt");
  REQUIRE(bt.diag_name=="FieldOverDt");
  REQUIRE(param(bt,"field_name")=="f_minus_f_prev");
}

TEST_CASE("names: atm_backtend expands through the same path as .tend()") {
  auto dsl = legacy_to_dsl("X_atm_backtend");
  REQUIRE(dsl.has_value());
  REQUIRE(*dsl=="X.tend()");

  // .tend() is itself shorthand, so translating it asks for another rewrite.
  edp::parser::Parser p{edp::Lexer{*dsl}};
  auto s = spec_from_ast(*p.parse(),"physics");
  REQUIRE_FALSE(s.rewrite_to.empty());

  edp::parser::Parser p2{edp::Lexer{s.rewrite_to}};
  REQUIRE(spec_from_ast(*p2.parse(),"physics").diag_name=="FieldOverDt");
}

TEST_CASE("names: malformed legacy names are not silently accepted") {
  // Every one of these is expected to throw out of create_diagnostic today.
  // The shim must decline them so they fall through to that error rather than
  // being coerced into some nearby diagnostic.
  for (const std::string n : {
        "BlaH_123_at_modeltop",            // misspelled
        "BlaH_123_at_400KPa",              // invalid pressure units
        "BlaH_123_at_1km_above_sealevel",  // invalid height units
        "BlaH_123_at_1m_above_the_surface",// invalid reference
        "BlaH_123_at_10.5m",               // missing _above_X
        "X_vert_med",                      // not a contraction we have
        "X_where_qc_bt_0",                 // not a comparison
        "X_zonal_avg_bins"}) {             // missing the count
    INFO("name: " << n);
    REQUIRE_FALSE(legacy_to_dsl(n).has_value());
  }
}

TEST_CASE("names: every legacy rewrite is itself valid DSL") {
  // A rewrite that does not parse would turn a clear error into a confusing
  // one, so check the whole table round-trips through the parser.
  for (const std::string n : {
        "X_at_lev_3","X_at_model_top","X_at_model_bot","X_at_500hPa",
        "X_at_10m_above_surface","X_atm_backtend","X_over_dt","X_prev",
        "X_horiz_avg","X_vert_avg","X_vert_sum_dp_weighted",
        "X_zonal_avg_20_bins","T_where_qc_gt_0","a_plus_b",
        "X_histogram_0_1_2","X_pvert_derivative"}) {
    auto dsl = legacy_to_dsl(n);
    INFO("name: " << n << " -> " << (dsl ? *dsl : std::string("<none>")));
    REQUIRE(dsl.has_value());
    REQUIRE_NOTHROW(canonical(*dsl));
  }
}

namespace {

// Stand-in for the diagnostic factory: the handful of products that are
// registered under exactly the name a user requests.
bool registered (const std::string& n) {
  return n=="RelativeHumidity" or n=="SeaLevelPressure" or n=="wind_speed" or
         n=="AerosolOpticalDepth550nm" or n=="Exner";
}

DiagSpec resolved (const std::string& request) {
  return resolve(request,"physics",registered);
}

} // anonymous namespace

TEST_CASE("resolve: bare names try factory, then named table, then legacy") {
  // A registered product wins, and is built with nothing but the grid.
  auto r = resolved("RelativeHumidity");
  REQUIRE(r.diag_name=="RelativeHumidity");
  REQUIRE(param(r,"grid_name")=="physics");

  // Not registered, but a canonical named diagnostic.
  REQUIRE(resolved("LiqWaterPath").diag_name=="WaterPath");

  // Neither: falls through to the legacy rewrite.
  REQUIRE(resolved("T_mid_horiz_avg").diag_name=="HorizAvg");

  // None of the above is an error naming the offender.
  REQUIRE_THROWS_AS(resolved("no_such_thing"),DslError);
}

TEST_CASE("resolve: DSL expressions resolve without any rewrite") {
  REQUIRE(resolved("T_mid.weighted('dp').mean(dim='lev')").diag_name=="VertContract");
  REQUIRE(resolved("T_mid.isel(lev=-1)").diag_name=="FieldAtLevel");
  REQUIRE(resolved("qc + qr").diag_name=="BinaryOp");
  REQUIRE(resolved("T_mid / dt").diag_name=="FieldOverDt");
}

TEST_CASE("resolve: rewrites chain until they land on something buildable") {
  // Two hops: the legacy name becomes X.tend(), which is itself shorthand for
  // the subtraction over dt.
  auto s = resolved("BlaH_123_atm_backtend");
  REQUIRE(s.diag_name=="FieldOverDt");
  // resolve() must follow rewrites to the end; a caller never sees a pending one.
  REQUIRE(s.rewrite_to.empty());

  // The same by the DSL spelling.
  REQUIRE(resolved("BlaH_123.tend()").diag_name=="FieldOverDt");

  // A legacy composite whose operand is itself a legacy name: only the outer
  // operation resolves here, the operand is left named for the IO layer.
  auto bt = resolved("f_minus_f_prev_over_dt");
  REQUIRE(bt.diag_name=="FieldOverDt");
  REQUIRE(param(bt,"field_name")=="f_minus_f_prev");
  // ...and asking for that operand resolves in turn.
  REQUIRE(resolved("f_minus_f_prev").diag_name=="BinaryOp");
}

TEST_CASE("resolve: every name the existing create_diag test uses still works") {
  // These must all produce a diagnostic.
  for (const std::string n : {
        "BlaH_123_at_model_top","BlaH_123_at_model_bot","BlaH_123_at_lev_10",
        "BlaH_123_at_10mb","BlaH_123_at_10hPa","BlaH_123_at_10Pa",
        "BlaH_123_at_10m_above_sealevel","BlaH_123_at_10m_above_surface",
        "BlaH_123_atm_backtend","BlaH_123_prev","BlaH_123_over_dt",
        "LiqWaterPath","IceWaterPath","RainWaterPath","RimeWaterPath",
        "VapWaterPath","LiqNumberPath","IceNumberPath","RainNumberPath",
        "MeridionalVapFlux","ZonalVapFlux","PotentialTemperature",
        "LiqPotentialTemperature","precip_liq_surf_mass_flux",
        "precip_ice_surf_mass_flux","precip_total_surf_mass_flux",
        "z_mid","z_int","geopotential_mid","geopotential_int",
        "height_mid","height_int","dz"}) {
    INFO("name: " << n);
    REQUIRE_NOTHROW(resolved(n));
    REQUIRE_FALSE(resolved(n).diag_name.empty());
  }

  // And these must all fail, as they do today.
  for (const std::string n : {
        "BlaH_123_at_modeltop","BlaH_123_at_400KPa",
        "BlaH_123_at_1km_above_sealevel","BlaH_123_at_1m_above_the_surface",
        "BlaH_123_at_10.5m"}) {
    INFO("name: " << n);
    REQUIRE_THROWS(resolved(n));
  }
}

TEST_CASE("resolve: unparseable requests surface as errors, not wrong diags") {
  // The parser reports these; resolve must not swallow them.
  REQUIRE_THROWS(resolved("T_mid + @foo"));
  REQUIRE_THROWS(resolved("T_mid.mean(dim='lev"));
  REQUIRE_THROWS(resolved("a = b = c"));
}

} // namespace diag_dsl
} // namespace scream
