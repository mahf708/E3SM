#include "catch2/catch.hpp"

#include "share/physics/physics_constants.hpp"

#include <ekat_units.hpp>

namespace {

TEST_CASE("physics_constants", "[physics]")
{
  using scream::Real;
  using PC = scream::physics::Constants<Real>;
  using namespace ekat::units;

  SECTION("backward_compatibility") {
    // Test that constants can be used as scalars (backward compatibility)
    Real rho = PC::RHO_H2O;
    REQUIRE(rho == 1000.0);
    
    Real cpair = PC::Cpair;
    REQUIRE(cpair == 1004.64);
    
    Real rair = PC::Rair;
    REQUIRE(rair == 287.042);
    
    Real g = PC::gravit;
    REQUIRE(g == 9.80616);
  }

  SECTION("value_member") {
    // Test that constants have .value member
    REQUIRE(PC::RHO_H2O.value == 1000.0);
    REQUIRE(PC::Cpair.value == 1004.64);
    REQUIRE(PC::Rair.value == 287.042);
    REQUIRE(PC::gravit.value == 9.80616);
    REQUIRE(PC::Tmelt.value == 273.15);
  }

  SECTION("unit_member") {
    // Test that constants have .unit member
    REQUIRE(PC::RHO_H2O.unit == kg / pow(m, 3));
    REQUIRE(PC::Cpair.unit == J / (kg * K));
    REQUIRE(PC::Rair.unit == J / (kg * K));
    REQUIRE(PC::gravit.unit == m / pow(s, 2));
    REQUIRE(PC::Tmelt.unit == K);
    REQUIRE(PC::Pi.unit == Units::nondimensional());
  }

  SECTION("arithmetic_with_constants") {
    // Test that constants can be used in arithmetic expressions
    Real density = PC::RHO_H2O;
    Real volume = 2.0;
    Real mass = density * volume;
    REQUIRE(mass == 2000.0);
    
    // Test with implicit conversion
    Real temp_diff = PC::Tmelt - 273.0;
    REQUIRE(temp_diff == 0.15);
  }

  SECTION("derived_constants") {
    // Test derived constants
    REQUIRE(PC::T_zerodegc.value == PC::Tmelt.value);
    REQUIRE(PC::RHOW.value == PC::RHO_H2O.value);
    REQUIRE(PC::RD.value == PC::Rair.value);
  }

  SECTION("constexpr_usage") {
    // Test that constants can still be used in ways similar to constexpr
    constexpr Real pi_val = 3.14159265358979323;
    Real calculated_pi = PC::Pi;
    REQUIRE(calculated_pi == pi_val);
  }
}

} // anonymous namespace
