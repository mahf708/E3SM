// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "channel_layout_yaml.hpp"
#include "emulator_test_support.hpp"

#include <algorithm>

namespace emulator {
namespace atm {
namespace test {

namespace {
fields::ChannelLayout ace2_eamv3() {
  return fields::read_channel_layout(
      config::Section::load_spec(emulator::test::spec_path("ace2-eamv3.yaml"))
          .section("network"));
}
} // namespace

using fields::InputSource;
using Temporal = coupling::BracketedState::Temporal;

TEST_CASE("The ACE2-EAMv3 spec has the checkpoint's 39 inputs and 44 outputs",
          "[ace][channels]") {
  const auto l = ace2_eamv3();
  REQUIRE(l.model_dt == 21600);
  REQUIRE(l.inputs.size() == 39);
  REQUIRE(l.outputs.size() == 44);
  REQUIRE(l.inputs.front() == "LANDFRAC");
  REQUIRE(l.inputs[6] == "TS");
  REQUIRE(l.inputs.back() == "V_7");
  REQUIRE(l.outputs[34] == "LHFLX"); // channel 35, 1-based
  REQUIRE(l.outputs.back() == "tendency_of_total_water_path_due_to_advection");
}

TEST_CASE("Every ACE input has its source", "[ace][channels]") {
  {
    const auto l = ace2_eamv3();
    // Predicted by the network, but owned by the component: TS is blended
    // with the coupler's surface temperature.
    REQUIRE(l.source("TS") == InputSource::Coupled);
    REQUIRE(l.source("ICEFRAC") == InputSource::Coupled);
    REQUIRE(l.source("PHIS") == InputSource::Boundary);
    REQUIRE(l.source("SOLIN") == InputSource::Forcing);
    REQUIRE(l.source("PS") == InputSource::Prognostic);
    REQUIRE(l.source("T_7") == InputSource::Prognostic);
    REQUIRE(l.inputs_from(InputSource::Coupled).size() == 4);
  }
}

TEST_CASE("ACE fluxes are interval means and the state is snapshots",
          "[ace][channels]") {
  const auto ace2 = ace2_eamv3();
  REQUIRE(ace2.temporal("PS") == Temporal::Snapshot);
  REQUIRE(ace2.temporal("U_7") == Temporal::Snapshot);
  REQUIRE(ace2.temporal("FSDS") == Temporal::IntervalMean);
  REQUIRE(ace2.temporal("surface_upward_shortwave_flux") ==
          Temporal::IntervalMean);
  REQUIRE(ace2.temporal("surface_precipitation_rate") ==
          Temporal::IntervalMean);
}

TEST_CASE("A layout with an input nothing sets is refused", "[channels]") {
  auto l = ace2_eamv3();
  l.boundary_inputs.clear(); // PHIS: not an output, now not a boundary either
  REQUIRE_THROWS_WITH(l.validate(), Catch::Contains("'PHIS' has no source"));

  auto twice = ace2_eamv3();
  twice.forcing_inputs.push_back("PHIS");
  REQUIRE_THROWS_WITH(twice.validate(),
                      Catch::Contains("more than one source"));

  auto typo = ace2_eamv3();
  typo.interval_mean_outputs.push_back("FSNS");
  REQUIRE_THROWS_WITH(typo.validate(), Catch::Contains("'FSNS'"));

  REQUIRE_THROWS_WITH(
      config::Section::load_spec(emulator::test::spec_path("ACE3.yaml")),
      Catch::Contains("ACE3.yaml"));
}

} // namespace test
} // namespace atm
} // namespace emulator

