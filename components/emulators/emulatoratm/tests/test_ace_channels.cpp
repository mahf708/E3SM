// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "ace_channels.hpp"

#include <algorithm>

namespace emulator {
namespace atm {
namespace test {

using fields::InputSource;
using Temporal = coupling::BracketedState::Temporal;

TEST_CASE("ACE2-EAMv3 has the checkpoint's 39 inputs and 44 outputs",
          "[ace][channels]") {
  const auto l = ace2_eamv3();
  REQUIRE(l.model_dt == 21600);
  REQUIRE(l.inputs.size() == 39);
  REQUIRE(l.outputs.size() == 44);
  REQUIRE(l.inputs.front() == "LANDFRAC");
  REQUIRE(l.inputs[6] == "TS");
  REQUIRE(l.inputs.back() == "V_7");
  REQUIRE(l.outputs[34] == "LHFLX"); // channel 35, 1-based, in the Fortran
  REQUIRE(l.outputs.back() == "tendency_of_total_water_path_due_to_advection");
}

TEST_CASE("SamudrACE-E3SMv3 has the checkpoint's 43 inputs and 51 outputs",
          "[ace][channels]") {
  const auto l = samudrace_e3smv3();
  REQUIRE(l.inputs.size() == 43);
  REQUIRE(l.outputs.size() == 51);
  REQUIRE(l.inputs[39] == "Qat2m");
  REQUIRE(l.inputs[42] == "Tat2m");
  REQUIRE(l.outputs[37] == "frozen_precipitation_rate");
  REQUIRE(l.outputs[45] == "TAUX");
  REQUIRE(l.outputs.back() == "Tat2m");
  REQUIRE(std::find(l.inputs.begin(), l.inputs.end(), "STW_7") !=
          l.inputs.end());
}

TEST_CASE("Every ACE input has its source", "[ace][channels]") {
  for (const auto &name : ace_layout_names()) {
    const auto l = ace_layout(name);
    INFO(name);
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
  REQUIRE(samudrace_e3smv3().source("Tat2m") == InputSource::Prognostic);
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

  const auto sam = samudrace_e3smv3();
  REQUIRE(sam.temporal("TAUX") == Temporal::IntervalMean);
  REQUIRE(sam.temporal("Tat2m") == Temporal::Snapshot);
  const auto kinds = sam.output_temporals();
  REQUIRE(std::count(kinds.begin(), kinds.end(), Temporal::IntervalMean) ==
          13);
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

  REQUIRE_THROWS_WITH(ace_layout("ACE3"), Catch::Contains("ACE2-EAMv3"));
}

} // namespace test
} // namespace atm
} // namespace emulator
