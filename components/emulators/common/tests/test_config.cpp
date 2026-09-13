#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "channel_layout_yaml.hpp"
#include "yaml_config.hpp"

namespace emulator {
namespace test {

using config::Section;

TEST_CASE("Ranges in names expand, and other names pass through", "[config]") {
  REQUIRE(config::expand_range("T_{0..2}") ==
          std::vector<std::string>{"T_0", "T_1", "T_2"});
  REQUIRE(config::expand_range("a{3..1}b") ==
          std::vector<std::string>{"a3b", "a2b", "a1b"});
  REQUIRE(config::expand_range("TAUX:next") ==
          std::vector<std::string>{"TAUX:next"});
}

TEST_CASE("A missing, mistyped or unknown key names its path", "[config]") {
  const auto s = Section::load_string(
      "model:\n  path: /x.pt\n  seed: two\n  devcie: cuda\n", "atm_in");
  const auto model = s.section("model");
  REQUIRE(model.string("path") == "/x.pt");
  REQUIRE_THROWS_WITH(model.string("dtype"),
                      Catch::Contains("atm_in: model.dtype") &&
                          Catch::Contains("missing"));
  REQUIRE_THROWS_WITH(model.integer("seed"),
                      Catch::Contains("atm_in: model.seed") &&
                          Catch::Contains("'two'"));
  REQUIRE_THROWS_WITH(model.only({"path", "seed", "device"}),
                      Catch::Contains("devcie") && Catch::Contains("device"));
  REQUIRE(model.string_or("device", "cpu") == "cpu");
  REQUIRE_FALSE(s.has("grid"));
  REQUIRE(s.optional_section("grid").keys().empty());
}

TEST_CASE("A layout reads from YAML and is validated there", "[config]") {
  const auto s = Section::load_string(R"(
network:
  name: toy
  timestep: 3600
  inputs: [F, "x_{0..1}"]
  outputs: ["x_{0..1}", flux]
  interval_mean_outputs: [flux]
  forcing_inputs: [F]
)", "toy.yaml");
  const auto l = fields::read_channel_layout(s.section("network"));
  REQUIRE(l.inputs == std::vector<std::string>{"F", "x_0", "x_1"});
  REQUIRE(l.model_dt == 3600);
  REQUIRE(l.source("x_1") == fields::InputSource::Prognostic);

  const auto bad = Section::load_string(R"(
network:
  name: toy
  timestep: 3600
  inputs: [F, x]
  outputs: [x]
)", "bad.yaml");
  // F has no source: not an output, and no role.
  REQUIRE_THROWS_WITH(fields::read_channel_layout(bad.section("network")),
                      Catch::Contains("bad.yaml: network") &&
                          Catch::Contains("F"));
}

} // namespace test
} // namespace emulator
