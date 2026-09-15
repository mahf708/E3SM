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

#include "emulator_test_support.hpp"

#include <filesystem>
#include <fstream>

namespace emulator {
namespace test {

TEST_CASE("A spec that extends another merges maps and replaces lists",
          "[config]") {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("spec_extends_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "base.yaml") << R"(
name: base
network: {name: n, timestep: 60, inputs: [a, b], coupled_inputs: [a, b]}
operators: [{operator: x}, {operator: y}]
coupler: {imports: [{name: I, units: "1"}], exports: [{name: E, units: K}]}
)";
  std::ofstream(dir / "variant.yaml") << R"(
extends: base.yaml
name: variant
network: {coupled_inputs: [b], boundary_inputs: [a]}
operators: [{operator: z}]
coupler: {imports: []}
)";
  const auto s = config::Section::load_spec((dir / "variant.yaml").string());
  REQUIRE(s.string("name") == "variant");
  REQUIRE_FALSE(s.has("extends"));
  const auto net = s.section("network");
  REQUIRE(net.names("inputs") == std::vector<std::string>{"a", "b"}); // kept
  REQUIRE(net.names("coupled_inputs") == std::vector<std::string>{"b"});
  REQUIRE(net.names("boundary_inputs") == std::vector<std::string>{"a"});
  REQUIRE(net.integer("timestep") == 60);
  REQUIRE(s.list("operators").size() == 1); // a list replaces
  REQUIRE(s.section("coupler").list("imports").empty());
  REQUIRE(s.section("coupler").list("exports").size() == 1);
  std::error_code ignored;
  std::filesystem::remove(dir / "base.yaml", ignored);
  std::filesystem::remove(dir / "variant.yaml", ignored);
  std::filesystem::remove(dir, ignored);
}

} // namespace test
} // namespace emulator
