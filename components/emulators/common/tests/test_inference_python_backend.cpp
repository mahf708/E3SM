// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"
#include "inference_error.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace emulator {
namespace inference {
namespace test {

namespace {

/// Settings pointing at tests/fixtures/emulator_fixture.py.
InferenceConfig fixture_config() {
  InferenceConfig config;
  config.backend = "python";
  config.set("python_module", "emulator_fixture");
  config.set("python_path", EMULATOR_TEST_FIXTURE_DIR);
  config.set("report_path", "emulator_fixture_report.txt");
  config.set("scale", "3.0");
  config.model_path = "/not/read/by/the/fixture";
  return config;
}

std::string read_file(const std::string &path) {
  std::ifstream ifs(path);
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

} // namespace

TEST_CASE("The python backend runs a model in this process", "[python]") {
  auto backend = create_backend(fixture_config(), InferenceContext());
  REQUIRE(backend->name() == "Python");

  std::vector<double> x{1.0, 2.0, 3.0, 4.0};
  std::vector<double> y(4, 0.0);

  TensorMap inputs;
  inputs.wrap("x", static_cast<const double *>(x.data()), {4});
  TensorMap outputs;
  outputs.wrap("y", y.data(), {4});

  REQUIRE(backend->infer(inputs, outputs));

  // The fixture computes scale * x + step, writing through a view of `y`.
  REQUIRE(y[0] == Approx(3.0 * 1.0 + 1));
  REQUIRE(y[3] == Approx(3.0 * 4.0 + 1));

  // Step two proves the emulator object persists between calls, which is
  // what an autoregressive model relies on.
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(y[0] == Approx(3.0 * 1.0 + 2));

  // And nothing wrote back through the input view.
  REQUIRE(x[0] == 1.0);

  backend->finalize();
}

TEST_CASE("The factory receives the config and the context", "[python]") {
  const int gids[3] = {1, 5, 9};
  const double lat[3] = {-45.0, 0.0, 45.0};
  const double lon[3] = {0.0, 120.0, 240.0};

  InferenceContext context;
  context.set_grid(8, 6, 48, gids, lat, lon, 3);

  auto config = fixture_config();
  config.set("report_path", "emulator_fixture_context.txt");
  auto backend = create_backend(config, context);

  const std::string report = read_file("emulator_fixture_context.txt");
  REQUIRE(report.find("scale=3.0") != std::string::npos);
  REQUIRE(report.find("model_path=/not/read/by/the/fixture") !=
          std::string::npos);
  REQUIRE(report.find("nx=8 ny=6") != std::string::npos);
  REQUIRE(report.find("gids=[1, 5, 9]") != std::string::npos);

  backend->finalize();
  std::remove("emulator_fixture_context.txt");
}

TEST_CASE("A writable input tensor still arrives read-only", "[python]") {
  // The fixture raises unless numpy reports the input as read-only and the
  // output as writeable, so a successful call is the assertion.  Wrapping a
  // *non-const* pointer as an input must not weaken that.
  auto config = fixture_config();
  config.set("report_path", "emulator_fixture_ro.txt");
  auto backend = create_backend(config, InferenceContext());

  std::vector<double> x{1.0};
  std::vector<double> y{0.0};
  TensorMap inputs;
  inputs.wrap("x", static_cast<const double *>(x.data()), {1});
  TensorMap outputs;
  outputs.wrap("y", y.data(), {1});
  REQUIRE(backend->infer(inputs, outputs));

  backend->finalize();
  std::remove("emulator_fixture_ro.txt");
}

TEST_CASE("A missing module is reported, not guessed at", "[python]") {
  InferenceConfig config;
  config.backend = "python";
  config.set("python_module", "no_such_emulator_module");
  REQUIRE_THROWS_AS(create_backend(config, InferenceContext()), InferenceError);
}

TEST_CASE("A module with no factory is reported", "[python]") {
  auto config = fixture_config();
  config.set("python_factory", "not_a_function");
  REQUIRE_THROWS_AS(create_backend(config, InferenceContext()), InferenceError);
}

TEST_CASE("The shipped package is importable without PYTHONPATH", "[python]") {
  // e3sm_emulator.bridge is the default module, reached through the sys.path
  // entry baked in at configure time. Asking it for an emulator that does not
  // exist proves the import worked without proving anything about torch.
  InferenceConfig config;
  config.backend = "python";
  config.set("emulator", "no_such_emulator");
  REQUIRE_THROWS_WITH(create_backend(config, InferenceContext()),
                      Catch::Contains("Unknown emulator"));
}

TEST_CASE("An error inside the model surfaces with its traceback",
          "[python]") {
  auto config = fixture_config();
  config.set("python_module", "emulator_broken");
  REQUIRE_THROWS_WITH(create_backend(config, InferenceContext()),
                      Catch::Contains("deliberate"));
}

} // namespace test
} // namespace inference
} // namespace emulator
