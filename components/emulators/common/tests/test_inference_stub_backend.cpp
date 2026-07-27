// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"
#include "inference_error.hpp"

#include <vector>

namespace emulator {
namespace inference {
namespace test {

TEST_CASE("StubBackend factory creation", "[stub_backend]") {
  InferenceConfig config;
  config.input_channels = 4;
  config.output_channels = 2;

  auto backend = create_backend(BackendType::STUB, config);

  REQUIRE(backend != nullptr);
  REQUIRE(backend->name() == "Stub");
  REQUIRE(backend->is_initialized()); // the factory returns a ready backend
}

TEST_CASE("StubBackend lifecycle", "[stub_backend]") {
  InferenceConfig config;
  config.input_channels = 4;
  config.output_channels = 2;

  auto backend = create_backend(BackendType::STUB, config);

  // Run inference (no-op: outputs unchanged)
  double inputs[4] = {1, 2, 3, 4};
  double outputs[2] = {99, 99};

  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(outputs[0] == 99.0);
  REQUIRE(outputs[1] == 99.0);

  // Finalize, twice: teardown has to be idempotent because a component may
  // finalize explicitly and then be destroyed.
  backend->finalize();
  backend->finalize();
  REQUIRE_FALSE(backend->is_initialized());
}

TEST_CASE("The tensor path carries names and shapes", "[stub_backend]") {
  InferenceConfig config;
  config.backend = "stub";
  auto backend = create_backend(config, InferenceContext());

  std::vector<double> T(12, 300.0);
  std::vector<double> dT(12, -1.0);

  TensorMap inputs;
  inputs.wrap("T", static_cast<const double *>(T.data()), {4, 3});
  TensorMap outputs;
  outputs.wrap("dT", dT.data(), {4, 3});

  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(dT[0] == -1.0); // unchanged, so a forgotten field is visible
}

TEST_CASE("The flat path needs its channel counts", "[stub_backend]") {
  InferenceConfig config;
  auto backend = create_backend(config, InferenceContext());

  double in[4] = {0, 0, 0, 0};
  double out[4] = {0, 0, 0, 0};
  REQUIRE_THROWS_AS(backend->infer(in, out, 1), InferenceError);
}

TEST_CASE("An unknown backend name is refused", "[stub_backend]") {
  InferenceConfig config;
  config.backend = "tensorflow";
  REQUIRE_THROWS_AS(create_backend(config, InferenceContext()), InferenceError);
}

TEST_CASE("A default context describes a serial run", "[stub_backend]") {
  InferenceContext context;
  REQUIRE(context.rank == 0);
  REQUIRE(context.size == 1);
  REQUIRE(context.is_root());
  REQUIRE(context.num_local_cols() == 0);
}

TEST_CASE("The context carries the coupler's decomposition", "[stub_backend]") {
  const int gids[3] = {1, 5, 9};
  const double lat[3] = {-45.0, 0.0, 45.0};
  const double lon[3] = {0.0, 120.0, 240.0};

  InferenceContext context;
  context.set_grid(8, 6, 48, gids, lat, lon, 3);

  REQUIRE(context.num_local_cols() == 3);
  REQUIRE(context.col_gids[2] == 9);
  REQUIRE(context.lat[0] == -45.0);
  REQUIRE(context.nx == 8);
  REQUIRE(context.to_string().find("3 of 48 columns") != std::string::npos);
}

TEST_CASE("make_context is usable without a launcher", "[stub_backend]") {
  // Built with MPI but never launched under it (a unit test, a tool), the
  // context must still describe a valid serial run rather than abort.
  const auto context = make_context(0);
  REQUIRE(context.size >= 1);
  REQUIRE(context.rank >= 0);
}

} // namespace test
} // namespace inference
} // namespace emulator
