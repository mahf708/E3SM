// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>

#include "create_inference_backend.hpp"

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
  REQUIRE_FALSE(backend->is_initialized());
}

TEST_CASE("StubBackend lifecycle", "[stub_backend]") {
  InferenceConfig config;
  config.input_channels = 4;
  config.output_channels = 2;

  auto backend = create_backend(BackendType::STUB, config);

  // Run inference through the flat-array path (no-op: outputs unchanged)
  double inputs[4] = {1, 2, 3, 4};
  double outputs[2] = {99, 99};

  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(outputs[0] == 99.0);
  REQUIRE(outputs[1] == 99.0);

  // infer() initializes lazily and counts calls
  REQUIRE(backend->is_initialized());
  REQUIRE(backend->infer_count() == 1);

  // Finalize is idempotent
  backend->finalize();
  backend->finalize();
  REQUIRE_FALSE(backend->is_initialized());
}

TEST_CASE("create_backend fallback for unknown type", "[stub_backend]") {
  InferenceConfig config;
  auto backend = create_backend(static_cast<BackendType>(999), config);

  REQUIRE(backend != nullptr);
  REQUIRE(backend->name() == "Stub"); // Falls back to stub
}

TEST_CASE("create_backend honors the configured key", "[stub_backend]") {
  InferenceConfig config;
  config.backend = "stub";
  auto by_config = create_backend(config);
  REQUIRE(by_config->name() == "Stub");

  auto by_key = create_backend("STUB", config); // keys are case-insensitive
  REQUIRE(by_key->name() == "Stub");

  auto initialized = create_and_init_backend(config);
  REQUIRE(initialized->is_initialized());
}

TEST_CASE("stub backend derives specs from channel counts", "[stub_backend]") {
  InferenceConfig config;
  config.input_channels = 3;
  config.output_channels = 2;

  auto backend = create_backend(config);

  const auto in_specs = backend->input_specs();
  REQUIRE(in_specs.size() == 1);
  REQUIRE(in_specs[0].name == "input");
  REQUIRE(in_specs[0].dims == std::vector<std::int64_t>{-1, 3});

  // Allocation helpers apply the requested batch size.
  TensorMap inputs = backend->make_inputs(8);
  REQUIRE(inputs.size() == 1);
  REQUIRE(inputs[0].dims() == std::vector<std::int64_t>{8, 3});
  TensorMap outputs = backend->make_outputs(8);
  REQUIRE(outputs[0].dims() == std::vector<std::int64_t>{8, 2});
}

TEST_CASE("flat path needs to know the column width", "[stub_backend]") {
  InferenceConfig config; // no channels, no specs
  auto backend = create_backend(config);

  double in[2] = {1, 2};
  double out[2] = {0, 0};
  REQUIRE_THROWS_AS(backend->infer(in, out, 1), InferenceError);

  config.input_channels = 2;
  config.output_channels = 2;
  auto ok = create_backend(config);
  REQUIRE_THROWS_AS(ok->infer(in, out, 0), InferenceError);
  REQUIRE_THROWS_AS(ok->infer(nullptr, out, 1), InferenceError);
}

TEST_CASE("stub modes exercise the tensor path", "[stub_backend]") {
  SECTION("zero") {
    InferenceConfig config;
    config.set("mode", std::string("zero"));
    auto backend = create_backend(config);

    TensorMap inputs;
    inputs.emplace("x", {2, 2});
    TensorMap outputs;
    Tensor &out = outputs.emplace("y", {2, 2});
    out.flat<double>(0) = 5.0;

    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(outputs.at("y").flat<double>(0) == 0.0);
  }

  SECTION("constant") {
    InferenceConfig config;
    config.set("mode", std::string("constant")).set("value", 3.5);
    auto backend = create_backend(config);

    TensorMap inputs;
    inputs.emplace("x", {2});
    TensorMap outputs;
    outputs.emplace("y", {2}, DType::FLOAT32);

    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(outputs.at("y").flat<float>(0) == 3.5f);
    REQUIRE(outputs.at("y").flat<float>(1) == 3.5f);
  }

  SECTION("copy converts precision") {
    InferenceConfig config;
    config.set("mode", std::string("copy"));
    auto backend = create_backend(config);

    TensorMap inputs;
    Tensor &in = inputs.emplace("x", {3}, DType::FLOAT64);
    in.flat<double>(0) = 1.25;
    in.flat<double>(1) = 2.5;
    in.flat<double>(2) = -3.75;

    TensorMap outputs;
    outputs.emplace("y", {3}, DType::FLOAT32);

    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(outputs.at("y").flat<float>(0) == 1.25f);
    REQUIRE(outputs.at("y").flat<float>(2) == -3.75f);
  }

  SECTION("affine writes into caller memory") {
    InferenceConfig config;
    config.set("mode", std::string("affine")).set("scale", 2.0).set("offset",
                                                                    1.0);
    config.outputs.push_back(TensorSpec("y", {-1, 2}, DType::FLOAT64));
    auto backend = create_backend(config);

    // The component's own arrays, wrapped as views: no copies anywhere.
    std::vector<double> field_in{1.0, 2.0, 3.0, 4.0};
    std::vector<double> field_out(4, 0.0);

    TensorMap inputs;
    inputs.wrap("x", static_cast<const double *>(field_in.data()), {2, 2});
    TensorMap outputs;
    outputs.wrap("y", field_out.data(), {2, 2});

    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(field_out[0] == 3.0);
    REQUIRE(field_out[3] == 9.0);
  }

  SECTION("outputs are allocated from specs when the caller passes none") {
    InferenceConfig config;
    config.set("mode", std::string("constant")).set("value", 1.0);
    config.outputs.push_back(TensorSpec("dT", {-1, 4}, DType::FLOAT32));
    auto backend = create_backend(config);

    TensorMap inputs;
    inputs.emplace("T", {6, 4}, DType::FLOAT32);
    TensorMap outputs; // empty on purpose

    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(outputs.size() == 1);
    REQUIRE(outputs.at("dT").dims() == std::vector<std::int64_t>{6, 4});
    REQUIRE(outputs.at("dT").flat<float>(0) == 1.0f);
  }

  SECTION("unknown mode is reported") {
    InferenceConfig config;
    config.set("mode", std::string("magic"));
    REQUIRE_THROWS_AS(create_backend(config), InferenceError);
  }
}

TEST_CASE("declared input specs are validated", "[stub_backend]") {
  InferenceConfig config;
  config.set("mode", std::string("zero"));
  config.inputs.push_back(TensorSpec("T", {-1, 4}, DType::FLOAT32));
  config.inputs.push_back(TensorSpec("ps", {-1}, DType::FLOAT32));
  auto backend = create_backend(config);

  TensorMap outputs;

  SECTION("a missing input is named in the error") {
    TensorMap inputs;
    inputs.emplace("T", {2, 4}, DType::FLOAT32);
    REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                        Catch::Contains("ps"));
  }

  SECTION("a wrong element type is named in the error") {
    TensorMap inputs;
    inputs.emplace("T", {2, 4}, DType::FLOAT64); // should be float32
    inputs.emplace("ps", {2}, DType::FLOAT32);
    REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                        Catch::Contains("float32"));
  }

  SECTION("a wrong extent is rejected") {
    TensorMap inputs;
    inputs.emplace("T", {2, 5}, DType::FLOAT32); // 5 != 4
    inputs.emplace("ps", {2}, DType::FLOAT32);
    REQUIRE_THROWS_AS(backend->infer(inputs, outputs), InferenceError);
  }

  SECTION("extra tensors are tolerated") {
    TensorMap inputs;
    inputs.emplace("T", {2, 4}, DType::FLOAT32);
    inputs.emplace("ps", {2}, DType::FLOAT32);
    inputs.emplace("unused_by_this_model", {2}, DType::FLOAT64);
    REQUIRE(backend->infer(inputs, outputs));
  }
}

} // namespace test
} // namespace inference
} // namespace emulator
