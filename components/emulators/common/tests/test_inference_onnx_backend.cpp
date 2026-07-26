// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>
#include <vector>

#include "create_inference_backend.hpp"

#ifndef EMULATOR_TEST_ONNX_MODEL
#error "EMULATOR_TEST_ONNX_MODEL must point at the generated .onnx fixture"
#endif

namespace emulator {
namespace inference {
namespace test {

namespace {

// The fixture model (see fixtures/make_onnx_model.py):
//   dT    = 2 * T + 1            [ncol, 3] float32
//   total = sum(T, axis=1) + ps  [ncol, 1] float32
InferenceConfig onnx_config() {
  InferenceConfig config;
  config.backend = "onnx";
  config.model_path = EMULATOR_TEST_ONNX_MODEL;
  return config;
}

} // namespace

TEST_CASE("the onnx backend is registered", "[onnx]") {
  REQUIRE(BackendRegistry::instance().has("onnx"));
  REQUIRE(BackendRegistry::instance().has("onnxruntime"));
}

TEST_CASE("the model's own signature is reported", "[onnx]") {
  auto backend = create_backend(onnx_config());
  REQUIRE(backend->name() == "ONNXRuntime");
  backend->initialize();

  const auto in_specs = backend->input_specs();
  REQUIRE(in_specs.size() == 2);
  REQUIRE(in_specs[0].name == "T");
  REQUIRE(in_specs[0].dtype == DType::FLOAT32);
  REQUIRE(in_specs[0].dims.size() == 2);
  REQUIRE(in_specs[0].dims[0] == -1); // dynamic ncol
  REQUIRE(in_specs[0].dims[1] == 3);
  REQUIRE(in_specs[1].name == "ps");

  const auto out_specs = backend->output_specs();
  REQUIRE(out_specs.size() == 2);
  REQUIRE(out_specs[0].name == "dT");
  REQUIRE(out_specs[1].name == "total");

  // make_inputs()/make_outputs() size the dynamic dimension for us.
  TensorMap inputs = backend->make_inputs(5);
  REQUIRE(inputs.at("T").dims() == std::vector<std::int64_t>{5, 3});
  REQUIRE(inputs.at("T").dtype() == DType::FLOAT32);
}

TEST_CASE("onnx inference on tensors the caller owns", "[onnx]") {
  auto backend = create_and_init_backend(onnx_config());

  const std::int64_t ncol = 4;
  TensorMap inputs = backend->make_inputs(ncol);
  Tensor &T = inputs.at("T");
  for (std::int64_t i = 0; i < T.size(); ++i) {
    T.flat<float>(i) = static_cast<float>(i + 1); // 1..12
  }
  Tensor &ps = inputs.at("ps");
  for (std::int64_t i = 0; i < ps.size(); ++i) {
    ps.flat<float>(i) = 100.0f;
  }

  TensorMap outputs; // empty: the backend allocates from the model's shapes
  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(outputs.size() == 2);
  REQUIRE(outputs.at("dT").dims() == std::vector<std::int64_t>{ncol, 3});
  REQUIRE(outputs.at("dT").flat<float>(0) == Approx(3.0f));  // 2*1 + 1
  REQUIRE(outputs.at("dT").flat<float>(11) == Approx(25.0f)); // 2*12 + 1

  REQUIRE(outputs.at("total").dims() == std::vector<std::int64_t>{ncol, 1});
  REQUIRE(outputs.at("total").flat<float>(0) == Approx(106.0f)); // 1+2+3+100
  REQUIRE(outputs.at("total").flat<float>(3) ==
          Approx(133.0f)); // 10+11+12+100
}

TEST_CASE("onnx results land in the caller's fields", "[onnx]") {
  auto backend = create_and_init_backend(onnx_config());

  // A component's r8 arrays: the backend converts in both directions, using
  // a scratch buffer that is allocated once.
  std::vector<double> T_field{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> ps_field{10.0, 20.0};
  std::vector<double> dT_field(6, -999.0);

  TensorMap inputs;
  inputs.wrap("T", static_cast<const double *>(T_field.data()), {2, 3});
  inputs.wrap("ps", static_cast<const double *>(ps_field.data()), {2, 1});
  TensorMap outputs;
  outputs.wrap("dT", dT_field.data(), {2, 3});

  REQUIRE(backend->infer(inputs, outputs));
  for (std::size_t i = 0; i < T_field.size(); ++i) {
    REQUIRE(dT_field[i] == Approx(2.0 * T_field[i] + 1.0));
  }

  // A second step with a different batch size reuses the same session and
  // grows the scratch buffer.
  std::vector<double> T2(12, 1.0);
  std::vector<double> ps2(4, 0.0);
  std::vector<double> dT2(12, 0.0);
  TensorMap inputs2;
  inputs2.wrap("T", static_cast<const double *>(T2.data()), {4, 3});
  inputs2.wrap("ps", static_cast<const double *>(ps2.data()), {4, 1});
  TensorMap outputs2;
  outputs2.wrap("dT", dT2.data(), {4, 3});
  REQUIRE(backend->infer(inputs2, outputs2));
  REQUIRE(dT2[0] == Approx(3.0));
  REQUIRE(backend->infer_count() == 2);
}

TEST_CASE("onnx backend reports bad configuration clearly", "[onnx]") {
  SECTION("missing model") {
    InferenceConfig config;
    config.backend = "onnx";
    auto backend = create_backend(config);
    REQUIRE_THROWS_WITH(backend->initialize(), Catch::Contains("model_path"));
  }

  SECTION("unreadable model") {
    auto config = onnx_config();
    config.model_path = "/nonexistent/model.onnx";
    auto backend = create_backend(config);
    REQUIRE_THROWS_AS(backend->initialize(), InferenceError);
  }

  SECTION("configured input that the model does not have") {
    auto config = onnx_config();
    config.inputs.push_back(TensorSpec("nope", {-1, 3}, DType::FLOAT32));
    auto backend = create_backend(config);
    REQUIRE_THROWS_WITH(backend->initialize(), Catch::Contains("nope"));
  }

  SECTION("configured precision that disagrees with the model") {
    auto config = onnx_config();
    config.inputs.push_back(TensorSpec("T", {-1, 3}, DType::FLOAT64));
    auto backend = create_backend(config);
    REQUIRE_THROWS_WITH(backend->initialize(), Catch::Contains("float32"));
  }

  SECTION("unknown device") {
    auto config = onnx_config();
    config.set("device", std::string("tpu"));
    auto backend = create_backend(config);
    REQUIRE_THROWS_WITH(backend->initialize(), Catch::Contains("device"));
  }
}

TEST_CASE("onnx backend reports bad tensors clearly", "[onnx]") {
  auto backend = create_and_init_backend(onnx_config());

  SECTION("missing input") {
    TensorMap inputs;
    inputs.emplace("T", {2, 3}, DType::FLOAT32);
    inputs.emplace("something_else", {2, 1}, DType::FLOAT32);
    TensorMap outputs;
    REQUIRE_THROWS_WITH(backend->infer(inputs, outputs), Catch::Contains("ps"));
  }

  SECTION("wrong extent") {
    TensorMap inputs;
    inputs.emplace("T", {2, 5}, DType::FLOAT32); // model wants 3 levels
    inputs.emplace("ps", {2, 1}, DType::FLOAT32);
    TensorMap outputs;
    REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                        Catch::Contains("T[-1,3]"));
  }

  SECTION("wrong rank") {
    TensorMap inputs;
    inputs.emplace("T", {6}, DType::FLOAT32);
    inputs.emplace("ps", {2, 1}, DType::FLOAT32);
    TensorMap outputs;
    REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                        Catch::Contains("rank"));
  }

  SECTION("destination too small") {
    TensorMap inputs = backend->make_inputs(2);
    TensorMap outputs;
    outputs.emplace("dT", {1, 3}, DType::FLOAT32);
    outputs.emplace("total", {1, 1}, DType::FLOAT32);
    REQUIRE_THROWS_AS(backend->infer(inputs, outputs), InferenceError);
  }
}

TEST_CASE("onnx sessions can be re-created", "[onnx]") {
  // Finalize releases the session; a later initialize builds a new one.
  auto backend = create_and_init_backend(onnx_config());
  backend->finalize();
  REQUIRE_FALSE(backend->is_initialized());

  backend->initialize();
  TensorMap inputs = backend->make_inputs(1);
  inputs.at("T").flat<float>(0) = 1.0f;
  TensorMap outputs;
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(outputs.at("dT").flat<float>(0) == Approx(3.0f));
}

} // namespace test
} // namespace inference
} // namespace emulator
