// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>
#include <vector>

#include "create_inference_backend.hpp"

#ifndef EMULATOR_TEST_TORCH_MODEL
#error "EMULATOR_TEST_TORCH_MODEL must point at the generated .pt fixture"
#endif

namespace emulator {
namespace inference {
namespace test {

namespace {

// The fixture module (see fixtures/make_torch_model.py) returns a dict:
//   dT    = 2 * T + 1                     [ncol, 3]
//   total = T.sum(dim=1, keepdim=True)+ps  [ncol, 1]
InferenceConfig torch_config() {
  InferenceConfig config;
  config.backend = "torch";
  config.model_path = EMULATOR_TEST_TORCH_MODEL;
  // TorchScript arguments are positional; declaring them is how the argument
  // order and the model's precision get written down.
  config.inputs.push_back(TensorSpec("T", {-1, 3}, DType::FLOAT32));
  config.inputs.push_back(TensorSpec("ps", {-1, 1}, DType::FLOAT32));
  return config;
}

} // namespace

TEST_CASE("the torch backend is registered", "[torch]") {
  REQUIRE(BackendRegistry::instance().has("torch"));
  REQUIRE(BackendRegistry::instance().has("libtorch"));
}

TEST_CASE("torchscript inference names its dict outputs", "[torch]") {
  auto backend = create_backend(torch_config());
  REQUIRE(backend->name() == "LibTorch");
  backend->initialize();

  TensorMap inputs = backend->make_inputs(4);
  Tensor &T = inputs.at("T");
  for (std::int64_t i = 0; i < T.size(); ++i) {
    T.flat<float>(i) = static_cast<float>(i + 1); // 1..12
  }
  Tensor &ps = inputs.at("ps");
  for (std::int64_t i = 0; i < ps.size(); ++i) {
    ps.flat<float>(i) = 100.0f;
  }

  TensorMap outputs; // empty: names come from the module's dict keys
  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(outputs.size() == 2);
  REQUIRE(outputs.has("dT"));
  REQUIRE(outputs.has("total"));
  REQUIRE(outputs.at("dT").dims() == std::vector<std::int64_t>{4, 3});
  REQUIRE(outputs.at("dT").flat<float>(0) == Approx(3.0f));   // 2*1 + 1
  REQUIRE(outputs.at("dT").flat<float>(11) == Approx(25.0f)); // 2*12 + 1
  REQUIRE(outputs.at("total").flat<float>(0) == Approx(106.0f));
  REQUIRE(outputs.at("total").flat<float>(3) == Approx(133.0f));
}

TEST_CASE("torch results land in the caller's r8 fields", "[torch]") {
  auto backend = create_and_init_backend(torch_config());

  // E3SM-side arrays are double; the model is single precision.  The declared
  // specs are what tells the backend to convert.
  std::vector<double> T_field{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> ps_field{10.0, 20.0};
  std::vector<double> dT_field(6, -999.0);
  std::vector<double> total_field(2, -999.0);

  TensorMap inputs;
  inputs.wrap("T", static_cast<const double *>(T_field.data()), {2, 3});
  inputs.wrap("ps", static_cast<const double *>(ps_field.data()), {2, 1});
  TensorMap outputs;
  outputs.wrap("dT", dT_field.data(), {2, 3});
  outputs.wrap("total", total_field.data(), {2, 1});

  REQUIRE(backend->infer(inputs, outputs));
  for (std::size_t i = 0; i < T_field.size(); ++i) {
    REQUIRE(dT_field[i] == Approx(2.0 * T_field[i] + 1.0));
  }
  REQUIRE(total_field[0] == Approx(16.0));
  REQUIRE(total_field[1] == Approx(35.0));

  // A second step with a different batch size reuses the module and grows the
  // scratch buffers.
  std::vector<double> T2(12, 1.0);
  std::vector<double> ps2(4, 0.5);
  TensorMap inputs2;
  inputs2.wrap("T", static_cast<const double *>(T2.data()), {4, 3});
  inputs2.wrap("ps", static_cast<const double *>(ps2.data()), {4, 1});
  TensorMap outputs2;
  REQUIRE(backend->infer(inputs2, outputs2));
  REQUIRE(outputs2.at("total").flat<float>(0) == Approx(3.5f));
  REQUIRE(backend->infer_count() == 2);
}

TEST_CASE("torch backend reports bad configuration clearly", "[torch]") {
  SECTION("missing model") {
    InferenceConfig config;
    config.backend = "torch";
    auto backend = create_backend(config);
    REQUIRE_THROWS_WITH(backend->initialize(), Catch::Contains("model_path"));
  }

  SECTION("unreadable model") {
    auto config = torch_config();
    config.model_path = "/nonexistent/model.pt";
    auto backend = create_backend(config);
    REQUIRE_THROWS_AS(backend->initialize(), InferenceError);
  }

  SECTION("unknown device") {
    auto config = torch_config();
    config.set("device", std::string("tpu"));
    auto backend = create_backend(config);
    REQUIRE_THROWS_WITH(backend->initialize(), Catch::Contains("device"));
  }

  SECTION("unknown method") {
    auto config = torch_config();
    config.set("method", std::string("no_such_method"));
    auto backend = create_and_init_backend(config);
    TensorMap inputs = backend->make_inputs(1);
    TensorMap outputs;
    REQUIRE_THROWS_AS(backend->infer(inputs, outputs), InferenceError);
  }

  SECTION("missing argument") {
    auto config = torch_config();
    auto backend = create_and_init_backend(config);
    TensorMap inputs;
    inputs.emplace("T", {2, 3}, DType::FLOAT32);
    inputs.emplace("nope", {2, 1}, DType::FLOAT32);
    inputs.emplace("also_nope", {2, 1}, DType::FLOAT32);
    TensorMap outputs;
    REQUIRE_THROWS_WITH(backend->infer(inputs, outputs), Catch::Contains("ps"));
  }

  SECTION("shape the model cannot use") {
    auto config = torch_config();
    auto backend = create_and_init_backend(config);
    TensorMap inputs;
    inputs.emplace("T", {2, 4}, DType::FLOAT32); // declared as 3 levels
    inputs.emplace("ps", {2, 1}, DType::FLOAT32);
    TensorMap outputs;
    REQUIRE_THROWS_AS(backend->infer(inputs, outputs), InferenceError);
  }
}

TEST_CASE("torch modules can be re-loaded", "[torch]") {
  auto backend = create_and_init_backend(torch_config());
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
