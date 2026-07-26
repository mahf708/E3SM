// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <algorithm>

#include "create_inference_backend.hpp"

namespace emulator {
namespace inference {
namespace test {

namespace {

/// Minimal out-of-tree backend: what a new runtime has to implement.
class CountingBackend : public InferenceBackend {
public:
  explicit CountingBackend(const InferenceConfig &config)
      : InferenceBackend(config) {}

  std::string name() const override { return "Counting"; }

  int init_calls = 0;
  int final_calls = 0;

protected:
  void init_impl() override { ++init_calls; }

  bool infer_impl(const TensorMap &inputs, TensorMap &outputs) override {
    // Sum every input element into the first output element.
    double sum = 0.0;
    for (const auto &t : inputs) {
      for (std::int64_t i = 0; i < t.size(); ++i) {
        sum += t.flat<double>(i);
      }
    }
    if (outputs.empty()) {
      outputs.emplace("sum", {1});
    }
    outputs[0].flat<double>(0) = sum;
    return true;
  }

  void final_impl() override { ++final_calls; }
};

} // namespace

TEST_CASE("built-in backends are registered", "[registry]") {
  auto &registry = BackendRegistry::instance();

  REQUIRE(registry.has("stub"));
  REQUIRE(registry.has("STUB")); // keys are case-insensitive
  REQUIRE(registry.has(" stub "));
  REQUIRE_FALSE(registry.has("definitely_not_a_backend"));

  const auto keys = registry.available();
  REQUIRE(std::find(keys.begin(), keys.end(), "stub") != keys.end());
  REQUIRE(registry.available_string().find("stub") != std::string::npos);
}

TEST_CASE("unknown backends produce an actionable message", "[registry]") {
  InferenceConfig config;
  auto &registry = BackendRegistry::instance();

  REQUIRE_THROWS_WITH(registry.create("nope", config),
                      Catch::Contains("Available") && Catch::Contains("stub"));

  // Backends that exist but were not compiled in say which option adds them.
  if (!registry.has("torch")) {
    REQUIRE_THROWS_WITH(registry.create("torch", config),
                        Catch::Contains("EMULATOR_ENABLE_TORCH"));
  }
  if (!registry.has("onnx")) {
    REQUIRE_THROWS_WITH(registry.create("onnx", config),
                        Catch::Contains("EMULATOR_ENABLE_ONNXRUNTIME"));
  }
  if (!registry.has("python")) {
    REQUIRE_THROWS_WITH(registry.create("python", config),
                        Catch::Contains("EMULATOR_ENABLE_PYTHON"));
  }
}

TEST_CASE("out-of-tree backends can register at run time", "[registry]") {
  auto &registry = BackendRegistry::instance();

  registry.register_backend("counting", [](const InferenceConfig &config) {
    return std::make_shared<CountingBackend>(config);
  });
  REQUIRE(registry.has("counting"));

  // Re-registering is an error unless explicitly overwriting.
  REQUIRE_THROWS_AS(registry.register_backend(
                        "counting",
                        [](const InferenceConfig &config) {
                          return std::make_shared<CountingBackend>(config);
                        }),
                    InferenceError);
  REQUIRE_NOTHROW(registry.register_backend(
      "counting",
      [](const InferenceConfig &config) {
        return std::make_shared<CountingBackend>(config);
      },
      /*overwrite=*/true));

  REQUIRE_THROWS_AS(registry.register_backend("", nullptr), InferenceError);
  REQUIRE_THROWS_AS(registry.register_backend("null_factory", nullptr),
                    InferenceError);

  InferenceConfig config;
  config.backend = "counting";
  auto backend = create_backend(config);
  REQUIRE(backend->name() == "Counting");

  TensorMap inputs;
  Tensor &x = inputs.emplace("x", {3});
  x.flat<double>(0) = 1.0;
  x.flat<double>(1) = 2.0;
  x.flat<double>(2) = 4.0;

  TensorMap outputs;
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(outputs.at("sum").flat<double>(0) == 7.0);
  REQUIRE(backend->infer_count() == 1);

  REQUIRE(registry.unregister_backend("counting"));
  REQUIRE_FALSE(registry.unregister_backend("counting"));
}

TEST_CASE("lifecycle hooks fire exactly once", "[registry]") {
  InferenceConfig config;
  CountingBackend backend(config);

  REQUIRE(backend.init_calls == 0);
  backend.initialize();
  backend.initialize(); // no-op
  REQUIRE(backend.init_calls == 1);

  backend.finalize();
  backend.finalize(); // no-op
  REQUIRE(backend.final_calls == 1);

  // Re-initialization after finalize is allowed (restart-like use).
  backend.initialize();
  REQUIRE(backend.init_calls == 2);
  backend.finalize();
  REQUIRE(backend.final_calls == 2);
}

} // namespace test
} // namespace inference
} // namespace emulator
