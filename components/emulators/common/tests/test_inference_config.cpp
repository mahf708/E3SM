// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "inference_config.hpp"

namespace emulator {
namespace inference {
namespace test {

TEST_CASE("InferenceConfig defaults", "[inference_config]") {
  InferenceConfig config;

  REQUIRE(config.backend == "stub");
  REQUIRE(config.model_path.empty());
  REQUIRE(config.input_channels == 0);
  REQUIRE(config.output_channels == 0);
  REQUIRE_FALSE(config.verbose);
  REQUIRE(config.inputs.empty());
  REQUIRE(config.outputs.empty());
  REQUIRE(config.options.empty());
}

TEST_CASE("InferenceConfig can be set", "[inference_config]") {
  InferenceConfig config;
  config.input_channels = 10;
  config.output_channels = 5;
  config.verbose = true;

  REQUIRE(config.input_channels == 10);
  REQUIRE(config.output_channels == 5);
  REQUIRE(config.verbose);
}

TEST_CASE("option accessors convert and validate", "[inference_config]") {
  InferenceConfig config;
  config.set("device", "cuda")
      .set("threads", 4)
      .set("scale", 0.5)
      .set("enabled", true);

  REQUIRE(config.has("device"));
  REQUIRE_FALSE(config.has("missing"));

  REQUIRE(config.get("device") == "cuda");
  REQUIRE(config.get("missing", "cpu") == "cpu");
  REQUIRE(config.get_int("threads") == 4);
  REQUIRE(config.get_int("missing", 7) == 7);
  REQUIRE(config.get_double("scale") == 0.5);
  REQUIRE(config.get_bool("enabled"));
  REQUIRE(config.get_bool("missing", true));

  REQUIRE(config.get_required("device") == "cuda");
  REQUIRE_THROWS_AS(config.get_required("missing"), InferenceError);
  REQUIRE_THROWS_AS(config.get_int("device"), InferenceError);

  // Fortran-style and human-style booleans both work.
  config.set("a", std::string(".true.")).set("b", std::string("Off"));
  REQUIRE(config.get_bool("a"));
  REQUIRE_FALSE(config.get_bool("b"));
  config.set("c", std::string("maybe"));
  REQUIRE_THROWS_AS(config.get_bool("c"), InferenceError);
}

TEST_CASE("from_string parses the line format", "[inference_config]") {
  const std::string text = R"(
# an emulator inference configuration
backend: onnx
model_path: /models/atm.onnx
verbose: true
input: T[-1,72]:float32
input: ps[-1]:float32
output: dT[-1,72]:float32
input_channels: 73
output_channels: 72
device: cpu             # trailing comments are dropped
option.intra_op_threads: 8
)";

  const auto config = InferenceConfig::from_string(text);

  REQUIRE(config.backend == "onnx");
  REQUIRE(config.model_path == "/models/atm.onnx");
  REQUIRE(config.verbose);

  // Declared inputs keep their file order (positional backends rely on it).
  REQUIRE(config.inputs.size() == 2);
  REQUIRE(config.inputs[0].name == "T");
  REQUIRE(config.inputs[0].dims == std::vector<std::int64_t>{-1, 72});
  REQUIRE(config.inputs[0].dtype == DType::FLOAT32);
  REQUIRE(config.inputs[1].name == "ps");

  REQUIRE(config.outputs.size() == 1);
  REQUIRE(config.outputs[0].name == "dT");

  REQUIRE(config.input_channels == 73);
  REQUIRE(config.output_channels == 72);

  // Unknown keys land in options, with or without the `option.` prefix.
  REQUIRE(config.get("device") == "cpu");
  REQUIRE(config.get_int("intra_op_threads") == 8);
}

TEST_CASE("from_string reports bad input", "[inference_config]") {
  REQUIRE_THROWS_AS(InferenceConfig::from_string("this line has no colon"),
                    InferenceError);
  REQUIRE_THROWS_AS(InferenceConfig::from_string("input: T[bogus]"),
                    InferenceError);
  REQUIRE_THROWS_AS(InferenceConfig::from_string("input_channels: many"),
                    InferenceError);
  REQUIRE_THROWS_AS(InferenceConfig::from_string("verbose: perhaps"),
                    InferenceError);
  REQUIRE_THROWS_AS(InferenceConfig::from_file("/nonexistent/inference.cfg"),
                    InferenceError);

  // Comments, blanks and quoted values are all tolerated.
  const auto ok = InferenceConfig::from_string(
      "\n! fortran-style comment\n\nmodel_path: \"/path/with # hash.onnx\"\n");
  REQUIRE(ok.model_path == "/path/with # hash.onnx");
}

TEST_CASE("prefixed parsing pulls settings out of a namelist",
          "[inference_config]") {
  // A component can hand its whole namelist to the inference layer.
  const std::string atm_in = R"(
nx: 90
ny: 45
grid: ne4pg2
inference.backend: python
inference.python_module: my_emulator
inference.input: T[-1,4]:float32
inference.verbose: true
)";

  const auto config = InferenceConfig::from_string_with_prefix(atm_in);

  REQUIRE(config.backend == "python");
  REQUIRE(config.get("python_module") == "my_emulator");
  REQUIRE(config.inputs.size() == 1);
  REQUIRE(config.verbose);
  // Component settings are not mistaken for inference options.
  REQUIRE_FALSE(config.has("nx"));
  REQUIRE_FALSE(config.has("grid"));
}

TEST_CASE("to_string reports the configuration", "[inference_config]") {
  InferenceConfig config;
  config.backend = "torch";
  config.model_path = "model.pt";
  config.inputs.push_back(TensorSpec("x", {-1, 2}, DType::FLOAT32));
  config.set("device", "cpu");

  const std::string dump = config.to_string();
  REQUIRE(dump.find("torch") != std::string::npos);
  REQUIRE(dump.find("model.pt") != std::string::npos);
  REQUIRE(dump.find("x[-1,2]:float32") != std::string::npos);
  REQUIRE(dump.find("device = cpu") != std::string::npos);
  REQUIRE(dump.find("<unspecified>") != std::string::npos); // no outputs
}

} // namespace test
} // namespace inference
} // namespace emulator
