// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>
#include <vector>

#include "create_inference_backend.hpp"
#include "python_inference_backend.hpp"
#include "python_interpreter.hpp"

#ifndef EMULATOR_TEST_PYTHON_DIR
#error "EMULATOR_TEST_PYTHON_DIR must point at the python fixtures directory"
#endif

namespace emulator {
namespace inference {
namespace test {

namespace {

InferenceConfig python_config(const std::string &module) {
  InferenceConfig config;
  config.backend = "python";
  config.set("python_module", module);
  config.set("python_path", std::string(EMULATOR_TEST_PYTHON_DIR));
  return config;
}

} // namespace

TEST_CASE("the python backend is registered", "[python]") {
  REQUIRE(BackendRegistry::instance().has("python"));
}

TEST_CASE("in-place python emulator writes into caller memory", "[python]") {
  auto config = python_config("emulator_inout");
  config.inputs.push_back(TensorSpec("T", {-1, 3}, DType::FLOAT64));
  config.outputs.push_back(TensorSpec("dT", {-1, 3}, DType::FLOAT64));
  config.set("scale", 2.0).set("offset", 1.0);

  auto backend = create_backend(config);
  REQUIRE(backend->name() == "Python");
  backend->initialize();

  // The component's own arrays; the backend hands numpy views of these to
  // python, so no copy happens in either direction.
  std::vector<double> field_in{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> field_out(6, -999.0);

  TensorMap inputs;
  inputs.wrap("T", static_cast<const double *>(field_in.data()), {2, 3});
  TensorMap outputs;
  outputs.wrap("dT", field_out.data(), {2, 3});

  REQUIRE(backend->infer(inputs, outputs));

  for (std::size_t i = 0; i < field_in.size(); ++i) {
    REQUIRE(field_out[i] == Approx(2.0 * field_in[i] + 1.0));
  }

  // Second step: python sees the updated inputs through the same views.
  field_in[0] = 10.0;
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(field_out[0] == Approx(21.0));
  REQUIRE(backend->infer_count() == 2);

  backend->finalize();
}

TEST_CASE("in-place python emulator can allocate its outputs", "[python]") {
  auto config = python_config("emulator_inout");
  config.inputs.push_back(TensorSpec("T", {-1, 2}, DType::FLOAT64));
  config.outputs.push_back(TensorSpec("dT", {-1, 2}, DType::FLOAT64));

  auto backend = create_backend(config);

  TensorMap inputs;
  Tensor &T = inputs.emplace("T", {4, 2});
  for (std::int64_t i = 0; i < T.size(); ++i) {
    T.flat<double>(i) = static_cast<double>(i);
  }

  TensorMap outputs; // empty: shapes come from the declared specs
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs.at("dT").dims() == std::vector<std::int64_t>{4, 2});
  REQUIRE(outputs.at("dT").flat<double>(3) == Approx(7.0)); // 2*3 + 1
}

TEST_CASE("python float32 model consumes float64 fields", "[python]") {
  // The common real case: E3SM carries r8 fields, the model is single
  // precision.  Declaring float32 specs makes the conversion explicit.
  auto config = python_config("emulator_inout");
  config.inputs.push_back(TensorSpec("T", {-1, 2}, DType::FLOAT32));
  config.outputs.push_back(TensorSpec("dT", {-1, 2}, DType::FLOAT32));

  auto backend = create_backend(config);

  std::vector<double> r8_field{1.5, 2.5, 3.5, 4.5};
  TensorMap inputs;
  Tensor &T = inputs.emplace("T", {2, 2}, DType::FLOAT32);
  T.copy_from_host(r8_field.data(), 4);

  TensorMap outputs;
  REQUIRE(backend->infer(inputs, outputs));

  std::vector<double> back(4, 0.0);
  outputs.at("dT").copy_to_host(back.data(), 4);
  REQUIRE(back[0] == Approx(4.0));
  REQUIRE(back[3] == Approx(10.0));
}

TEST_CASE("return-style python emulator", "[python]") {
  auto config = python_config("emulator_return");
  config.inputs.push_back(TensorSpec("T", {-1, 3}, DType::FLOAT64));
  config.inputs.push_back(TensorSpec("ps", {-1}, DType::FLOAT64));
  config.outputs.push_back(TensorSpec("total", {-1}, DType::FLOAT32));
  config.outputs.push_back(TensorSpec("doubled", {-1, 3}, DType::FLOAT64));

  auto backend = create_backend(config);
  backend->initialize();

  // Style detection: infer(self, inputs) takes one argument.
  const auto *py = dynamic_cast<const PythonBackend *>(backend.get());
  REQUIRE(py != nullptr);
  REQUIRE_FALSE(py->uses_inout_style());

  TensorMap inputs;
  Tensor &T = inputs.emplace("T", {2, 3});
  for (std::int64_t i = 0; i < T.size(); ++i) {
    T.flat<double>(i) = 1.0;
  }
  Tensor &ps = inputs.emplace("ps", {2});
  ps.flat<double>(0) = 10.0;
  ps.flat<double>(1) = 20.0;

  TensorMap outputs;
  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(outputs.size() == 2);
  // Declared order is preserved even though python returned a dict.
  REQUIRE(outputs[0].name() == "total");
  REQUIRE(outputs.at("total").dtype() == DType::FLOAT32);
  REQUIRE(outputs.at("total").flat<float>(0) == Approx(13.0f));
  REQUIRE(outputs.at("total").flat<float>(1) == Approx(23.0f));
  REQUIRE(outputs.at("doubled").flat<double>(0) == Approx(2.0));
}

TEST_CASE("return-style results can land in caller memory", "[python]") {
  auto config = python_config("emulator_return");
  config.inputs.push_back(TensorSpec("T", {-1, 3}, DType::FLOAT64));
  config.inputs.push_back(TensorSpec("ps", {-1}, DType::FLOAT64));

  auto backend = create_backend(config);

  TensorMap inputs;
  Tensor &T = inputs.emplace("T", {1, 3});
  T.flat<double>(0) = 1.0;
  T.flat<double>(1) = 2.0;
  T.flat<double>(2) = 3.0;
  inputs.emplace("ps", {1}).flat<double>(0) = 4.0;

  // Only ask for one of the two returned fields, straight into our array.
  std::vector<double> total(1, 0.0);
  TensorMap outputs;
  outputs.wrap("total", total.data(), {1});

  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(total[0] == Approx(10.0));
}

TEST_CASE("module-level python infer needs no class", "[python]") {
  auto config = python_config("emulator_module_level");
  config.outputs.push_back(TensorSpec("y", {-1}, DType::FLOAT64));

  auto backend = create_backend(config);

  TensorMap inputs;
  Tensor &x = inputs.emplace("x", {3});
  x.flat<double>(0) = 1.0;
  x.flat<double>(1) = -2.0;
  x.flat<double>(2) = 3.0;

  TensorMap outputs;
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(outputs.at("y").flat<double>(0) == Approx(-1.0));
  REQUIRE(outputs.at("y").flat<double>(1) == Approx(2.0));
}

TEST_CASE("python errors surface with their traceback", "[python]") {
  SECTION("missing module") {
    auto config = python_config("no_such_emulator_module");
    auto backend = create_backend(config);
    REQUIRE_THROWS_WITH(backend->initialize(),
                        Catch::Contains("no_such_emulator_module"));
  }

  SECTION("missing python_module option") {
    InferenceConfig config;
    config.backend = "python";
    auto backend = create_backend(config);
    REQUIRE_THROWS_WITH(backend->initialize(),
                        Catch::Contains("python_module"));
  }

  SECTION("factory raises") {
    auto config = python_config("emulator_broken");
    config.set("fail_at", std::string("create"));
    auto backend = create_backend(config);
    REQUIRE_THROWS_WITH(backend->initialize(),
                        Catch::Contains("deliberate failure in "
                                        "create_emulator"));
  }

  SECTION("infer raises") {
    auto config = python_config("emulator_broken");
    config.outputs.push_back(TensorSpec("y", {-1}, DType::FLOAT64));
    auto backend = create_backend(config);

    TensorMap inputs;
    inputs.emplace("x", {2});
    TensorMap outputs;
    REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                        Catch::Contains("deliberate failure in infer"));
  }

  SECTION("no destination for the results") {
    auto config = python_config("emulator_inout"); // in-place style
    auto backend = create_backend(config);         // but no outputs declared

    TensorMap inputs;
    inputs.emplace("T", {1, 3});
    TensorMap outputs;
    REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                        Catch::Contains("output"));
  }
}

TEST_CASE("a failed init does not leak an interpreter customer", "[python]") {
  const int before = PyInterpreter::instance().num_customers();

  auto config = python_config("emulator_broken");
  config.set("fail_at", std::string("create"));
  auto backend = create_backend(config);
  REQUIRE_THROWS_AS(backend->initialize(), InferenceError);

  REQUIRE(PyInterpreter::instance().num_customers() == before);
}

TEST_CASE("several python backends share one interpreter", "[python]") {
  auto config_a = python_config("emulator_inout");
  config_a.outputs.push_back(TensorSpec("dT", {-1, 2}, DType::FLOAT64));
  auto config_b = python_config("emulator_module_level");
  config_b.outputs.push_back(TensorSpec("y", {-1}, DType::FLOAT64));

  auto a = create_and_init_backend(config_a);
  auto b = create_and_init_backend(config_b);
  REQUIRE(PyInterpreter::instance().num_customers() >= 2);

  TensorMap in_a;
  in_a.emplace("T", {1, 2}).flat<double>(0) = 1.0;
  TensorMap out_a;
  REQUIRE(a->infer(in_a, out_a));

  TensorMap in_b;
  in_b.emplace("x", {2}).flat<double>(0) = 5.0;
  TensorMap out_b;
  REQUIRE(b->infer(in_b, out_b));
  REQUIRE(out_b.at("y").flat<double>(0) == Approx(-5.0));

  // Finalizing one must not tear the interpreter out from under the other.
  a->finalize();
  REQUIRE(b->infer(in_b, out_b));
  b->finalize();
}

} // namespace test
} // namespace inference
} // namespace emulator
