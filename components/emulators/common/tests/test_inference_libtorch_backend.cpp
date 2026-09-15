// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"
#include "inference_error.hpp"
#include "libtorch_inference_backend.hpp"

// Only to ask whether this machine has a GPU, which decides whether the
// `device: cuda` case should throw or run.  Nothing else here needs torch.
#include <torch/cuda.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace emulator {
namespace inference {
namespace test {

namespace {

/// The fixtures are generated into the build tree; see the test CMakeLists.
std::string fixture(const std::string &name) {
  return std::string(EMULATOR_TEST_TORCH_FIXTURE_DIR) + "/" + name;
}

InferenceConfig affine_config() {
  InferenceConfig config;
  config.backend = "libtorch";
  config.model_path = fixture("libtorch_affine.pt");
  return config;
}

/// 1..24 as doubles, the [1,3,2,4] field the fixtures were traced on.
std::vector<double> ramp(std::size_t n = 24) {
  std::vector<double> v(n);
  std::iota(v.begin(), v.end(), 1.0);
  return v;
}

const std::vector<std::int64_t> kFieldDims{1, 3, 2, 4};

} // namespace

TEST_CASE("The libtorch backend runs a traced module", "[libtorch]") {
  auto backend = create_backend(affine_config(), InferenceContext());
  REQUIRE(backend->name() == "libtorch");
  REQUIRE(backend->is_initialized());

  const auto x = ramp();
  std::vector<double> y(24, -999.0);

  TensorMap inputs;
  inputs.wrap("x", x.data(), kFieldDims);
  TensorMap outputs;
  outputs.wrap("y", y.data(), kFieldDims);

  REQUIRE(backend->infer(inputs, outputs));

  // 2*x + 1 is exact in float32 for these values, so the double -> float ->
  // double round trip has to reproduce it to the bit.
  for (std::size_t i = 0; i < y.size(); ++i) {
    REQUIRE(y[i] == 2.0 * x[i] + 1.0);
  }

  // A second step, because a coupled run makes thousands of them and the
  // module has to survive being called again.
  std::fill(y.begin(), y.end(), -999.0);
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(y[0] == 3.0);

  // And nothing wrote back through the input view.
  REQUIRE(x[0] == 1.0);

  backend->finalize();
}

TEST_CASE("available_backends advertises libtorch", "[libtorch]") {
  const auto names = available_backends();
  REQUIRE(std::find(names.begin(), names.end(), "libtorch") != names.end());
}

TEST_CASE("The dtype option decides the precision actually used",
          "[libtorch]") {
  // 0.1 is not representable in either binary format, so 2*0.1+1 computed in
  // float32 and in float64 differ -- which is what makes this a test of the
  // conversion rather than of the arithmetic.
  const double exact = 2.0 * 0.1 + 1.0;
  const std::vector<double> x{0.1};
  std::vector<double> y{0.0};

  const std::vector<std::int64_t> one{1, 1, 1, 1};

  SECTION("float32, the default, loses the tail") {
    auto backend = create_backend(affine_config(), InferenceContext());
    TensorMap in;
    in.wrap("x", x.data(), one);
    TensorMap out;
    out.wrap("y", y.data(), one);
    REQUIRE(backend->infer(in, out));
    REQUIRE(y[0] == Approx(exact));
    REQUIRE(y[0] != exact);
    backend->finalize();
  }

  SECTION("float64 keeps every bit") {
    auto config = affine_config();
    config.set("dtype", "float64");
    auto backend = create_backend(config, InferenceContext());
    TensorMap in;
    in.wrap("x", x.data(), one);
    TensorMap out;
    out.wrap("y", y.data(), one);
    REQUIRE(backend->infer(in, out));
    REQUIRE(y[0] == exact);
    backend->finalize();
  }
}

TEST_CASE("A tuple return lands in the outputs in order", "[libtorch]") {
  auto config = affine_config();
  config.model_path = fixture("libtorch_tuple.pt");
  auto backend = create_backend(config, InferenceContext());

  const auto x = ramp();
  std::vector<double> a(24, 0.0);
  std::vector<double> b(24, 0.0);

  TensorMap inputs;
  inputs.wrap("x", x.data(), kFieldDims);
  TensorMap outputs;
  outputs.wrap("a", a.data(), kFieldDims);
  outputs.wrap("b", b.data(), kFieldDims);

  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(a[0] == 2.0 * x[0] + 1.0);
  REQUIRE(b[0] == x[0] - 1.0);
  REQUIRE(a[23] == 2.0 * x[23] + 1.0);
  REQUIRE(b[23] == x[23] - 1.0);

  backend->finalize();
}

TEST_CASE("A wrong number of outputs is refused", "[libtorch]") {
  // The single-tensor module with two outputs supplied: matched in order, so
  // the counts have to agree rather than the extra output being ignored.
  auto backend = create_backend(affine_config(), InferenceContext());

  const auto x = ramp();
  std::vector<double> a(24, 0.0);
  std::vector<double> b(24, 0.0);

  TensorMap inputs;
  inputs.wrap("x", x.data(), kFieldDims);
  TensorMap outputs;
  outputs.wrap("a", a.data(), kFieldDims);
  outputs.wrap("b", b.data(), kFieldDims);

  REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                      Catch::Contains("counts must agree"));
  backend->finalize();
}

TEST_CASE("An output shape mismatch throws with both shapes", "[libtorch]") {
  // 24 elements in, 12 declared out.  Reshaping to fit is exactly the
  // silent-transpose bug this refuses to enable.
  auto backend = create_backend(affine_config(), InferenceContext());

  const auto x = ramp();
  std::vector<double> y(12, 0.0);

  TensorMap inputs;
  inputs.wrap("x", x.data(), kFieldDims);
  TensorMap outputs;
  outputs.wrap("y", y.data(), {1, 3, 2, 2});

  REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                      Catch::Contains("[1,3,2,4]") &&
                          Catch::Contains("[1,3,2,2]") &&
                          Catch::Contains("'y'"));
  backend->finalize();
}

TEST_CASE("An input shape the module rejects is reported", "[libtorch]") {
  // The channels fixture is fixed to 3 channels, like every real conv net.
  // Handing it 4 fails inside forward(), and the error has to name the
  // shapes that were handed over.
  auto config = affine_config();
  config.model_path = fixture("libtorch_channels.pt");
  auto backend = create_backend(config, InferenceContext());

  const auto x = ramp(32);
  std::vector<double> y(8, 0.0);

  TensorMap inputs;
  inputs.wrap("x", x.data(), {1, 4, 2, 4});
  TensorMap outputs;
  outputs.wrap("y", y.data(), {1, 1, 2, 4});

  REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                      Catch::Contains("forward()") &&
                          Catch::Contains("x[1,4,2,4]"));
  backend->finalize();
}

TEST_CASE("The channels fixture computes what it says", "[libtorch]") {
  // A model whose output shape differs from its input shape, which is the
  // case the "pass each tensor through with its declared shape" contract
  // exists for.
  auto config = affine_config();
  config.model_path = fixture("libtorch_channels.pt");
  auto backend = create_backend(config, InferenceContext());

  // [1,3,2,4]: channel 0 is 1..8, channel 1 is 9..16, channel 2 is 17..24.
  const auto x = ramp();
  std::vector<double> y(8, 0.0);

  TensorMap inputs;
  inputs.wrap("x", x.data(), kFieldDims);
  TensorMap outputs;
  outputs.wrap("y", y.data(), {1, 1, 2, 4});

  REQUIRE(backend->infer(inputs, outputs));
  for (int i = 0; i < 8; ++i) {
    REQUIRE(y[i] == Approx(x[i] + 10.0 * x[i + 8] + 100.0 * x[i + 16]));
  }

  backend->finalize();
}

TEST_CASE("A missing or unreadable model is reported with its path",
          "[libtorch]") {
  auto config = affine_config();
  config.model_path = fixture("no_such_module.pt");
  REQUIRE_THROWS_WITH(create_backend(config, InferenceContext()),
                      Catch::Contains("no_such_module.pt"));

  SECTION("and so is no path at all") {
    auto empty = affine_config();
    empty.model_path.clear();
    REQUIRE_THROWS_WITH(create_backend(empty, InferenceContext()),
                        Catch::Contains("model_path"));
  }
}

TEST_CASE("A file that is not TorchScript is reported, not guessed at",
          "[libtorch]") {
  // This source file: readable, and definitely not a zip archive.
  auto config = affine_config();
  config.model_path = __FILE__;
  REQUIRE_THROWS_AS(create_backend(config, InferenceContext()), InferenceError);
}

TEST_CASE("CUDA is never silently swapped for the CPU", "[libtorch]") {
  auto config = affine_config();
  config.set("device", "cuda");

  if (torch::cuda::is_available()) {
    // A GPU build on a GPU node: the device path has to actually run, and
    // the answers must not depend on where it ran.
    auto backend = create_backend(config, InferenceContext());
    const auto x = ramp();
    std::vector<double> y(24, 0.0);
    TensorMap inputs;
    inputs.wrap("x", x.data(), kFieldDims);
    TensorMap outputs;
    outputs.wrap("y", y.data(), kFieldDims);
    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(y[0] == 2.0 * x[0] + 1.0);
    REQUIRE(y[23] == 2.0 * x[23] + 1.0);
    backend->finalize();
  } else {
    // A CPU-only build or a CPU node: the request is fatal.  Falling back
    // would leave a run that was sized for a GPU quietly ~100x slower.
    REQUIRE_THROWS_WITH(create_backend(config, InferenceContext()),
                        Catch::Contains("CUDA") &&
                            Catch::Contains("device: cpu"));
  }
}

TEST_CASE("An out-of-range or unknown device is refused", "[libtorch]") {
  SECTION("a device that is not a device") {
    auto config = affine_config();
    config.set("device", "tpu");
    REQUIRE_THROWS_WITH(create_backend(config, InferenceContext()),
                        Catch::Contains("'cpu', 'cuda' or 'cuda:N'"));
  }

  SECTION("a GPU ordinal this machine does not have") {
    auto config = affine_config();
    config.set("device", "cuda:99");
    // Either message is correct, depending on the machine: no CUDA at all,
    // or CUDA with fewer than 100 devices.  Both must throw.
    REQUIRE_THROWS_AS(create_backend(config, InferenceContext()),
                      InferenceError);
  }
}

TEST_CASE("An unknown dtype is refused", "[libtorch]") {
  auto config = affine_config();
  config.set("dtype", "bfloat16");
  REQUIRE_THROWS_WITH(create_backend(config, InferenceContext()),
                      Catch::Contains("'float32' or 'float64'"));
}

TEST_CASE("num_threads is accepted and validated", "[libtorch]") {
  auto config = affine_config();
  config.set("num_threads", "2");
  auto backend = create_backend(config, InferenceContext());

  const auto x = ramp();
  std::vector<double> y(24, 0.0);
  TensorMap inputs;
  inputs.wrap("x", x.data(), kFieldDims);
  TensorMap outputs;
  outputs.wrap("y", y.data(), kFieldDims);
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(y[5] == 2.0 * x[5] + 1.0);
  backend->finalize();

  auto bad = affine_config();
  bad.set("num_threads", "-1");
  REQUIRE_THROWS_AS(create_backend(bad, InferenceContext()), InferenceError);
}

TEST_CASE("An empty field still goes through", "[libtorch]") {
  // A rank owning no columns is normal on a large layout, and inference is
  // collective, so a zero-length field must not throw.  The module is
  // elementwise, so an empty input is an empty output.
  auto backend = create_backend(affine_config(), InferenceContext());

  TensorMap inputs;
  inputs.wrap("x", static_cast<const double *>(nullptr), {0});
  TensorMap outputs;
  outputs.wrap("y", static_cast<double *>(nullptr), {0});
  REQUIRE(backend->infer(inputs, outputs));

  backend->finalize();
}

TEST_CASE("No inputs at all is refused", "[libtorch]") {
  auto backend = create_backend(affine_config(), InferenceContext());
  TensorMap inputs;
  TensorMap outputs;
  REQUIRE_THROWS_AS(backend->infer(inputs, outputs), InferenceError);
  backend->finalize();
}

TEST_CASE("Inference before initialize() is refused", "[libtorch]") {
  LibTorchBackend backend{affine_config(), InferenceContext()};
  TensorMap inputs;
  TensorMap outputs;
  REQUIRE_THROWS_AS(backend.infer(inputs, outputs), InferenceError);
}

TEST_CASE("The lifecycle is idempotent at both ends", "[libtorch]") {
  auto backend = create_backend(affine_config(), InferenceContext());

  backend->initialize(); // already initialized by the factory
  REQUIRE(backend->is_initialized());

  backend->finalize();
  backend->finalize(); // a component may finalize and then be destroyed
  REQUIRE_FALSE(backend->is_initialized());

  // And it can be brought back up: the module reloads onto a backend whose
  // weights were dropped by finalize().
  backend->initialize();
  const auto x = ramp();
  std::vector<double> y(24, 0.0);
  TensorMap inputs;
  inputs.wrap("x", x.data(), kFieldDims);
  TensorMap outputs;
  outputs.wrap("y", y.data(), kFieldDims);
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(y[0] == 3.0);
  backend->finalize();
}

TEST_CASE("Verbose reports once at init and once at the first step",
          "[libtorch]") {
  // Not a test of the text, only that turning it on does not change any
  // answer and does not fail on the second step -- the per-step report is
  // deliberately first-call only, because this runs every coupler timestep.
  auto config = affine_config();
  config.verbose = true;
  auto backend = create_backend(config, InferenceContext());

  const auto x = ramp();
  std::vector<double> y(24, 0.0);
  TensorMap inputs;
  inputs.wrap("x", x.data(), kFieldDims);
  TensorMap outputs;
  outputs.wrap("y", y.data(), kFieldDims);
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(y[0] == 3.0);
  backend->finalize();
}

TEST_CASE("A seeded model draws the same noise at the same step",
          "[libtorch]") {
  auto config = affine_config();
  config.model_path = fixture("libtorch_noise.pt");
  config.set("seed", "20260912");
  // On a GPU node the draw happens on the device generator, which is seeded
  // separately from the CPU one; the property has to hold on both.
  const std::string device =
      GENERATE(values<std::string>({"cpu", "cuda"}));
  if (device == "cuda" && !torch::cuda::is_available()) {
    SUCCEED("no CUDA device here; the cpu case covers the property");
    return;
  }
  config.set("device", device);
  INFO("device " << device);

  const std::vector<double> x(24, 0.0);
  auto draw = [&](InferenceBackend &backend, std::int64_t step) {
    std::vector<double> y(24, 0.0);
    backend.set_step(step);
    TensorMap inputs;
    inputs.wrap("x", x.data(), kFieldDims);
    TensorMap outputs;
    outputs.wrap("y", y.data(), kFieldDims);
    REQUIRE(backend.infer(inputs, outputs));
    return y;
  };

  auto run = create_backend(config, InferenceContext());
  const auto step5 = draw(*run, 5);
  const auto step6 = draw(*run, 6);
  const auto step7 = draw(*run, 7);
  REQUIRE(step5 != step6);
  REQUIRE(step6 != step7);

  // A restart at step 6: a new backend in a new process, which has made no
  // draws.  Seeding once at init would give it the stream position of a run
  // that made none, not of the run that made two.
  auto restarted = create_backend(config, InferenceContext());
  REQUIRE(draw(*restarted, 6) == step6);
  REQUIRE(draw(*restarted, 7) == step7);

  // Recomputing a step is idempotent.
  REQUIRE(draw(*run, 5) == step5);

  run->finalize();
  restarted->finalize();
}

TEST_CASE("A seed without a step is refused", "[libtorch]") {
  auto config = affine_config();
  config.model_path = fixture("libtorch_noise.pt");
  config.set("seed", "1");
  auto backend = create_backend(config, InferenceContext());

  const std::vector<double> x(24, 0.0);
  std::vector<double> y(24, 0.0);
  TensorMap inputs;
  inputs.wrap("x", x.data(), kFieldDims);
  TensorMap outputs;
  outputs.wrap("y", y.data(), kFieldDims);
  REQUIRE_THROWS_WITH(backend->infer(inputs, outputs),
                      Catch::Contains("set_step()"));
  backend->finalize();

  auto bad = affine_config();
  bad.set("seed", "-3");
  REQUIRE_THROWS_WITH(create_backend(bad, InferenceContext()),
                      Catch::Contains("non-negative integer"));
}

} // namespace test
} // namespace inference
} // namespace emulator
