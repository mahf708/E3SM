/**
 * @file libtorch_inference_backend.cpp
 * @brief TorchScript inference backend implementation.
 *
 * Two things in here are worth knowing before changing anything:
 *
 *  1. The dtype round trip.  Framework tensors are `double`; the traced
 *     checkpoints are float32.  Going in, that conversion rides on
 *     `torch::from_blob` + `.to()`, so torch does the cast in its own
 *     vectorised kernel over a zero-copy view of E3SM memory.  Coming back
 *     out the destination is caller memory that torch knows nothing about,
 *     so the cast happens on torch's side (`.to(kFloat64)`) and the copy
 *     into the field is an explicit loop -- there is no `from_blob` trick
 *     that avoids it, and a loop says plainly what is happening.
 *
 *  2. No shape policy.  Inputs go to `forward` with the shapes their
 *     Tensors declare and outputs are checked only on element count.  See
 *     the class comment for why a helpful reshape here would be a bug
 *     factory.
 */

#include "libtorch_inference_backend.hpp"

#include "inference_error.hpp"

#include <ATen/Parallel.h> // at::set_num_threads, at::get_num_threads
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/csrc/jit/python/update_graph_executor_opt.h> // setGraphExecutorOptimize, plain C++
#include <torch/utils.h> // torch::manual_seed

#include "fpe_guard.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

/// "[1,7,180,360]", for error messages and the verbose report.
std::string shape_string(const c10::IntArrayRef &sizes) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    oss << (i ? "," : "") << sizes[i];
  }
  oss << "]";
  return oss.str();
}

std::string shape_string(const std::vector<std::int64_t> &dims) {
  return shape_string(c10::IntArrayRef(dims.data(), dims.size()));
}

/**
 * @brief Parse the `device` option.
 *
 * A silent fall back to the CPU when CUDA is missing is the one thing this
 * must not do: the run still produces answers, roughly a hundred times
 * slower, and nobody notices until the wall clock does.  So an unavailable
 * CUDA is fatal here.
 */
torch::Device parse_device(const std::string &spec) {
  const std::string s = lower(spec);

  if (s == "cpu") {
    return torch::Device(torch::kCPU);
  }

  if (s == "cuda" || s.rfind("cuda:", 0) == 0) {
    EMULATOR_INFER_REQUIRE(
        torch::cuda::is_available(),
        "The libtorch backend was asked for device '"
            << spec
            << "', but this build of libtorch reports no usable CUDA device. "
               "Refusing to fall back to the CPU silently: that is a ~100x "
               "slowdown on a run that was sized for a GPU. Set "
               "`device: cpu` if the CPU is really what you want.");

    if (s == "cuda") {
      return torch::Device(torch::kCUDA);
    }

    const std::string index = s.substr(5);
    int ordinal = -1;
    try {
      ordinal = std::stoi(index);
    } catch (const std::exception &) {
      EMULATOR_INFER_REQUIRE(false, "Could not read the device index in '"
                                        << spec << "' as an integer.");
    }
    EMULATOR_INFER_REQUIRE(
        ordinal >= 0 &&
            ordinal < static_cast<int>(torch::cuda::device_count()),
        "The libtorch backend was asked for '"
            << spec << "', but this machine has "
            << torch::cuda::device_count() << " CUDA device(s).");
    return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(ordinal));
  }

  EMULATOR_INFER_REQUIRE(false, "Unknown libtorch device '"
                                    << spec
                                    << "'. Use 'cpu', 'cuda' or 'cuda:N'.");
  return torch::Device(torch::kCPU); // not reached
}

/// Parse the `dtype` option: the precision the module's weights are in.
torch::ScalarType parse_dtype(const std::string &spec) {
  const std::string s = lower(spec);
  if (s == "float32" || s == "float" || s == "single" || s == "f32") {
    return torch::kFloat32;
  }
  if (s == "float64" || s == "double" || s == "f64") {
    return torch::kFloat64;
  }
  EMULATOR_INFER_REQUIRE(false, "Unknown libtorch dtype '"
                                    << spec
                                    << "'. Use 'float32' or 'float64'.");
  return torch::kFloat32; // not reached
}

/**
 * @brief Flatten whatever `forward` returned into a list of tensors.
 *
 * A single tensor, a tuple of tensors and a list of tensors are all normal
 * TorchScript return types and all three appear in the wild -- ACE returns
 * one tensor, a model with a diagnostic head returns a tuple.  Anything
 * else (a dict, a scalar, a nested tuple) is refused by name rather than
 * being flattened on a guess.
 */
std::vector<at::Tensor> collect_outputs(const torch::jit::IValue &result) {
  std::vector<at::Tensor> tensors;

  if (result.isTensor()) {
    tensors.push_back(result.toTensor());
    return tensors;
  }
  if (result.isTuple()) {
    for (const auto &element : result.toTuple()->elements()) {
      EMULATOR_INFER_REQUIRE(element.isTensor(),
                             "The module returned a tuple containing a "
                                 << element.tagKind()
                                 << ", not a tensor. The libtorch backend "
                                    "handles a tensor, or a tuple or list of "
                                    "tensors.");
      tensors.push_back(element.toTensor());
    }
    return tensors;
  }
  if (result.isTensorList()) {
    for (const auto &tensor : result.toTensorList()) {
      tensors.push_back(tensor);
    }
    return tensors;
  }

  EMULATOR_INFER_REQUIRE(false,
                         "The module returned a "
                             << result.tagKind()
                             << ". The libtorch backend handles a tensor, or "
                                "a tuple or list of tensors.");
  return tensors; // not reached
}

/**
 * @brief The seed for one step: a pure function of (seed, step).
 *
 * splitmix64 on both, so neighbouring steps do not get neighbouring seeds.
 * torch's generators are not known to misbehave on those, but nothing is
 * gained by finding out.
 */
std::uint64_t step_seed(std::uint64_t seed, std::int64_t step) {
  auto mix = [](std::uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  };
  return mix(seed ^ mix(static_cast<std::uint64_t>(step)));
}

} // namespace

// ===========================================================================
// Impl
// ===========================================================================

struct LibTorchBackend::Impl {
  torch::jit::Module module;
  bool loaded = false;

  torch::Device device{torch::kCPU};
  torch::ScalarType dtype = torch::kFloat32;
  int num_threads = 0;
  bool seeded = false;
  std::uint64_t seed = 0;
  int calls = 0;

  /// A torch tensor of the module's dtype, on the module's device, holding
  /// this field's values.
  at::Tensor to_torch(const Tensor &tensor) const {
    const auto opts = torch::TensorOptions().dtype(torch::kFloat64);

    at::Tensor host;
    if (tensor.size() == 0) {
      // from_blob over a null pointer is not something to find out about at
      // run time.  A rank owning no columns is normal on a large layout and
      // inference is collective, so an empty field has to go through.
      host = torch::empty(tensor.dims(), opts);
    } else {
      // A view of E3SM memory: nothing is copied here.  The cast away from
      // const is safe because the only thing done with this tensor is to
      // read it -- `.to()` below produces the buffer the module actually
      // sees whenever a conversion is needed.  When no conversion is needed
      // (a float64 module on the CPU) the module receives the view itself,
      // which is the zero-copy fast path; a traced module is a pure
      // function of its inputs and does not write to them.
      host = torch::from_blob(const_cast<double *>(tensor.cdata()),
                              tensor.dims(), opts);
    }
    // One conversion, not a cast on the host followed by a transfer: that
    // would allocate a float32 host copy of every field just to throw it away.
    return host.to(device, dtype);
  }

  /// Copy one of the module's outputs back into the component's field.
  void from_torch(const at::Tensor &result, Tensor &tensor) const {
    EMULATOR_INFER_REQUIRE(
        result.numel() == tensor.size(),
        "The module returned "
            << shape_string(result.sizes()) << " (" << result.numel()
            << " elements) for output '" << tensor.name() << "' "
            << shape_string(tensor.dims()) << " (" << tensor.size()
            << " elements). Fix the shapes at the call site rather than "
               "reshaping here: the two are not interchangeable, and a "
               "silently transposed field produces no NaN and no complaint.");

    if (tensor.size() == 0) {
      return;
    }

    // Back to the host, back to double, and made contiguous so the copy
    // below can walk it linearly.  Each of these is a no-op when it is
    // already true, so a float64 CPU module pays for none of them.
    const at::Tensor host =
        result.to(torch::kCPU).to(torch::kFloat64).contiguous();

    // An explicit loop rather than a torch copy: the destination is caller
    // memory (usually a view straight into an E3SM field) that torch has no
    // handle on, so something has to walk it either way.
    const double *src = host.data_ptr<double>();
    double *dst = tensor.data();
    for (std::int64_t i = 0; i < tensor.size(); ++i) {
      dst[i] = src[i];
    }
  }
};

// ===========================================================================
// LibTorchBackend
// ===========================================================================

LibTorchBackend::LibTorchBackend(const InferenceConfig &config,
                                 const InferenceContext &context)
    : InferenceBackend(config, context), m_impl(new Impl()) {}

LibTorchBackend::~LibTorchBackend() {
  try {
    LibTorchBackend::finalize();
  } catch (const std::exception &e) {
    // A destructor must not throw, and a failed teardown is not worth
    // aborting a run that is already shutting down.
    std::cerr << "[emulator::inference] warning: libtorch backend teardown "
              << "failed: " << e.what() << "\n";
  }
}

void LibTorchBackend::init_impl() {
  m_impl->device = parse_device(m_config.get("device", "cpu"));
  m_impl->dtype = parse_dtype(m_config.get("dtype", "float32"));
  m_impl->num_threads = m_config.get_int("num_threads", 0);

  // TorchScript's profiling executor runs a module's first calls on the
  // graph as traced and later calls on a re-optimized, fused graph, and the
  // two do not give bit-identical float32 results.  A run and its restart
  // then make their first inferences on different graphs, and the restart
  // cannot be exact.  Off by default here; on is faster once warm.
  // Process-wide: it is a global torch setting, not a per-module one.
  const bool optimize = m_config.get_bool("jit_optimize", false);
  torch::jit::setGraphExecutorOptimize(optimize);

  const std::string seed = m_config.get("seed");
  m_impl->seeded = !seed.empty();
  if (m_impl->seeded) {
    // Digits only: std::stoull accepts "-3" and wraps it to 2^64 - 3.
    bool valid = seed.find_first_not_of("0123456789") == std::string::npos;
    if (valid) {
      try {
        m_impl->seed = std::stoull(seed);
      } catch (const std::out_of_range &) {
        valid = false;
      }
    }
    EMULATOR_INFER_REQUIRE(valid, "Could not read the libtorch seed '"
                                      << seed
                                      << "' as a non-negative integer.");
  }

  EMULATOR_INFER_REQUIRE(
      m_impl->num_threads >= 0,
      "Negative num_threads " << m_impl->num_threads
                              << " for the libtorch backend; 0 means 'leave "
                                 "torch's own default alone'.");
  if (m_impl->num_threads > 0) {
    // Once, here, rather than per step: set_num_threads reconfigures the
    // intra-op pool, which is not something to do inside a timestep loop.
    at::set_num_threads(m_impl->num_threads);
  }

  EMULATOR_INFER_REQUIRE(!m_config.model_path.empty(),
                         "The libtorch backend needs `model_path` to point at "
                         "a TorchScript archive saved by torch.jit.save().");
  {
    std::ifstream probe(m_config.model_path, std::ios::binary);
    EMULATOR_INFER_REQUIRE(static_cast<bool>(probe),
                           "Cannot open the TorchScript module '"
                               << m_config.model_path
                               << "': no such file, or not readable.");
  }

  try {
    // Loading straight onto the target device, so a GPU model never has to
    // exist twice.
    // Loading runs the module's own initializers, which is enough to raise
    // an FPE before a single forward pass.
    FpeGuard no_fpe;
    m_impl->module = torch::jit::load(m_config.model_path, m_impl->device);
  } catch (const std::exception &e) {
    throw InferenceError(
        "Could not load the TorchScript module '" + m_config.model_path +
        "'. It must be an archive written by torch.jit.save(), and its "
        "bytecode version must be one this libtorch understands (they do not "
        "have to be the same torch release, but a much newer archive will be "
        "refused). Torch said:\n" +
        std::string(e.what()));
  }

  m_impl->module.to(m_impl->device);
  m_impl->module.eval(); // no dropout, no batchnorm updates
  m_impl->loaded = true;

  if (m_config.verbose && m_context.is_root()) {
    std::cout << "[emulator::inference] libtorch loaded "
              << m_config.model_path << " on " << m_impl->device.str()
              << ", model dtype "
              << (m_impl->dtype == torch::kFloat64 ? "float64" : "float32")
              << ", intra-op threads "
              << (m_impl->num_threads > 0
                      ? std::to_string(m_impl->num_threads)
                      : "default(" + std::to_string(at::get_num_threads()) +
                            ")")
              << "\n";
  }
}

bool LibTorchBackend::infer_impl(const TensorMap &inputs, TensorMap &outputs) {
  EMULATOR_INFER_REQUIRE(m_impl->loaded,
                         "The libtorch module is not loaded.");
  EMULATOR_INFER_REQUIRE(inputs.size() > 0,
                         "The libtorch backend was given no input tensors; "
                         "forward() takes the inputs positionally, in "
                         "TensorMap order.");

  // A stochastic model draws on every forward pass, so the generator is
  // reseeded before every one, from the step and not from a call count.
  // See InferenceBackend::set_step() for why seeding once is not enough.
  if (m_impl->seeded) {
    EMULATOR_INFER_REQUIRE(
        m_step >= 0,
        "The libtorch backend has `seed` set but was never told the step. "
        "Call set_step() with the component's counted step before infer(): "
        "without it every step would draw the same noise.");
    torch::manual_seed(step_seed(m_impl->seed, m_step));
  }

  ++m_impl->calls;
  // Only on the first step: this runs every coupler timestep, and a line of
  // log per step per rank is how a log directory fills up.
  const bool report =
      m_config.verbose && m_context.is_root() && m_impl->calls == 1;

  std::vector<torch::jit::IValue> args;
  args.reserve(inputs.size());
  for (const auto &tensor : inputs) {
    if (report) {
      std::cout << "[emulator::inference] libtorch input " << tensor.name()
                << " " << shape_string(tensor.dims()) << "\n";
    }
    args.emplace_back(m_impl->to_torch(tensor));
  }

  torch::jit::IValue result;
  try {
    // Inference only: no autograd graph, which otherwise quietly retains
    // every intermediate for the life of the step.
    torch::NoGradGuard no_grad;
    // torch's kernels raise benign floating-point exceptions -- a softmax or
    // a normalization dividing by a zero it is about to mask is ordinary
    // inside a network and fatal under the traps an E3SM debug build
    // enables.  Suspended only around the model call, and restored exactly.
    FpeGuard no_fpe;
    result = m_impl->module.forward(args);
  } catch (const std::exception &e) {
    std::ostringstream shapes;
    for (const auto &tensor : inputs) {
      shapes << " " << tensor.name() << shape_string(tensor.dims());
    }
    std::string hint;
    const std::string what = e.what();
    if (m_impl->device.is_cpu() &&
        what.find("cuda") != std::string::npos) {
      // The real ACE trace does this: fme's sht_fix.py builds a tensor with
      // device=cuda:0, and tracing bakes that constant into the graph.
      hint = " The module mentions CUDA while running on the CPU: a module "
             "traced on a GPU keeps the device of any tensor it created "
             "during tracing, so it cannot run on the CPU at all. Trace it "
             "on the CPU, or run it with `device: cuda`.";
    }
    throw InferenceError("The TorchScript module '" + m_config.model_path +
                         "' failed in forward() with inputs" + shapes.str() +
                         "." + hint + " Torch said:\n" + what);
  }

  const std::vector<at::Tensor> results = collect_outputs(result);
  EMULATOR_INFER_REQUIRE(
      results.size() == outputs.size(),
      "The module returned " << results.size() << " tensor(s) but "
                             << outputs.size()
                             << " output tensor(s) were supplied. They are "
                                "matched in order, so the counts must agree.");

  std::size_t i = 0;
  for (auto &tensor : outputs) {
    if (report) {
      std::cout << "[emulator::inference] libtorch output " << tensor.name()
                << " " << shape_string(tensor.dims()) << " <- module "
                << shape_string(results[i].sizes()) << "\n";
    }
    m_impl->from_torch(results[i], tensor);
    ++i;
  }

  return true;
}

void LibTorchBackend::final_impl() {
  if (!m_impl->loaded) {
    return;
  }
  // Replacing the module is what actually drops the weights (and the device
  // memory holding them).  Assigning a fresh empty Module is the only way
  // torch::jit::Module offers to do that.
  m_impl->module = torch::jit::Module();
  m_impl->loaded = false;
  m_impl->calls = 0;
}

} // namespace inference
} // namespace emulator
