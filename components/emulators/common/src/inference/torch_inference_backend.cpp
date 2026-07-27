/**
 * @file torch_inference_backend.cpp
 * @brief LibTorch (TorchScript) inference backend implementation.
 */

#include "torch_inference_backend.hpp"

#include <cctype>
#include <iostream>
#include <string>
#include <vector>

#include <ATen/Parallel.h> // at::set_num_threads
#include <torch/cuda.h>    // torch::cuda::is_available
#include <torch/script.h>

namespace emulator {
namespace inference {

namespace {

/// Map a torch element type onto ours; false for types we do not carry.
bool from_torch_type(torch::ScalarType scalar_type, DType &out) {
  switch (scalar_type) {
  case torch::kFloat:
    out = DType::FLOAT32;
    return true;
  case torch::kDouble:
    out = DType::FLOAT64;
    return true;
  case torch::kInt:
    out = DType::INT32;
    return true;
  case torch::kLong:
    out = DType::INT64;
    return true;
  default:
    return false;
  }
}

torch::ScalarType to_torch_type(DType dtype) {
  switch (dtype) {
  case DType::FLOAT32:
    return torch::kFloat;
  case DType::FLOAT64:
    return torch::kDouble;
  case DType::INT32:
    return torch::kInt;
  case DType::INT64:
    return torch::kLong;
  }
  return torch::kFloat;
}

/// Lowercase and strip whitespace, for option values.
std::string normalize(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      out.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return out;
}

/**
 * @brief Apply LibTorch's process-wide thread settings, once.
 *
 * `at::set_num_threads` and `at::set_num_interop_threads` configure the whole
 * process, not one module: two backends cannot hold different policies, and
 * `set_num_interop_threads` *throws* once the interop pool exists, so a second
 * backend calling it would abort a perfectly good run.  Apply the first
 * request, then report — rather than obey — any later disagreement.
 */
void apply_thread_settings(int intra, int inter, bool verbose) {
  static bool applied = false;
  static int applied_intra = -1;
  static int applied_inter = -1;

  if (applied) {
    if ((intra > 0 && intra != applied_intra) ||
        (inter > 0 && inter != applied_inter)) {
      std::cerr << "[emulator::inference] warning: LibTorch thread counts are "
                   "process-wide and were already set to intra="
                << applied_intra << " inter=" << applied_inter
                << "; ignoring the request for intra=" << intra
                << " inter=" << inter << ".\n";
    }
    return;
  }

  // Torch throws if the pools are already running; that is not a reason to
  // fail model setup, so report and carry on with whatever it has.
  try {
    if (intra > 0) {
      at::set_num_threads(intra);
      applied_intra = intra;
    }
    if (inter > 0) {
      at::set_num_interop_threads(inter);
      applied_inter = inter;
    }
  } catch (const std::exception &e) {
    std::cerr << "[emulator::inference] warning: LibTorch would not accept the "
                 "requested thread counts (intra="
              << intra << " inter=" << inter << "): " << e.what() << "\n";
  }
  applied = true;

  if (verbose) {
    std::cout << "[torch inference] threads: intra=" << at::get_num_threads()
              << " interop=" << at::get_num_interop_threads() << "\n";
  }
}

/// Flatten a TorchScript result into (name, tensor) pairs.
void collect_outputs(const torch::jit::IValue &value,
                     std::vector<std::string> &names,
                     std::vector<torch::Tensor> &tensors) {
  if (value.isTensor()) {
    names.push_back(std::string());
    tensors.push_back(value.toTensor());
    return;
  }
  if (value.isTuple()) {
    for (const auto &element : value.toTuple()->elements()) {
      collect_outputs(element, names, tensors);
    }
    return;
  }
  if (value.isTensorList()) {
    for (const auto &tensor : value.toTensorList()) {
      names.push_back(std::string());
      tensors.push_back(tensor);
    }
    return;
  }
  if (value.isList()) {
    for (const auto &element : value.toList()) {
      collect_outputs(element, names, tensors);
    }
    return;
  }
  if (value.isGenericDict()) {
    for (const auto &entry : value.toGenericDict()) {
      EMULATOR_INFER_REQUIRE(entry.value().isTensor(),
                             "The TorchScript module returned a dict entry "
                             "that is not a tensor.");
      names.push_back(entry.key().isString() ? entry.key().toStringRef()
                                             : std::string());
      tensors.push_back(entry.value().toTensor());
    }
    return;
  }
  EMULATOR_INFER_REQUIRE(false,
                         "The TorchScript module returned a "
                             << value.tagKind()
                             << ", which this backend cannot unpack. Return a "
                                "tensor, a tuple/list of tensors, or a "
                                "Dict[str, Tensor].");
}

} // namespace

// ===========================================================================
// Impl
// ===========================================================================

struct TorchBackend::Impl {
  torch::jit::script::Module module;
  torch::Device device{torch::kCPU};
  std::string method = "forward";
  bool loaded = false;

  /// Per-input conversion buffers, reused across steps.
  std::vector<Tensor> scratch;

  /**
   * @brief Torch tensor sharing (or converted from) our tensor's memory.
   *
   * `target` is the element type the model wants, from the declared spec; when
   * it matches the data we hand torch our own pointer with from_blob.
   */
  torch::Tensor make_tensor(const Tensor &tensor, DType target,
                            const std::string &spec_name, std::size_t index) {
    std::vector<std::int64_t> sizes = tensor.dims();
    if (sizes.empty()) {
      sizes.push_back(tensor.size());
    }

    const void *data = nullptr;
    if (tensor.dtype() == target) {
      data = tensor.cdata();
    } else {
      Tensor &buffer = scratch[index];
      buffer.set_name(spec_name.empty() ? tensor.name() : spec_name);
      buffer.resize(tensor.dims(), target);
      buffer.copy_from(tensor);
      data = buffer.cdata();
    }

    // from_blob does not take ownership and does not copy; the memory must
    // stay put for the duration of the call, which it does.
    auto options = torch::TensorOptions().dtype(to_torch_type(target));
    torch::Tensor result =
        torch::from_blob(const_cast<void *>(data), sizes, options);
    if (device.type() != torch::kCPU) {
      result = result.to(device); // an unavoidable host-to-device copy
    }
    return result;
  }
};

// ===========================================================================
// TorchBackend
// ===========================================================================

TorchBackend::TorchBackend(const InferenceConfig &config)
    : InferenceBackend(config), m_impl(new Impl()) {}

TorchBackend::~TorchBackend() {
  try {
    TorchBackend::finalize();
  } catch (const std::exception &e) {
    std::cerr << "[emulator::inference] warning: LibTorch teardown failed: "
              << e.what() << "\n";
  }
}

void TorchBackend::init_impl() {
  EMULATOR_INFER_REQUIRE(!m_config.model_path.empty(),
                         "The torch backend needs 'model_path' to point at a "
                         "TorchScript archive (torch.jit.script(...).save()).");

  const std::string device = normalize(m_config.get("device", "cpu"));
  if (device == "cuda" || device == "gpu") {
    EMULATOR_INFER_REQUIRE(torch::cuda::is_available(),
                           "device=cuda was requested but this LibTorch build "
                           "reports no CUDA device.");
    // No default ordinal: without one, every rank sharing a node would load
    // its own copy of the model onto device 0.  create_executor() fills this
    // in from the InferenceContext; a bare backend has to be told.
    EMULATOR_INFER_REQUIRE(
        m_config.has("device_id"),
        "device=" << m_config.get("device")
                  << " needs a 'device_id'. Build the backend through "
                     "create_executor() so the ordinal comes from the "
                     "InferenceContext, or set the option explicitly.");
    const int device_id = m_config.get_int("device_id");
    EMULATOR_INFER_REQUIRE(device_id >= 0 &&
                               device_id < torch::cuda::device_count(),
                           "device_id " << device_id << " is out of range; "
                                        << "this host reports "
                                        << torch::cuda::device_count()
                                        << " CUDA device(s).");
    m_impl->device =
        torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(device_id));
  } else {
    EMULATOR_INFER_REQUIRE(device == "cpu",
                           "Unknown device '" << device
                                              << "' for the torch backend. "
                                                 "Valid values: cpu, cuda.");
  }

  m_impl->method = m_config.get("method", "forward");

  // One thread per rank by default: see apply_thread_settings, and the same
  // MPI oversubscription argument as the onnx backend.  `auto` leaves the
  // process-wide setting alone.
  const std::string intra = normalize(m_config.get("intra_op_threads", "1"));
  const std::string inter = normalize(m_config.get("inter_op_threads", "1"));
  apply_thread_settings(intra == "auto" ? 0 : m_config.get_int(
                                                  "intra_op_threads", 1),
                        inter == "auto" ? 0 : m_config.get_int(
                                                  "inter_op_threads", 1),
                        m_config.verbose);

  try {
    m_impl->module = torch::jit::load(m_config.model_path, m_impl->device);
    m_impl->module.eval();
    m_impl->loaded = true;
  } catch (const c10::Error &e) {
    throw InferenceError("LibTorch failed to load '" + m_config.model_path +
                         "': " + e.what());
  }

  m_impl->scratch.clear();
  m_impl->scratch.resize(
      std::max<std::size_t>(m_config.inputs.size(), 16));

  if (m_config.verbose) {
    std::cout << "[torch inference] loaded " << m_config.model_path
              << " method=" << m_impl->method << " device=" << device << "\n";
  }
}

bool TorchBackend::infer_impl(const TensorMap &inputs, TensorMap &outputs) {
  // No autograd graph, no gradient bookkeeping: this is inference only.
  c10::InferenceMode inference_mode;

  const auto specs = m_config.inputs;
  const std::size_t n_args = specs.empty() ? inputs.size() : specs.size();
  if (m_impl->scratch.size() < n_args) {
    m_impl->scratch.resize(n_args);
  }

  std::vector<torch::jit::IValue> args;
  args.reserve(n_args);

  for (std::size_t i = 0; i < n_args; ++i) {
    const Tensor *tensor = nullptr;
    DType target = DType::FLOAT32;
    std::string spec_name;

    if (specs.empty()) {
      // Undeclared: pass the tensors through in the order given, unconverted.
      tensor = &inputs[i];
      target = tensor->dtype();
    } else {
      spec_name = specs[i].name;
      target = specs[i].dtype;
      tensor = inputs.find(spec_name);
      if (tensor == nullptr && inputs.size() == n_args) {
        tensor = &inputs[i]; // positional fallback
      }
      EMULATOR_INFER_REQUIRE(tensor != nullptr,
                             "TorchScript argument " << i << " ('" << spec_name
                                                     << "') is not among the "
                                                        "tensors provided: "
                                                     << inputs.names_string()
                                                     << ".");
    }

    args.push_back(m_impl->make_tensor(*tensor, target, spec_name, i));
  }

  torch::jit::IValue result;
  try {
    auto method = m_impl->module.get_method(m_impl->method);
    result = method(args);
  } catch (const c10::Error &e) {
    throw InferenceError("LibTorch failed while running " +
                         m_config.model_path + "." + m_impl->method + "(): " +
                         e.what());
  }

  std::vector<std::string> names;
  std::vector<torch::Tensor> results;
  collect_outputs(result, names, results);

  // Name whatever came back: dict keys if present, else the declared output
  // specs, else positional names.
  const auto out_specs = output_specs();
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (!names[i].empty()) {
      continue;
    }
    if (i < out_specs.size()) {
      names[i] = out_specs[i].name;
    } else {
      names[i] = "output_" + std::to_string(i);
    }
  }

  for (std::size_t i = 0; i < results.size(); ++i) {
    // Results may be non-contiguous views or live on a device; make them a
    // plain contiguous CPU block before copying out.
    const torch::Tensor cpu_result =
        results[i].to(torch::kCPU).contiguous();

    DType dtype = DType::FLOAT32;
    EMULATOR_INFER_REQUIRE(
        from_torch_type(cpu_result.scalar_type(), dtype),
        "TorchScript output '"
            << names[i] << "' has element type " << cpu_result.scalar_type()
            << ", which the emulator inference layer does not carry "
               "(supported: float32, float64, int32, int64).");

    std::vector<std::int64_t> dims(cpu_result.sizes().begin(),
                                   cpu_result.sizes().end());
    const Tensor source = Tensor::wrap(names[i], cpu_result.const_data_ptr(),
                                       dims, dtype);

    Tensor *dest = outputs.find(names[i]);
    if (dest == nullptr && outputs.size() == results.size()) {
      dest = &outputs[i]; // positional fallback
    }
    if (dest == nullptr) {
      dest = &outputs.add(Tensor(names[i], dims, dtype));
    }
    EMULATOR_INFER_REQUIRE(source.size() == dest->size(),
                           "TorchScript output '"
                               << names[i] << "' has " << source.size()
                               << " elements but the destination "
                               << dest->to_string() << " holds " << dest->size()
                               << ".");
    dest->copy_from(source);
  }

  return true;
}

void TorchBackend::final_impl() {
  m_impl->scratch.clear();
  if (m_impl->loaded) {
    // Replace the module with an empty one to drop the weights.
    m_impl->module = torch::jit::script::Module();
    m_impl->loaded = false;
  }
}

} // namespace inference
} // namespace emulator
