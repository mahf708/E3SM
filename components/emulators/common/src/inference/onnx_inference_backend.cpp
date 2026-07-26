/**
 * @file onnx_inference_backend.cpp
 * @brief ONNX Runtime inference backend implementation.
 */

#include "onnx_inference_backend.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

#include <onnxruntime_cxx_api.h>

namespace emulator {
namespace inference {

namespace {

/// Map an ONNX element type onto ours; returns false for types we do not carry.
bool from_onnx_type(ONNXTensorElementDataType onnx_type, DType &out) {
  switch (onnx_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    out = DType::FLOAT32;
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    out = DType::FLOAT64;
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    out = DType::INT32;
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    out = DType::INT64;
    return true;
  default:
    return false;
  }
}

ONNXTensorElementDataType to_onnx_type(DType dtype) {
  switch (dtype) {
  case DType::FLOAT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case DType::FLOAT64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  case DType::INT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  case DType::INT64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

GraphOptimizationLevel parse_optimization_level(const std::string &raw) {
  if (raw == "disable" || raw == "none") {
    return GraphOptimizationLevel::ORT_DISABLE_ALL;
  }
  if (raw == "basic") {
    return GraphOptimizationLevel::ORT_ENABLE_BASIC;
  }
  if (raw == "extended") {
    return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
  }
  if (raw == "all" || raw.empty()) {
    return GraphOptimizationLevel::ORT_ENABLE_ALL;
  }
  EMULATOR_INFER_REQUIRE(false, "Unknown optimization_level '"
                                    << raw
                                    << "'. Valid values: disable, basic, "
                                       "extended, all.");
  return GraphOptimizationLevel::ORT_ENABLE_ALL; // unreachable
}

OrtLoggingLevel parse_log_severity(const std::string &raw) {
  if (raw == "verbose") {
    return ORT_LOGGING_LEVEL_VERBOSE;
  }
  if (raw == "info") {
    return ORT_LOGGING_LEVEL_INFO;
  }
  if (raw == "warning" || raw.empty()) {
    return ORT_LOGGING_LEVEL_WARNING;
  }
  if (raw == "error") {
    return ORT_LOGGING_LEVEL_ERROR;
  }
  if (raw == "fatal") {
    return ORT_LOGGING_LEVEL_FATAL;
  }
  EMULATOR_INFER_REQUIRE(false, "Unknown log_severity '"
                                    << raw
                                    << "'. Valid values: verbose, info, "
                                       "warning, error, fatal.");
  return ORT_LOGGING_LEVEL_WARNING; // unreachable
}

} // namespace

// ===========================================================================
// Impl
// ===========================================================================

struct OnnxBackend::Impl {
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info{nullptr};

  std::vector<TensorSpec> input_specs;
  std::vector<TensorSpec> output_specs;
  std::vector<const char *> input_names;  ///< Points into input_specs names
  std::vector<const char *> output_names; ///< Points into output_specs names

  /// Per-input conversion buffers, reused across steps.
  std::vector<Tensor> scratch;

  /// Read the model's declared inputs and outputs.
  void read_signature() {
    Ort::AllocatorWithDefaultOptions allocator;

    const std::size_t n_in = session->GetInputCount();
    input_specs.clear();
    input_specs.reserve(n_in);
    for (std::size_t i = 0; i < n_in; ++i) {
      auto name = session->GetInputNameAllocated(i, allocator);
      // The type info must outlive the shape info, which only borrows from it.
      Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      DType dtype = DType::FLOAT32;
      EMULATOR_INFER_REQUIRE(from_onnx_type(info.GetElementType(), dtype),
                             "Model input '"
                                 << name.get() << "' has ONNX element type "
                                 << static_cast<int>(info.GetElementType())
                                 << ", which the emulator inference layer does "
                                    "not carry (supported: float32, float64, "
                                    "int32, int64).");
      input_specs.push_back(TensorSpec(name.get(), info.GetShape(), dtype));
    }

    const std::size_t n_out = session->GetOutputCount();
    output_specs.clear();
    output_specs.reserve(n_out);
    for (std::size_t i = 0; i < n_out; ++i) {
      auto name = session->GetOutputNameAllocated(i, allocator);
      Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      DType dtype = DType::FLOAT32;
      EMULATOR_INFER_REQUIRE(from_onnx_type(info.GetElementType(), dtype),
                             "Model output '"
                                 << name.get() << "' has ONNX element type "
                                 << static_cast<int>(info.GetElementType())
                                 << ", which the emulator inference layer does "
                                    "not carry.");
      output_specs.push_back(TensorSpec(name.get(), info.GetShape(), dtype));
    }

    // Cache the raw name pointers Run() wants.  Safe because the spec vectors
    // are not modified after this point.
    input_names.clear();
    for (const auto &spec : input_specs) {
      input_names.push_back(spec.name.c_str());
    }
    output_names.clear();
    for (const auto &spec : output_specs) {
      output_names.push_back(spec.name.c_str());
    }

    scratch.clear();
    scratch.resize(input_specs.size());
  }

  /// Wrap a tensor as an ORT value, converting into scratch if needed.
  Ort::Value make_value(const Tensor &tensor, const TensorSpec &spec,
                        std::size_t index) {
    const void *data = nullptr;
    std::vector<std::int64_t> dims = tensor.dims();

    if (tensor.dtype() == spec.dtype) {
      data = tensor.cdata();
    } else {
      // One-time allocation, then reused: converting r8 fields for a single
      // precision model must not malloc every step.
      Tensor &buffer = scratch[index];
      buffer.set_name(spec.name);
      buffer.resize(dims, spec.dtype);
      buffer.copy_from(tensor);
      data = buffer.cdata();
    }

    // ORT does not modify input tensors; the const_cast is the C API's shape.
    const std::size_t nbytes =
        static_cast<std::size_t>(tensor.size()) * dtype_size(spec.dtype);
    return Ort::Value::CreateTensor(memory_info, const_cast<void *>(data),
                                    nbytes, dims.data(), dims.size(),
                                    to_onnx_type(spec.dtype));
  }
};

// ===========================================================================
// OnnxBackend
// ===========================================================================

OnnxBackend::OnnxBackend(const InferenceConfig &config)
    : InferenceBackend(config), m_impl(new Impl()) {}

OnnxBackend::~OnnxBackend() {
  try {
    OnnxBackend::finalize();
  } catch (const std::exception &e) {
    std::cerr << "[emulator::inference] warning: ONNX Runtime teardown "
              << "failed: " << e.what() << "\n";
  }
}

std::vector<TensorSpec> OnnxBackend::input_specs() const {
  if (!m_impl->input_specs.empty()) {
    return m_impl->input_specs; // what the model says, once loaded
  }
  return InferenceBackend::input_specs();
}

std::vector<TensorSpec> OnnxBackend::output_specs() const {
  if (!m_impl->output_specs.empty()) {
    return m_impl->output_specs;
  }
  return InferenceBackend::output_specs();
}

void OnnxBackend::init_impl() {
  EMULATOR_INFER_REQUIRE(!m_config.model_path.empty(),
                         "The onnx backend needs 'model_path' to point at a "
                         ".onnx file.");

  try {
    m_impl->env = std::make_unique<Ort::Env>(
        parse_log_severity(m_config.get("log_severity")), "e3sm_emulator");

    Ort::SessionOptions options;
    if (m_config.has("intra_op_threads")) {
      options.SetIntraOpNumThreads(m_config.get_int("intra_op_threads"));
    }
    if (m_config.has("inter_op_threads")) {
      options.SetInterOpNumThreads(m_config.get_int("inter_op_threads"));
    }
    options.SetGraphOptimizationLevel(
        parse_optimization_level(m_config.get("optimization_level")));

    const std::string device = m_config.get("device", "cpu");
    if (device == "cuda" || device == "gpu") {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = m_config.get_int("device_id", 0);
      // Throws if this build of ONNX Runtime has no CUDA provider, which is a
      // clearer failure than silently running on the CPU.
      options.AppendExecutionProvider_CUDA(cuda_options);
    } else {
      EMULATOR_INFER_REQUIRE(device == "cpu",
                             "Unknown device '" << device
                                                << "' for the onnx backend. "
                                                   "Valid values: cpu, cuda.");
    }

    m_impl->memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    m_impl->session = std::make_unique<Ort::Session>(
        *m_impl->env, m_config.model_path.c_str(), options);

    m_impl->read_signature();
  } catch (const Ort::Exception &e) {
    throw InferenceError("ONNX Runtime failed to load '" + m_config.model_path +
                         "': " + e.what());
  }

  // If the configuration also declares specs, hold the model to them: a
  // silent mismatch between namelist and model is a debugging nightmare.
  const auto check = [this](const std::vector<TensorSpec> &declared,
                            const std::vector<TensorSpec> &actual,
                            const char *what) {
    for (const auto &spec : declared) {
      const auto it = std::find_if(
          actual.begin(), actual.end(),
          [&spec](const TensorSpec &s) { return s.name == spec.name; });
      EMULATOR_INFER_REQUIRE(it != actual.end(),
                             "Configured " << what << " '" << spec.name
                                           << "' is not " << what << " of "
                                           << m_config.model_path << ".");
      EMULATOR_INFER_REQUIRE(it->dtype == spec.dtype,
                             "Configured " << what << " '" << spec.name
                                           << "' is " << dtype_name(spec.dtype)
                                           << " but the model declares "
                                           << dtype_name(it->dtype) << ".");
    }
  };
  check(m_config.inputs, m_impl->input_specs, "input");
  check(m_config.outputs, m_impl->output_specs, "output");

  if (m_config.verbose) {
    std::cout << "[onnx inference] loaded " << m_config.model_path << " with "
              << m_impl->input_specs.size() << " input(s) and "
              << m_impl->output_specs.size() << " output(s)\n";
  }
}

bool OnnxBackend::infer_impl(const TensorMap &inputs, TensorMap &outputs) {
  const auto &in_specs = m_impl->input_specs;
  const auto &out_specs = m_impl->output_specs;

  std::vector<Ort::Value> in_values;
  in_values.reserve(in_specs.size());

  for (std::size_t i = 0; i < in_specs.size(); ++i) {
    const TensorSpec &spec = in_specs[i];
    const Tensor *tensor = inputs.find(spec.name);
    if (tensor == nullptr && inputs.size() == in_specs.size()) {
      // Positional fallback: a component whose field names do not match the
      // model's still works as long as the order does.
      tensor = &inputs[i];
    }
    EMULATOR_INFER_REQUIRE(tensor != nullptr,
                           "Model " << m_config.model_path << " needs input '"
                                    << spec.name << "' (expected "
                                    << spec.to_string()
                                    << "), which is not among the tensors "
                                       "provided: "
                                    << inputs.names_string() << ".");
    EMULATOR_INFER_REQUIRE(
        static_cast<int>(spec.dims.size()) == tensor->rank(),
        "Model input '" << spec.name << "' expects rank " << spec.dims.size()
                        << " (" << spec.to_string() << ") but got "
                        << tensor->to_string() << ".");
    for (std::size_t d = 0; d < spec.dims.size(); ++d) {
      EMULATOR_INFER_REQUIRE(spec.dims[d] < 0 ||
                                 spec.dims[d] == tensor->dims()[d],
                             "Model input '"
                                 << spec.name << "' expects "
                                 << spec.to_string() << " but got "
                                 << tensor->to_string() << ".");
    }

    try {
      in_values.push_back(m_impl->make_value(*tensor, spec, i));
    } catch (const Ort::Exception &e) {
      throw InferenceError("ONNX Runtime rejected input '" + spec.name +
                           "': " + e.what());
    }
  }

  std::vector<Ort::Value> out_values;
  try {
    out_values = m_impl->session->Run(
        Ort::RunOptions{nullptr}, m_impl->input_names.data(), in_values.data(),
        in_values.size(), m_impl->output_names.data(),
        m_impl->output_names.size());
  } catch (const Ort::Exception &e) {
    throw InferenceError("ONNX Runtime failed while running " +
                         m_config.model_path + ": " + e.what());
  }

  EMULATOR_INFER_REQUIRE(out_values.size() == out_specs.size(),
                         "ONNX Runtime returned " << out_values.size()
                                                  << " outputs, expected "
                                                  << out_specs.size() << ".");

  for (std::size_t i = 0; i < out_specs.size(); ++i) {
    const TensorSpec &spec = out_specs[i];
    const auto info = out_values[i].GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> dims = info.GetShape();

    Tensor *dest = outputs.find(spec.name);
    if (dest == nullptr && outputs.size() == out_specs.size()) {
      dest = &outputs[i]; // positional fallback, as for inputs
    }
    if (dest == nullptr) {
      // The caller wants everything the model produces: allocate it here,
      // using the shape the runtime actually returned.
      dest = &outputs.add(Tensor(spec.name, dims, spec.dtype));
    }

    // The runtime owns the result buffer; wrap it and let Tensor::copy_from
    // handle any precision change on the way into the caller's memory.
    const Tensor source =
        Tensor::wrap(spec.name, out_values[i].GetTensorRawData(), dims,
                     spec.dtype);
    EMULATOR_INFER_REQUIRE(source.size() == dest->size(),
                           "Model output '"
                               << spec.name << "' has " << source.size()
                               << " elements but the destination "
                               << dest->to_string() << " holds "
                               << dest->size() << ".");
    dest->copy_from(source);
  }

  return true;
}

void OnnxBackend::final_impl() {
  m_impl->scratch.clear();
  m_impl->input_names.clear();
  m_impl->output_names.clear();
  m_impl->input_specs.clear();
  m_impl->output_specs.clear();
  m_impl->session.reset();
  m_impl->env.reset();
}

} // namespace inference
} // namespace emulator
