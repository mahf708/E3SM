/**
 * @file inference_backend.cpp
 * @brief Shared InferenceBackend behavior: lifecycle, validation, flat path.
 */

#include "inference_backend.hpp"

#include <iostream>

namespace emulator {
namespace inference {

namespace {

/// Elements per batch element implied by a spec (dynamic extents count as 1).
std::int64_t per_sample_size(const TensorSpec &spec) {
  std::int64_t n = 1;
  for (std::int64_t d : spec.dims) {
    if (d > 0) {
      n *= d;
    }
  }
  return n;
}

} // namespace

std::string backend_type_name(BackendType type) {
  switch (type) {
  case BackendType::STUB:
    return "stub";
  case BackendType::PYTHON:
    return "python";
  case BackendType::TORCH:
    return "torch";
  case BackendType::ONNX:
    return "onnx";
  }
  return "stub"; // unknown values fall back to the dependency-free backend
}

InferenceBackend::InferenceBackend(const InferenceConfig &config)
    : m_config(config) {}

InferenceBackend::~InferenceBackend() {
  // Derived destructors have already run, so only touch our own state; a
  // backend that needs teardown must call finalize() from its destructor.
  m_initialized = false;
}

void InferenceBackend::initialize() {
  if (m_initialized) {
    return;
  }
  try {
    init_impl();
  } catch (...) {
    // Release whatever the failed setup managed to acquire (an interpreter
    // customer, a partially built session) before the error propagates.
    final_impl();
    throw;
  }
  m_initialized = true;
  if (m_config.verbose) {
    std::cout << to_string() << std::flush;
  }
}

void InferenceBackend::finalize() {
  if (!m_initialized) {
    return;
  }
  final_impl();
  m_initialized = false;
}

bool InferenceBackend::infer(const TensorMap &inputs, TensorMap &outputs) {
  if (!m_initialized) {
    initialize();
  }
  validate_inputs(inputs);
  const bool ok = infer_impl(inputs, outputs);
  if (ok) {
    ++m_infer_count;
  }
  return ok;
}

bool InferenceBackend::infer(const double *inputs, double *outputs,
                             int batch_size) {
  EMULATOR_INFER_REQUIRE(batch_size > 0,
                         name() << ": batch_size must be positive (got "
                                << batch_size << ").");

  const auto in_specs = input_specs();
  const auto out_specs = output_specs();

  std::int64_t n_in = m_config.input_channels;
  std::int64_t n_out = m_config.output_channels;
  std::string in_name = "input";
  std::string out_name = "output";

  if (in_specs.size() == 1) {
    in_name = in_specs[0].name;
    if (n_in <= 0) {
      n_in = per_sample_size(in_specs[0]);
    }
  }
  if (out_specs.size() == 1) {
    out_name = out_specs[0].name;
    if (n_out <= 0) {
      n_out = per_sample_size(out_specs[0]);
    }
  }

  EMULATOR_INFER_REQUIRE(
      n_in > 0 && n_out > 0,
      name() << ": the flat-array infer() path needs to know how many values "
                "each column carries. Set input_channels/output_channels, or "
                "declare exactly one input and one output spec. Got "
             << in_specs.size() << " input spec(s) and " << out_specs.size()
             << " output spec(s).");
  EMULATOR_INFER_REQUIRE(inputs != nullptr && outputs != nullptr,
                         name() << ": null buffer passed to the flat-array "
                                   "infer() path.");

  // Views over the caller's memory: nothing is copied here.
  m_flat_inputs.clear();
  m_flat_outputs.clear();
  m_flat_inputs.wrap(in_name, inputs, {batch_size, n_in});
  m_flat_outputs.wrap(out_name, outputs, {batch_size, n_out});

  return infer(m_flat_inputs, m_flat_outputs);
}

std::vector<TensorSpec> InferenceBackend::input_specs() const {
  if (!m_config.inputs.empty()) {
    return m_config.inputs;
  }
  if (m_config.input_channels > 0) {
    return {TensorSpec("input", {-1, m_config.input_channels})};
  }
  return {};
}

std::vector<TensorSpec> InferenceBackend::output_specs() const {
  if (!m_config.outputs.empty()) {
    return m_config.outputs;
  }
  if (m_config.output_channels > 0) {
    return {TensorSpec("output", {-1, m_config.output_channels})};
  }
  return {};
}

TensorMap InferenceBackend::make_inputs(std::int64_t batch) const {
  return make_tensors(input_specs(), batch);
}

TensorMap InferenceBackend::make_outputs(std::int64_t batch) const {
  return make_tensors(output_specs(), batch);
}

void InferenceBackend::validate_inputs(const TensorMap &inputs) const {
  const auto specs = input_specs();
  for (const auto &spec : specs) {
    const Tensor *t = inputs.find(spec.name);
    if (t == nullptr) {
      // A single unnamed-by-convention input is matched positionally so that
      // callers using the flat path or ad-hoc names still work.
      if (inputs.size() == 1 && specs.size() == 1) {
        continue;
      }
      EMULATOR_INFER_REQUIRE(false,
                             name() << ": missing input tensor '" << spec.name
                                    << "'. Provided: " << inputs.names_string()
                                    << ".");
    }
    // A backend that converts precision only needs the shape to line up.
    TensorSpec effective = spec;
    if (converts_element_types()) {
      effective.dtype = t->dtype();
    }

    std::string why;
    EMULATOR_INFER_REQUIRE(spec_matches(effective, *t, &why),
                           name() << ": input '" << spec.name
                                  << "' does not match its spec ("
                                  << spec.to_string() << "): " << why << ".");
  }
}

std::string InferenceBackend::to_string() const {
  std::ostringstream oss;
  oss << "Inference backend '" << name() << "'\n";
  oss << "  initialized  : " << std::boolalpha << m_initialized << "\n";
  oss << "  infer calls  : " << m_infer_count << "\n";
  if (!m_config.model_path.empty()) {
    oss << "  model        : " << m_config.model_path << "\n";
  }

  const auto dump = [&oss](const char *label,
                           const std::vector<TensorSpec> &specs) {
    oss << "  " << label << " : ";
    if (specs.empty()) {
      oss << "<unspecified>";
    } else {
      for (std::size_t i = 0; i < specs.size(); ++i) {
        oss << (i ? ", " : "") << specs[i].to_string();
      }
    }
    oss << "\n";
  };
  dump("inputs      ", input_specs());
  dump("outputs     ", output_specs());
  return oss.str();
}

} // namespace inference
} // namespace emulator
