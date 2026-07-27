/**
 * @file inference_executor.cpp
 * @brief Execution policies above the local inference engines.
 */

#include "inference_executor.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

#include "create_inference_backend.hpp"

namespace emulator {
namespace inference {

namespace {

std::string normalize(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '-') {
      out.push_back('_');
    } else if (!std::isspace(static_cast<unsigned char>(c))) {
      out.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return out;
}

/// What a not-yet-implemented policy would take to build.
const char *policy_gap(ExecutionPolicy policy) {
  switch (policy) {
  case ExecutionPolicy::GPU_GROUP:
    return "It needs a node/device sub-communicator, variable-size batch "
           "gather and scatter, and reusable communication buffers. Until "
           "then, give the component one rank per device and use "
           "local_replica.";
  case ExecutionPolicy::SPATIAL_DISTRIBUTED:
    return "It needs the model itself to perform collectives, which neither "
           "ONNX nor TorchScript can express portably; the first workable "
           "route is the python backend with a model that already uses "
           "torch.distributed. Until then, a global model can run on a "
           "component configured with few ranks, letting the coupler gather "
           "for it.";
  case ExecutionPolicy::LOCAL_REPLICA:
    return "";
  }
  return "";
}

/**
 * @brief Apply context-derived settings to a backend configuration.
 *
 * This is where knowing the parallel environment actually buys something:
 * device assignment stops being a guess, and only one rank narrates.
 */
InferenceConfig adapt_config(const InferenceConfig &config,
                             const InferenceContext &context) {
  InferenceConfig adapted = config;

  // Diagnostics: one voice per run unless asked otherwise.  A thousand ranks
  // each printing their model summary is not a log, it is a denial of service
  // against whoever has to read it.
  if (adapted.verbose && !context.is_root() &&
      !adapted.get_bool("verbose_all_ranks", false)) {
    adapted.verbose = false;
  }

  const std::string device = normalize(adapted.get("device", "cpu"));
  if (device == "cuda" || device == "gpu") {
    if (!adapted.has("device_id")) {
      if (context.device_id != k_no_device) {
        adapted.set("device_id", context.device_id);
      } else if (context.node_size <= 1) {
        adapted.set("device_id", 0); // one rank on the node: unambiguous
      } else {
        EMULATOR_INFER_REQUIRE(
            false,
            "device=" << adapted.get("device") << " with " << context.node_size
                      << " ranks of this component on the node, and no device "
                         "assignment. Every rank would use device 0, "
                         "multiplying model memory and serializing kernels. "
                         "Set the context's device_id (see "
                         "assign_device_round_robin), set the 'device_id' "
                         "option explicitly, or give the component one rank "
                         "per device.");
      }
    }
  }

  return adapted;
}

} // namespace

// ===========================================================================
// Policy names
// ===========================================================================

std::string execution_policy_name(ExecutionPolicy policy) {
  switch (policy) {
  case ExecutionPolicy::LOCAL_REPLICA:
    return "local_replica";
  case ExecutionPolicy::GPU_GROUP:
    return "gpu_group";
  case ExecutionPolicy::SPATIAL_DISTRIBUTED:
    return "spatial_distributed";
  }
  return "local_replica";
}

ExecutionPolicy execution_policy_from_string(const std::string &name) {
  const std::string key = normalize(name);
  if (key.empty() || key == "local_replica" || key == "local" ||
      key == "replica") {
    return ExecutionPolicy::LOCAL_REPLICA;
  }
  if (key == "gpu_group" || key == "device_group") {
    return ExecutionPolicy::GPU_GROUP;
  }
  if (key == "spatial_distributed" || key == "spatial" ||
      key == "distributed") {
    return ExecutionPolicy::SPATIAL_DISTRIBUTED;
  }
  EMULATOR_INFER_REQUIRE(false,
                         "Unknown execution policy '"
                             << name
                             << "'. Valid values: local_replica, gpu_group, "
                                "spatial_distributed.");
  return ExecutionPolicy::LOCAL_REPLICA; // unreachable
}

// ===========================================================================
// InferenceExecutor
// ===========================================================================

InferenceExecutor::InferenceExecutor(std::shared_ptr<InferenceBackend> backend,
                                     InferenceContext context)
    : m_backend(std::move(backend)), m_context(context) {
  EMULATOR_INFER_REQUIRE(m_backend != nullptr,
                         "An inference executor needs a backend.");
}

void InferenceExecutor::init_impl() { m_backend->initialize(); }

void InferenceExecutor::final_impl() { m_backend->finalize(); }

void InferenceExecutor::initialize() {
  if (m_initialized) {
    return;
  }
  init_impl();
  m_initialized = true;
}

void InferenceExecutor::finalize() {
  if (!m_initialized) {
    return;
  }
  final_impl();
  m_initialized = false;
}

bool InferenceExecutor::infer(const TensorMap &inputs, TensorMap &outputs) {
  if (!m_initialized) {
    initialize();
  }
  return infer_impl(inputs, outputs);
}

std::string InferenceExecutor::to_string() const {
  std::ostringstream oss;
  oss << "Inference executor '" << execution_policy_name(policy()) << "'\n";
  oss << "  context      : " << m_context.to_string() << "\n";
  oss << "  owns model   : " << std::boolalpha << owns_model() << "\n";
  oss << m_backend->to_string();
  return oss.str();
}

// ===========================================================================
// LocalReplicaExecutor
// ===========================================================================

LocalReplicaExecutor::LocalReplicaExecutor(
    std::shared_ptr<InferenceBackend> backend, InferenceContext context)
    : InferenceExecutor(std::move(backend), context) {}

bool LocalReplicaExecutor::infer_impl(const TensorMap &inputs,
                                      TensorMap &outputs) {
  // Purely local: this rank's columns, this rank's replica, no communication.
  return m_backend->infer(inputs, outputs);
}

// ===========================================================================
// Factory
// ===========================================================================

std::unique_ptr<InferenceExecutor>
create_executor(const InferenceConfig &config,
                const InferenceContext &context) {
  const ExecutionPolicy policy =
      execution_policy_from_string(config.get("execution_policy"));

  EMULATOR_INFER_REQUIRE(policy == ExecutionPolicy::LOCAL_REPLICA,
                         "Execution policy '"
                             << execution_policy_name(policy)
                             << "' is not implemented yet. "
                             << policy_gap(policy));

  auto backend = create_backend(adapt_config(config, context));
  return std::unique_ptr<InferenceExecutor>(
      new LocalReplicaExecutor(std::move(backend), context));
}

std::unique_ptr<InferenceExecutor>
create_and_init_executor(const InferenceConfig &config,
                         const InferenceContext &context) {
  auto executor = create_executor(config, context);
  executor->initialize();
  return executor;
}

} // namespace inference
} // namespace emulator
