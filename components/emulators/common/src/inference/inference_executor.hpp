/**
 * @file inference_executor.hpp
 * @brief How a model is executed across ranks, above the local engines.
 */

#ifndef E3SM_EMULATOR_INFERENCE_EXECUTOR_HPP
#define E3SM_EMULATOR_INFERENCE_EXECUTOR_HPP

#include <memory>
#include <string>

#include "inference_backend.hpp"
#include "inference_context.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Available execution policies.
 *
 * An InferenceBackend is a **local** model engine: it loads a model in one
 * process and evaluates it on the tensors that process hands it.  How that
 * relates to the other ranks of a component is a separate question, and this
 * is where it is answered.
 *
 * The distinction is not cosmetic.  A local `InferenceBackend::infer()` is an
 * ordinary function call; a distributed `InferenceExecutor::infer()` is a
 * **collective** — every rank of the context's communicator must call it, the
 * same number of times, in the same order, or the run hangs.  Two different
 * contracts deserve two different types.
 */
enum class ExecutionPolicy {
  /**
   * Every rank owns a model replica and infers on its own local columns.
   * No communication.  Correct only for models whose output for a column
   * depends on that column alone (`y_i = f(x_i)`): pointwise
   * parameterizations, column MLPs, per-column vertical networks.
   */
  LOCAL_REPLICA,

  /**
   * Ranks sharing an accelerator gather their local batches onto one owner
   * rank, infer once, and scatter the results back.  For layouts with
   * several MPI ranks per GPU, where per-rank batches are too small to be
   * worth a kernel launch.  Not implemented yet.
   */
  GPU_GROUP,

  /**
   * Every rank evaluates part of one global model, with collectives inside
   * the model.  For models with a global receptive field (spherical
   * transforms, global attention, graph networks over the whole grid).
   * Not implemented yet; realistically reachable first through the Python
   * backend with a model that already speaks torch.distributed.
   */
  SPATIAL_DISTRIBUTED
};

/// @brief Policy name as it appears in configuration.
std::string execution_policy_name(ExecutionPolicy policy);

/// @brief Parse a policy name; throws InferenceError on an unknown name.
ExecutionPolicy execution_policy_from_string(const std::string &name);

/**
 * @brief Runs a model across the ranks of a component.
 *
 * Owns a local InferenceBackend and an InferenceContext, and decides what the
 * other ranks have to do with it.
 *
 * ## Collective contract
 * initialize(), infer() and finalize() are collective over
 * `context().comm` for every policy except LOCAL_REPLICA (which is purely
 * local and may be called by any subset of ranks).  Treat them as collective
 * unconditionally unless you know the policy: that is the safe habit, and it
 * is what lets a run switch policy from a namelist without changing code.
 */
class InferenceExecutor {
public:
  InferenceExecutor(std::shared_ptr<InferenceBackend> backend,
                    InferenceContext context);
  virtual ~InferenceExecutor() = default;

  InferenceExecutor(const InferenceExecutor &) = delete;
  InferenceExecutor &operator=(const InferenceExecutor &) = delete;

  /// @brief Which policy this executor implements.
  virtual ExecutionPolicy policy() const = 0;

  /**
   * @brief Bring the model up.  Collective (see the class contract).
   * @throws InferenceError if the backend cannot be initialized
   */
  void initialize();

  /**
   * @brief Evaluate the model.  Collective (see the class contract).
   *
   * Shapes are **local** for LOCAL_REPLICA: `inputs` holds this rank's
   * columns and `outputs` receives this rank's results.
   *
   * @return true on success
   */
  bool infer(const TensorMap &inputs, TensorMap &outputs);

  /// @brief Release the model.  Collective (see the class contract).
  void finalize();

  bool is_initialized() const { return m_initialized; }

  /**
   * @brief True if this rank holds a model replica.
   *
   * Always true for LOCAL_REPLICA; false on the non-owner ranks of a policy
   * that concentrates the model on fewer ranks.
   */
  virtual bool owns_model() const { return true; }

  /// @brief The local engine.  Prefer infer() unless you mean "this rank only".
  InferenceBackend &backend() { return *m_backend; }
  const InferenceBackend &backend() const { return *m_backend; }

  const InferenceContext &context() const { return m_context; }

  /// @brief Multi-line description of policy, context and backend.
  std::string to_string() const;

protected:
  /// @brief Policy-specific evaluation; called with the executor initialized.
  virtual bool infer_impl(const TensorMap &inputs, TensorMap &outputs) = 0;

  /// @brief Policy-specific setup.  Default: initialize the local backend.
  virtual void init_impl();

  /// @brief Policy-specific teardown.  Default: finalize the local backend.
  virtual void final_impl();

  std::shared_ptr<InferenceBackend> m_backend;
  InferenceContext m_context;
  bool m_initialized = false;
};

/**
 * @brief One model replica per rank, no communication.
 *
 * The default, and the only policy implemented today.  `infer()` forwards
 * straight to the local backend, so it costs nothing over calling the backend
 * directly — the point of routing through an executor is that the call site
 * does not change when the policy does.
 */
class LocalReplicaExecutor : public InferenceExecutor {
public:
  LocalReplicaExecutor(std::shared_ptr<InferenceBackend> backend,
                       InferenceContext context);

  ExecutionPolicy policy() const override {
    return ExecutionPolicy::LOCAL_REPLICA;
  }

protected:
  bool infer_impl(const TensorMap &inputs, TensorMap &outputs) override;
};

/**
 * @brief Build an executor: backend plus execution policy, from configuration.
 *
 * Reads `execution_policy` from the configuration (default `local_replica`)
 * and adapts the backend configuration to the context before the backend is
 * created:
 *  - an accelerator ordinal is taken from the context when the configuration
 *    does not name one, and running several ranks per device without saying
 *    so is refused rather than silently piling them onto device 0;
 *  - diagnostics are limited to rank 0 unless `verbose_all_ranks` is set.
 *
 * @throws InferenceError for an unknown or not-yet-implemented policy, with a
 *         message saying what the policy would need
 */
std::unique_ptr<InferenceExecutor>
create_executor(const InferenceConfig &config, const InferenceContext &context);

/// @brief create_executor() followed by initialize().
std::unique_ptr<InferenceExecutor>
create_and_init_executor(const InferenceConfig &config,
                         const InferenceContext &context);

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_EXECUTOR_HPP
