/**
 * @file inference_backend.hpp
 * @brief Abstract interface for neural network inference backends.
 */

#ifndef E3SM_EMULATOR_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_INFERENCE_BACKEND_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "inference_config.hpp"
#include "tensor.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Enumeration of built-in inference backend types.
 *
 * Kept for source compatibility and for Fortran/C callers that prefer an
 * integer.  The string keys used by BackendRegistry are the primary way to
 * name a backend, since out-of-tree backends cannot extend this enum.
 */
enum class BackendType {
  STUB = 0,   ///< No-op / synthetic backend for testing (no ML dependencies)
  PYTHON = 1, ///< Embedded Python interpreter bridge
  TORCH = 2,  ///< LibTorch (TorchScript)
  ONNX = 3    ///< ONNX Runtime
};

/// @brief Registry key for a BackendType ("stub", "python", "torch", "onnx").
std::string backend_type_name(BackendType type);

/**
 * @brief Abstract interface for inference backends.
 *
 * A backend maps a set of named input tensors to a set of named output
 * tensors.  Tensors may view memory owned by the caller, so a component can
 * pass its own field or coupling buffers straight through with no copy.
 *
 * ## Lifecycle
 * 1. Construction takes the full InferenceConfig (cheap; no model loading).
 * 2. initialize() loads the model / starts the interpreter.  It is called
 *    automatically by the first infer() if the caller does not call it.
 * 3. infer() may be called any number of times.
 * 4. finalize() releases resources.  It is idempotent and is also called
 *    from the destructor.
 *
 * ## Error handling
 * Configuration and setup problems throw InferenceError, which carries a
 * message naming the offending option/tensor.  infer() returns false for a
 * failure a caller may plausibly want to handle itself.
 *
 * ## Threading
 * A backend instance is not thread safe.  Use one instance per thread (or
 * per MPI rank, which is the expected E3SM usage).
 */
class InferenceBackend {
public:
  explicit InferenceBackend(const InferenceConfig &config);
  virtual ~InferenceBackend();

  InferenceBackend(const InferenceBackend &) = delete;
  InferenceBackend &operator=(const InferenceBackend &) = delete;

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * @brief Load the model and prepare for inference.
   *
   * Safe to call more than once; subsequent calls are no-ops.
   * @throws InferenceError if the backend cannot be brought up
   */
  void initialize();

  bool is_initialized() const { return m_initialized; }

  /**
   * @brief Run inference on named tensors.
   *
   * @param inputs  Input tensors; owning or views over caller memory
   * @param outputs Output tensors.  Backends that know their output shapes
   *                will populate an empty map with owning tensors; otherwise
   *                the caller supplies destinations (typically views over
   *                its own fields).
   * @return true on success
   * @throws InferenceError on shape/type/name errors
   */
  bool infer(const TensorMap &inputs, TensorMap &outputs);

  /**
   * @brief Flat-array convenience path.
   *
   * Treats the caller's arrays as `[batch_size, input_channels]` and
   * `[batch_size, output_channels]` FLOAT64 blocks and wraps them without
   * copying.  Requires either declared single input/output specs or non-zero
   * `input_channels`/`output_channels` in the configuration.
   *
   * @param inputs     Input data, batch_size * input_channels doubles
   * @param outputs    Output data, batch_size * output_channels doubles
   * @param batch_size Number of samples (columns) in this call
   * @return true on success
   */
  bool infer(const double *inputs, double *outputs, int batch_size = 1);

  /**
   * @brief Release model/runtime resources.  Idempotent.
   */
  void finalize();

  // -------------------------------------------------------------------------
  // Introspection
  // -------------------------------------------------------------------------

  /// @brief Human-readable backend name (e.g. "Stub", "ONNXRuntime").
  virtual std::string name() const = 0;

  /**
   * @brief Declared input tensors.
   *
   * Defaults to the configured specs (or a single `input[-1,input_channels]`
   * spec derived from the channel counts).  Backends that can interrogate a
   * model override this with what the model actually expects.
   */
  virtual std::vector<TensorSpec> input_specs() const;

  /// @brief Declared output tensors.  @see input_specs()
  virtual std::vector<TensorSpec> output_specs() const;

  /**
   * @brief Whether this backend converts element types for the caller.
   *
   * Backends that stage data through their own buffers (ONNX Runtime,
   * LibTorch) accept `real(r8)` fields for a single-precision model and
   * convert on the way in and out, so a dtype difference from the declared
   * spec is not an error for them.  Backends that hand the caller's memory
   * straight to the model (the Python bridge) require an exact match.
   */
  virtual bool converts_element_types() const { return false; }

  /// @brief Configuration this backend was constructed with.
  const InferenceConfig &config() const { return m_config; }

  /// @brief Number of successful infer() calls so far.
  std::int64_t infer_count() const { return m_infer_count; }

  /// @brief Allocate a TensorMap matching input_specs() for `batch` columns.
  TensorMap make_inputs(std::int64_t batch = 1) const;

  /// @brief Allocate a TensorMap matching output_specs() for `batch` columns.
  TensorMap make_outputs(std::int64_t batch = 1) const;

  /// @brief Multi-line description of the backend and its tensors.
  std::string to_string() const;

protected:
  /// @brief Backend-specific setup (load model, start interpreter, ...).
  virtual void init_impl() = 0;

  /// @brief Backend-specific inference.  Called with m_initialized true.
  virtual bool infer_impl(const TensorMap &inputs, TensorMap &outputs) = 0;

  /// @brief Backend-specific teardown.  Default: nothing to do.
  virtual void final_impl() {}

  /**
   * @brief Check `inputs` against input_specs() and report a clear error.
   *
   * Called by infer() before infer_impl().  Only specs with a matching name
   * are enforced, so extra tensors in the map are tolerated (a component may
   * pass a superset of what a given model consumes).
   */
  void validate_inputs(const TensorMap &inputs) const;

  InferenceConfig m_config;       ///< Backend configuration
  bool m_initialized = false;     ///< Set by initialize()
  std::int64_t m_infer_count = 0; ///< Successful infer() calls

private:
  /// Scratch tensor maps reused by the flat-array infer() overload.
  TensorMap m_flat_inputs;
  TensorMap m_flat_outputs;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_BACKEND_HPP
