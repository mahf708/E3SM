/**
 * @file onnx_inference_backend.hpp
 * @brief Inference backend built on ONNX Runtime.
 */

#ifndef E3SM_EMULATOR_ONNX_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_ONNX_INFERENCE_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>

#include "inference_backend.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Runs an ONNX model with ONNX Runtime.
 *
 * The dependency-light production path: one shared library, no Python in the
 * process, and a model format most training frameworks can export.  The model
 * carries its own input/output names, shapes and element types, so this
 * backend does not need them configured — input_specs() and output_specs()
 * report what the model actually declares once initialize() has run, which
 * makes a shape or precision mismatch a clear error instead of garbage
 * results.
 *
 * Input tensors are handed to the runtime in place whenever their element
 * type matches the model's; when it does not (E3SM `real(r8)` fields into a
 * single-precision model, the usual case) the conversion goes through a
 * scratch buffer that is allocated once and reused.
 *
 * ## Options
 *  - `intra_op_threads`  threads within an operator (default: runtime default)
 *  - `inter_op_threads`  threads across operators
 *  - `optimization_level` `disable`, `basic`, `extended` or `all` (default)
 *  - `device`            `cpu` (default) or `cuda`
 *  - `device_id`         CUDA device ordinal (default 0)
 *  - `log_severity`      `verbose`, `info`, `warning` (default), `error`,
 *                        `fatal`
 *
 * `model_path` is required and must name a `.onnx` file.
 */
class OnnxBackend : public InferenceBackend {
public:
  explicit OnnxBackend(const InferenceConfig &config);
  ~OnnxBackend() override;

  /// @copydoc InferenceBackend::name
  std::string name() const override { return "ONNXRuntime"; }

  /// @brief Inputs the loaded model declares (config specs before init).
  std::vector<TensorSpec> input_specs() const override;

  /// @brief Outputs the loaded model declares (config specs before init).
  std::vector<TensorSpec> output_specs() const override;

  /// @brief True: r8 fields are converted through a reused scratch buffer.
  bool converts_element_types() const override { return true; }

protected:
  /// @copydoc InferenceBackend::init_impl
  void init_impl() override;

  /// @copydoc InferenceBackend::infer_impl
  bool infer_impl(const TensorMap &inputs, TensorMap &outputs) override;

  /// @copydoc InferenceBackend::final_impl
  void final_impl() override;

private:
  /// ONNX Runtime types are hidden so this header needs no ORT includes.
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_ONNX_INFERENCE_BACKEND_HPP
