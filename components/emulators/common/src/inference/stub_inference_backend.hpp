/**
 * @file stub_inference_backend.hpp
 * @brief Dependency-free inference backend for testing the data pipeline.
 */

#ifndef E3SM_EMULATOR_STUB_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_STUB_INFERENCE_BACKEND_HPP

#include "inference_backend.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Synthetic backend that needs no ML dependencies.
 *
 * Exercises everything around the model — configuration, tensor plumbing,
 * coupling, restarts — without any runtime installed, which makes it the
 * backend to use in CI and when bringing up a new component.
 *
 * Options (`mode` selects what the outputs become):
 *  - `noop`     : outputs untouched (default; the historical behavior)
 *  - `zero`     : outputs zeroed
 *  - `constant` : outputs filled with option `value` (default 0)
 *  - `copy`     : output i receives input i, cast as needed; a size mismatch
 *                 is an error
 *  - `affine`   : output i receives `scale * input i + offset` (options
 *                 `scale`, default 1, and `offset`, default 0)
 *
 * @see InferenceBackend for the interface contract
 */
class StubBackend : public InferenceBackend {
public:
  /// @brief What the stub writes into the output tensors.
  enum class Mode { NOOP, ZERO, CONSTANT, COPY, AFFINE };

  explicit StubBackend(const InferenceConfig &config);
  ~StubBackend() override;

  /// @copydoc InferenceBackend::name
  std::string name() const override { return "Stub"; }

  /// @brief Mode parsed from the configuration.
  Mode mode() const { return m_mode; }

protected:
  /// @copydoc InferenceBackend::init_impl
  void init_impl() override;

  /// @copydoc InferenceBackend::infer_impl
  bool infer_impl(const TensorMap &inputs, TensorMap &outputs) override;

private:
  /// Ensure `outputs` has a destination for each declared output spec.
  void ensure_outputs(const TensorMap &inputs, TensorMap &outputs) const;

  Mode m_mode = Mode::NOOP;
  double m_value = 0.0;
  double m_scale = 1.0;
  double m_offset = 0.0;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_STUB_INFERENCE_BACKEND_HPP
