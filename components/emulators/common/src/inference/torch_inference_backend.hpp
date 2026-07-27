/**
 * @file torch_inference_backend.hpp
 * @brief Inference backend built on LibTorch (TorchScript).
 */

#ifndef E3SM_EMULATOR_TORCH_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_TORCH_INFERENCE_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>

#include "inference_backend.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Runs a TorchScript module with LibTorch.
 *
 * The path with the least distance from PyTorch training: a model saved with
 * `torch.jit.script(model).save(...)` or `torch.jit.trace(...)` runs here with
 * no Python in the process, keeping custom layers and control flow that an
 * ONNX export may not survive.  The cost is a heavy dependency (LibTorch is
 * hundreds of MB) and a version-sensitive build.
 *
 * Unlike ONNX, a TorchScript module carries no reliable description of its
 * inputs: `forward` takes positional arguments whose shapes and precision are
 * not declared.  Inputs are therefore passed **in the order of the declared
 * `input:` specs** (falling back to the order of the tensors handed to
 * infer()), and a declared spec's element type is what the argument is
 * converted to.  Declaring specs is optional but strongly recommended: it is
 * the only place the model's expected precision can be written down.
 *
 * Tensors whose element type already matches are passed to the model in place;
 * others are converted through a scratch buffer allocated once and reused.
 *
 * Results are accepted as a single tensor, a tuple/list of tensors, or a
 * `Dict[str, Tensor]`.  Names come from the dict when there is one, otherwise
 * from the declared `output:` specs, otherwise `output_0`, `output_1`, ...
 *
 * ## Options
 *  - `device`         `cpu` (default) or `cuda`
 *  - `device_id`      CUDA device ordinal; required for `cuda`, and normally
 *                     filled in from the InferenceContext by create_executor()
 *  - `method`         module method to call (default `forward`)
 *  - `intra_op_threads` `at::set_num_threads` (default 1; `auto` to leave it)
 *  - `inter_op_threads` `at::set_num_interop_threads` (default 1)
 *
 * LibTorch thread counts are process-wide, not per module: the first backend
 * to set them wins, and a later instance asking for something different is
 * warned about rather than obeyed (asking twice would otherwise throw).
 *
 * `model_path` is required and must name a TorchScript archive.
 */
class TorchBackend : public InferenceBackend {
public:
  explicit TorchBackend(const InferenceConfig &config);
  ~TorchBackend() override;

  /// @copydoc InferenceBackend::name
  std::string name() const override { return "LibTorch"; }

  /// @brief True: declared precision is applied via a reused scratch buffer.
  bool converts_element_types() const override { return true; }

protected:
  /// @copydoc InferenceBackend::init_impl
  void init_impl() override;

  /// @copydoc InferenceBackend::infer_impl
  bool infer_impl(const TensorMap &inputs, TensorMap &outputs) override;

  /// @copydoc InferenceBackend::final_impl
  void final_impl() override;

private:
  /// Torch types are hidden so this header needs no LibTorch includes.
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_TORCH_INFERENCE_BACKEND_HPP
