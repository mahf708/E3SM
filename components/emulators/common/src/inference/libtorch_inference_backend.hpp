/**
 * @file libtorch_inference_backend.hpp
 * @brief Inference backend that runs a TorchScript module in this process.
 */

#ifndef E3SM_EMULATOR_LIBTORCH_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_LIBTORCH_INFERENCE_BACKEND_HPP

#include <memory>
#include <string>

#include "inference_backend.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Runs a traced TorchScript module through the libtorch C++ API.
 *
 * `model_path` must be a TorchScript archive written by `torch.jit.save()`.
 * No Python interpreter is involved at run time. A TorchScript module is a
 * pure function of its inputs, so unlike a Python-backed model it cannot
 * contain an MPI collective.
 *
 * Options, all read from `config.options`:
 *   - `device`      `cpu` (default), `cuda`, or `cuda:N`.  Asking for CUDA
 *                   on a build or machine without it is fatal; see below.
 *   - `dtype`       `float32` (default) or `float64`: the precision the
 *                   *module's* parameters were saved in.
 *   - `num_threads` intra-op thread count; 0 (default) leaves torch alone.
 *   - `jit_optimize` let TorchScript re-optimize the graph after its first
 *                   calls (default false).  Faster when warm, but the
 *                   optimized graph's float32 results differ from the first
 *                   calls', so a restarted run does not reproduce the
 *                   continuous one.  Process-wide.
 *   - `seed`        for a stochastic model: reseed torch's generators before
 *                   every forward pass from (seed, step), where the step is
 *                   what the component passed to set_step().  Unset (the
 *                   default) leaves the generators alone.
 *
 * Shapes are the caller's business, not this backend's: each input tensor
 * is handed to `forward` with exactly the shape its Tensor declares, in
 * TensorMap order, and only output element counts are checked.  A mismatch
 * throws with both shapes in the message rather than reshaping to fit,
 * since a silently transposed field produces no NaN and no complaint.
 *
 * Framework tensors are always `double`, so a float32 module incurs a
 * convert-in / convert-out per step; that cost is unavoidable at this
 * boundary. This backend uses only the stable `torch::jit` surface, but
 * the archive and the libtorch runtime still must agree on bytecode
 * version -- a mismatch surfaces as a load error naming the path.
 *
 * @see InferenceBackend for the base interface
 */
class LibTorchBackend : public InferenceBackend {
public:
  LibTorchBackend(const InferenceConfig &config,
                  const InferenceContext &context);
  ~LibTorchBackend() override;

  /// @copydoc InferenceBackend::name
  std::string name() const override { return "libtorch"; }

protected:
  void init_impl() override;
  bool infer_impl(const TensorMap &inputs, TensorMap &outputs) override;
  void final_impl() override;

private:
  /// Everything torch, kept out of this header so that a component including
  /// it does not need libtorch's headers (or its twenty-second compile).
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_LIBTORCH_INFERENCE_BACKEND_HPP
