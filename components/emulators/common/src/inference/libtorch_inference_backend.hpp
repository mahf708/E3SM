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
 * `model_path` is an archive written by `torch.jit.save` -- what the traced
 * ACE atmosphere and Samudra ocean checkpoints are.  No Python interpreter
 * is involved at run time, which is the whole reason to prefer this over
 * PythonBackend: nothing has to agree about interpreters, site-packages or
 * the GIL, and a model cannot import its way into a different numpy than
 * the one E3SM was built against.  What it gives up in exchange is
 * expressiveness -- a TorchScript module is a pure function of its inputs,
 * so it cannot contain an MPI collective the way a Python model could.
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
 * Shapes are the caller's business, not this backend's.  Each input tensor
 * is handed to `forward` as a positional argument with exactly the shape
 * its Tensor declares, in TensorMap order, and the module's outputs are
 * copied back into the output tensors in order.  Only element counts are
 * checked, and a mismatch throws with both shapes in the message: the
 * alternative -- reshaping to fit -- is how a transposed field reaches a
 * model that then produces no NaN and no complaint.  In particular nothing
 * here knows that ACE and Samudra take `[1, channels, ny, nx]`; that lives
 * in whatever built the Tensor.
 *
 * The framework's tensors are always `double` (every E3SM field is
 * `real(r8)`), so a float32 module means a convert-in / convert-out per
 * step.  That is a real cost on a big field and it is unavoidable at this
 * boundary; it is also the same cost the Python backend pays inside the
 * model.
 *
 * As far as this backend knows it is version-agnostic: it uses only the
 * stable `torch::jit` surface.  The archive and the libtorch it is loaded
 * into still have to agree with each other -- a module saved by a much newer
 * torch can use bytecode an older libtorch will refuse -- and that failure
 * shows up as a load error naming the path.
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
