/**
 * @file python_inference_backend.hpp
 * @brief Inference backend that keeps the model in Python.
 */

#ifndef E3SM_EMULATOR_PYTHON_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_PYTHON_INFERENCE_BACKEND_HPP

#include <memory>
#include <string>

#include "inference_backend.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Runs a Python emulator in this process.
 *
 * At initialization the backend imports a module (`python_module`, default
 * `e3sm_emulator.bridge`) and calls a factory (`python_factory`, default
 * `create_emulator`) with a dict holding every configuration option plus a
 * `context` entry describing this rank and the columns it owns.  The factory
 * returns an object with an `infer(inputs, outputs)` method, called once per
 * step:
 *
 * @code{.py}
 *     def infer(self, inputs, outputs):
 *         outputs["dT"][:] = self.model(inputs["T"])
 * @endcode
 *
 * `inputs` and `outputs` are dicts of numpy arrays that *view E3SM memory*
 * directly -- nothing is copied in either direction, and the model writes
 * its results into the component's own arrays.  Input arrays are read-only,
 * so a bug in a model cannot corrupt component state.  Zero-copy ends at the
 * numpy view: a model that moves the array to a GPU or converts it to
 * float32 pays for that copy like anybody else.
 *
 * The backend does not know or care whether the model is distributed.  It
 * hands the model the component communicator's rank, size and rendezvous
 * (see InferenceContext), and a model that wants a process group builds one
 * from those.  That is the only route to genuinely distributed inference
 * available here -- neither ONNX nor TorchScript can express a collective --
 * which is why this is the backend worth having first.
 */
class PythonBackend : public InferenceBackend {
public:
  PythonBackend(const InferenceConfig &config, const InferenceContext &context);
  ~PythonBackend() override;

  /// @copydoc InferenceBackend::name
  std::string name() const override { return "Python"; }

protected:
  void init_impl() override;
  bool infer_impl(const TensorMap &inputs, TensorMap &outputs) override;
  void final_impl() override;

private:
  /// @brief Import numpy and the module, and build the emulator.
  ///
  /// Separate from init_impl() so that everything it acquires can be rolled
  /// back in one place when any step of it throws.
  void load_model();

  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_PYTHON_INFERENCE_BACKEND_HPP
