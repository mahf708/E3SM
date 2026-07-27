/**
 * @file python_inference_backend.hpp
 * @brief Inference backend that calls a Python emulator in-process.
 */

#ifndef E3SM_EMULATOR_PYTHON_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_PYTHON_INFERENCE_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>

#include "inference_backend.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Runs a Python emulator through an embedded CPython interpreter.
 *
 * This is the path with the shortest distance from a trained model to a
 * running E3SM component: the model stays in Python (PyTorch, JAX, Keras,
 * whatever the science team trained it in), and E3SM calls it directly.  Only
 * the tensor *metadata* crosses the boundary — the arrays handed to Python are
 * numpy views of the C++ (or Fortran) memory, so a step copies nothing.
 *
 * ## The Python side
 *
 * A module is imported and one object is obtained from it:
 *  1. the callable named by the `python_factory` option, if given;
 *  2. otherwise a module-level `create_emulator(config)` if it exists;
 *  3. otherwise the module itself, if it has a module-level `infer`.
 *
 * The factory receives one argument: a dict of configuration values (all
 * options as strings, plus `model_path`, `verbose`, `input_channels`,
 * `output_channels`, `inputs` and `outputs` spec lists).
 *
 * The resulting object's `infer` (or `python_infer_method`) is called once per
 * step in one of two styles, detected from its signature:
 *
 * ```python
 * # in-place style (two parameters): write into the pre-shaped output views.
 * def infer(self, inputs, outputs):
 *     outputs["dT"][:] = self.model(inputs["T"])
 *
 * # return style (one parameter): hand back a dict (or a single array).
 * def infer(self, inputs):
 *     return {"dT": self.model(inputs["T"])}
 * ```
 *
 * The in-place style is the one that avoids all copies; the return style is
 * more convenient and costs one copy per output.  Set `python_call_style` to
 * `inout` or `return` to override the detection.
 *
 * An optional `finalize()` (or `python_finalize_method`) is called on
 * shutdown.
 *
 * ## Options
 *  - `python_module`   (required) module to import, e.g. `my_emulator`
 *  - `python_path`     colon-separated directories prepended to `sys.path`
 *  - `python_factory`  factory callable in the module
 *  - `python_infer_method`    default `infer`
 *  - `python_finalize_method` default `finalize`
 *  - `python_call_style`      `auto` (default), `inout` or `return`
 *  - `python_add_cwd`         also put the run directory on `sys.path`
 *                             (default true)
 *
 * ## Lifetime and threading
 * The numpy arrays are only valid during the call: a Python emulator must not
 * stash them for later (copy instead).  One interpreter is shared per process
 * and the GIL is held for the duration of each `infer`, so two threads of one
 * rank cannot infer concurrently.  Separate MPI ranks are separate processes
 * with separate interpreters and do not contend.
 *
 * ## What this backend does not do
 * It is a local engine, like the others: it does not accept, duplicate or
 * validate a communicator, does not initialize `torch.distributed`, and does
 * not propagate collective failures.  A model that wants to be MPI-parallel
 * can be handed a communicator handle through the option map and rebuild it
 * with mpi4py — a convention available to model code, with the model owning
 * what follows.  It is nonetheless the only backend where the model *can*
 * contain collectives, since neither ONNX nor TorchScript can express them.
 */
class PythonBackend : public InferenceBackend {
public:
  explicit PythonBackend(const InferenceConfig &config);
  ~PythonBackend() override;

  /// @copydoc InferenceBackend::name
  std::string name() const override { return "Python"; }

  /// @brief Module this backend imports.
  const std::string &module_name() const { return m_module_name; }

  /// @brief True if the Python callable is called as `infer(inputs, outputs)`.
  bool uses_inout_style() const;

protected:
  /// @copydoc InferenceBackend::init_impl
  void init_impl() override;

  /// @copydoc InferenceBackend::infer_impl
  bool infer_impl(const TensorMap &inputs, TensorMap &outputs) override;

  /// @copydoc InferenceBackend::final_impl
  void final_impl() override;

private:
  /// Python objects are hidden so this header stays free of <Python.h>.
  struct Impl;
  std::unique_ptr<Impl> m_impl;
  std::string m_module_name;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_PYTHON_INFERENCE_BACKEND_HPP
