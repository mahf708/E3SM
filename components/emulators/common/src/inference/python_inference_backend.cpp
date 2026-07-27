/**
 * @file python_inference_backend.cpp
 * @brief Embedded-Python inference backend implementation.
 *
 * The numpy interaction goes through the ordinary Python API
 * (`numpy.frombuffer` over a memoryview of our own storage) rather than the
 * numpy C API.  That keeps numpy a *runtime* dependency only — building E3SM
 * never needs numpy headers — while still handing Python zero-copy views of
 * E3SM memory.
 */

#include "python_inference_backend.hpp"

#include "python_interpreter.hpp"

#include <iostream>
#include <sstream>
#include <vector>

#ifndef EMULATOR_PYTHON_PACKAGE_DIR
#define EMULATOR_PYTHON_PACKAGE_DIR ""
#endif

namespace emulator {
namespace inference {

namespace {

/// Split "a:b:c" into {"a","b","c"}, dropping empties.
std::vector<std::string> split_paths(const std::string &joined) {
  std::vector<std::string> out;
  std::istringstream iss(joined);
  std::string item;
  while (std::getline(iss, item, ':')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

/// Build a Python tuple of ints from tensor dims.
PyRef dims_tuple(const std::vector<std::int64_t> &dims) {
  PyRef tuple(PyTuple_New(static_cast<Py_ssize_t>(dims.size())));
  EMULATOR_INFER_REQUIRE(static_cast<bool>(tuple),
                         "Out of memory building a shape tuple.");
  for (std::size_t i = 0; i < dims.size(); ++i) {
    // PyTuple_SetItem steals the reference it is given.
    PyTuple_SetItem(tuple.get(), static_cast<Py_ssize_t>(i),
                    PyLong_FromLongLong(static_cast<long long>(dims[i])));
  }
  return tuple;
}

void dict_set(PyObject *dict, const char *key, PyRef value) {
  EMULATOR_INFER_REQUIRE(static_cast<bool>(value),
                         "Out of memory building the value for '" << key
                                                                  << "'.");
  PyDict_SetItemString(dict, key, value.get());
}

void dict_set_string(PyObject *dict, const char *key, const std::string &v) {
  dict_set(dict, key, py_string(v));
}

void dict_set_int(PyObject *dict, const char *key, long long v) {
  dict_set(dict, key, PyRef(PyLong_FromLongLong(v)));
}

} // namespace

// ===========================================================================
// Impl
// ===========================================================================

struct PythonBackend::Impl {
  bool interpreter_started = false;

  PyRef module;         ///< The imported bridge module
  PyRef emulator;       ///< The object returned by the factory
  PyRef infer_callable; ///< Its infer method
  PyRef numpy;
  PyRef np_frombuffer;

  std::string infer_method = "infer";
  std::string finalize_method = "finalize";

  /**
   * @brief A numpy array sharing memory with the given buffer.
   *
   * `writable == false` yields a read-only array, which is what a component
   * wants for its input fields: a bug in the model then cannot corrupt state.
   */
  PyRef wrap_buffer(const void *base, std::size_t nbytes,
                    const std::vector<std::int64_t> &dims, const char *dtype,
                    bool writable, const std::string &what) {
    if (nbytes == 0) {
      // frombuffer rejects empty buffers; hand back a correctly shaped empty.
      PyRef shape = dims_tuple(dims);
      PyRef empty = numpy.attr("empty");
      PyRef args(PyTuple_Pack(1, shape.get()));
      PyRef kwargs(PyDict_New());
      dict_set_string(kwargs.get(), "dtype", dtype);
      PyRef arr(PyObject_Call(empty.get(), args.get(), kwargs.get()));
      if (!arr) {
        py_throw("allocating an empty array for '" + what + "'");
      }
      return arr;
    }

    // memoryview over our storage, then frombuffer + reshape.  Both are
    // views: nothing is copied, and writes land in E3SM memory.  The cast
    // away from const is required by the C API even for PyBUF_READ, which is
    // what actually makes the resulting array read-only.
    PyRef view(PyMemoryView_FromMemory(
        const_cast<char *>(static_cast<const char *>(base)),
        static_cast<Py_ssize_t>(nbytes), writable ? PyBUF_WRITE : PyBUF_READ));
    if (!view) {
      py_throw("creating a memoryview for '" + what + "'");
    }

    PyRef args(PyTuple_Pack(1, view.get()));
    PyRef kwargs(PyDict_New());
    dict_set_string(kwargs.get(), "dtype", dtype);
    PyRef flat(PyObject_Call(np_frombuffer.get(), args.get(), kwargs.get()));
    if (!flat) {
      py_throw("wrapping '" + what + "' as a numpy array");
    }

    if (dims.size() <= 1) {
      return flat;
    }
    PyRef shape = dims_tuple(dims);
    PyRef reshaped(
        PyObject_CallMethod(flat.get(), "reshape", "O", shape.get()));
    if (!reshaped) {
      py_throw("reshaping '" + what + "'");
    }
    return reshaped;
  }

  /// An owning numpy copy, for metadata Python may outlive or mutate.
  PyRef copy_buffer(const void *base, std::size_t nbytes,
                    const std::vector<std::int64_t> &dims, const char *dtype,
                    const std::string &what) {
    PyRef view = wrap_buffer(base, nbytes, dims, dtype, false, what);
    PyRef copy(PyObject_CallMethod(view.get(), "copy", nullptr));
    if (!copy) {
      py_throw("copying '" + what + "'");
    }
    return copy;
  }

  /// Dict of name -> numpy view for a whole TensorMap.
  PyRef tensor_dict(TensorMap &tensors, bool writable) {
    PyRef dict(PyDict_New());
    EMULATOR_INFER_REQUIRE(static_cast<bool>(dict),
                           "Out of memory building the tensor dict.");
    for (auto &tensor : tensors) {
      const void *base =
          writable ? static_cast<const void *>(tensor.data()) : tensor.cdata();
      PyRef arr = wrap_buffer(base, tensor.nbytes(), tensor.dims(), "float64",
                              writable, tensor.name());
      if (PyDict_SetItemString(dict.get(), tensor.name().c_str(), arr.get()) !=
          0) {
        py_throw("adding '" + tensor.name() + "' to the tensor dict");
      }
    }
    return dict;
  }
};

// ===========================================================================
// PythonBackend
// ===========================================================================

PythonBackend::PythonBackend(const InferenceConfig &config,
                             const InferenceContext &context)
    : InferenceBackend(config, context), m_impl(new Impl()) {
  m_module_name = config.get("python_module", "e3sm_emulator.bridge");
  m_impl->infer_method = config.get("python_infer_method", "infer");
  m_impl->finalize_method = config.get("python_finalize_method", "finalize");
}

PythonBackend::~PythonBackend() {
  try {
    PythonBackend::finalize();
  } catch (const std::exception &e) {
    // A destructor must not throw; a failed Python teardown is worth a note
    // but not worth aborting a run that is already shutting down.
    std::cerr << "[emulator::inference] warning: Python backend teardown "
              << "failed: " << e.what() << "\n";
  }
}

void PythonBackend::init_impl() {
  PyInterpreter::instance().initialize();
  m_impl->interpreter_started = true;

  PyGilGuard gil;

  // --- sys.path ----------------------------------------------------------
  // The package shipped alongside this source goes on last so that it ends
  // up *after* anything the user named: a site can override it wholesale.
  const std::string shipped = EMULATOR_PYTHON_PACKAGE_DIR;
  if (!shipped.empty()) {
    PyInterpreter::instance().add_sys_path(shipped);
  }
  if (m_config.get_bool("python_add_cwd", true)) {
    PyInterpreter::instance().add_sys_path(".");
  }
  // Later entries end up earlier in sys.path; walk backwards so the user's
  // first entry wins.
  const auto paths = split_paths(m_config.get("python_path"));
  for (auto it = paths.rbegin(); it != paths.rend(); ++it) {
    PyInterpreter::instance().add_sys_path(*it);
  }

  // --- numpy -------------------------------------------------------------
  {
    FpeGuard fpe_guard; // importing numpy raises benign FPEs
    m_impl->numpy = PyRef(PyImport_ImportModule("numpy"));
  }
  if (!m_impl->numpy) {
    throw InferenceError(
        "The python inference backend needs numpy in the interpreter E3SM is "
        "linked against, but importing it failed:\n" +
        py_take_error());
  }
  m_impl->np_frombuffer = m_impl->numpy.attr("frombuffer");
  EMULATOR_INFER_REQUIRE(static_cast<bool>(m_impl->np_frombuffer),
                         "The imported 'numpy' has no frombuffer; is it "
                         "really numpy?");

  // --- the bridge module -------------------------------------------------
  {
    FpeGuard fpe_guard; // the module imports torch, and torch traps too
    m_impl->module = PyRef(PyImport_ImportModule(m_module_name.c_str()));
  }
  if (!m_impl->module) {
    throw InferenceError("Could not import the python emulator module '" +
                         m_module_name +
                         "'. Is its directory on sys.path (option "
                         "'python_path')?\n" +
                         py_take_error());
  }

  // --- the configuration dict handed to the factory ----------------------
  PyRef cfg(PyDict_New());
  EMULATOR_INFER_REQUIRE(static_cast<bool>(cfg),
                         "Out of memory building the config dict.");
  dict_set_string(cfg.get(), "backend", m_config.backend);
  dict_set_string(cfg.get(), "model_path", m_config.model_path);
  dict_set_int(cfg.get(), "input_channels", m_config.input_channels);
  dict_set_int(cfg.get(), "output_channels", m_config.output_channels);
  PyDict_SetItemString(cfg.get(), "verbose",
                       m_config.verbose ? Py_True : Py_False);
  const auto name_list = [](const std::vector<std::string> &names) {
    PyRef list(PyList_New(0));
    for (const auto &n : names) {
      PyRef item = py_string(n);
      PyList_Append(list.get(), item.get());
    }
    return list;
  };
  dict_set(cfg.get(), "inputs", name_list(m_config.inputs));
  dict_set(cfg.get(), "outputs", name_list(m_config.outputs));
  for (const auto &kv : m_config.options) {
    dict_set_string(cfg.get(), kv.first.c_str(), kv.second);
  }

  // --- the context: ranks and this rank's columns ------------------------
  PyRef ctx(PyDict_New());
  dict_set_int(ctx.get(), "rank", m_context.rank);
  dict_set_int(ctx.get(), "world_size", m_context.size);
  dict_set_int(ctx.get(), "local_rank", m_context.local_rank);
  dict_set_int(ctx.get(), "local_size", m_context.local_size);
  dict_set_string(ctx.get(), "node_name", m_context.node_name);
  dict_set_string(ctx.get(), "master_addr", m_context.master_addr);
  dict_set_int(ctx.get(), "master_port", m_context.master_port);
  dict_set_int(ctx.get(), "nx", m_context.nx);
  dict_set_int(ctx.get(), "ny", m_context.ny);
  dict_set_int(ctx.get(), "num_global_cols", m_context.num_global_cols);
  {
    const auto n = static_cast<std::int64_t>(m_context.col_gids.size());
    // Copies, not views: this is small, one-time metadata that the model is
    // free to keep, reorder or sort for as long as it likes.
    dict_set(ctx.get(),
             "col_gids", m_impl->copy_buffer(m_context.col_gids.data(),
                                             m_context.col_gids.size() *
                                                 sizeof(int),
                                             {n}, "int32", "col_gids"));
    dict_set(ctx.get(), "lat",
             m_impl->copy_buffer(m_context.lat.data(),
                                 m_context.lat.size() * sizeof(double),
                                 {static_cast<std::int64_t>(
                                     m_context.lat.size())},
                                 "float64", "lat"));
    dict_set(ctx.get(), "lon",
             m_impl->copy_buffer(m_context.lon.data(),
                                 m_context.lon.size() * sizeof(double),
                                 {static_cast<std::int64_t>(
                                     m_context.lon.size())},
                                 "float64", "lon"));
  }
  PyDict_SetItemString(cfg.get(), "context", ctx.get());

  // --- build the emulator ------------------------------------------------
  const std::string factory_name =
      m_config.get("python_factory", "create_emulator");
  PyRef factory = m_impl->module.attr(factory_name.c_str());
  EMULATOR_INFER_REQUIRE(static_cast<bool>(factory),
                         "Module '" << m_module_name << "' has no attribute '"
                                    << factory_name
                                    << "' (option 'python_factory').");
  EMULATOR_INFER_REQUIRE(PyCallable_Check(factory.get()) != 0,
                         "'" << factory_name << "' in module '"
                             << m_module_name << "' is not callable.");
  {
    FpeGuard fpe_guard; // the factory loads the model
    m_impl->emulator =
        PyRef(PyObject_CallFunctionObjArgs(factory.get(), cfg.get(), nullptr));
  }
  if (!m_impl->emulator) {
    py_throw("calling " + m_module_name + "." + factory_name + "(config)");
  }

  m_impl->infer_callable = m_impl->emulator.attr(m_impl->infer_method.c_str());
  EMULATOR_INFER_REQUIRE(
      static_cast<bool>(m_impl->infer_callable) &&
          PyCallable_Check(m_impl->infer_callable.get()) != 0,
      "The python emulator from '"
          << m_module_name << "' has no callable '" << m_impl->infer_method
          << "(inputs, outputs)' method (option 'python_infer_method').");
}

bool PythonBackend::infer_impl(const TensorMap &inputs, TensorMap &outputs) {
  PyGilGuard gil;

  // Read-only views for the inputs, writable views for the outputs.  The
  // const_cast is confined here: tensor_dict never writes through an input,
  // it only needs a non-const TensorMap& to iterate.
  PyRef in_dict =
      m_impl->tensor_dict(const_cast<TensorMap &>(inputs), /*writable=*/false);
  PyRef out_dict = m_impl->tensor_dict(outputs, /*writable=*/true);

  FpeGuard fpe_guard; // model kernels are entitled to raise benign FPEs
  PyRef result(PyObject_CallFunctionObjArgs(
      m_impl->infer_callable.get(), in_dict.get(), out_dict.get(), nullptr));
  if (!result) {
    py_throw("calling " + m_module_name + "." + m_impl->infer_method +
             "(inputs, outputs)");
  }
  return true;
}

void PythonBackend::final_impl() {
  if (!m_impl->interpreter_started) {
    return;
  }

  {
    PyGilGuard gil;
    if (m_impl->emulator && !m_impl->finalize_method.empty()) {
      PyRef fin = m_impl->emulator.attr(m_impl->finalize_method.c_str());
      if (fin && PyCallable_Check(fin.get()) != 0) {
        PyRef result(PyObject_CallObject(fin.get(), nullptr));
        if (!result) {
          // Report, but do not throw: teardown must not break a run that has
          // already produced its answers.
          std::cerr << "[emulator::inference] warning: " << m_module_name << "."
                    << m_impl->finalize_method << "() failed:\n"
                    << py_take_error() << "\n";
        }
      }
    }

    // Drop our references while the interpreter is still alive.  This is what
    // actually releases the model and its weights.
    m_impl->infer_callable = PyRef();
    m_impl->emulator = PyRef();
    m_impl->module = PyRef();
    m_impl->np_frombuffer = PyRef();
    m_impl->numpy = PyRef();
  }

  PyInterpreter::instance().finalize();
  m_impl->interpreter_started = false;
}

} // namespace inference
} // namespace emulator
