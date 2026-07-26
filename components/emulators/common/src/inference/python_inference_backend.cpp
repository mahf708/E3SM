/**
 * @file python_inference_backend.cpp
 * @brief Embedded-Python inference backend implementation.
 *
 * The numpy interaction deliberately goes through the ordinary Python API
 * (`numpy.frombuffer` over a memoryview of our own storage) instead of the
 * numpy C API: that keeps numpy a *runtime* dependency only, so building
 * E3SM never needs numpy headers, while still handing Python zero-copy views
 * of E3SM memory.
 */

#include "python_inference_backend.hpp"
#include "python_interpreter.hpp"

#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

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

} // namespace

// ===========================================================================
// Impl
// ===========================================================================

struct PythonBackend::Impl {
  enum class CallStyle { AUTO, INOUT, RETURN };

  bool interpreter_started = false;

  // Cached Python objects.
  PyRef module;          ///< The imported user module
  PyRef emulator;        ///< Object owning the infer method
  PyRef infer_callable;  ///< The per-step callable
  PyRef numpy;           ///< numpy module
  PyRef np_frombuffer;   ///< numpy.frombuffer
  PyRef np_ascontiguous; ///< numpy.ascontiguousarray
  PyRef np_empty;        ///< numpy.empty
  std::map<DType, PyRef> np_dtypes;

  CallStyle style = CallStyle::AUTO;
  std::string infer_method = "infer";
  std::string finalize_method = "finalize";

  /// numpy dtype object for one of our element types.
  PyObject *dtype_object(DType dtype) {
    const auto it = np_dtypes.find(dtype);
    EMULATOR_INFER_REQUIRE(it != np_dtypes.end(),
                           "No numpy dtype cached for "
                               << dtype_name(dtype) << ".");
    return it->second.get();
  }

  /**
   * @brief numpy array sharing the tensor's memory (no copy).
   *
   * Pass the tensor's writable pointer to get a writable array; pass nullptr
   * for a read-only array, which is what a component wants for its input
   * fields — a Python bug then cannot corrupt model state.
   */
  PyRef make_array(const Tensor &tensor, void *mutable_data) {
    PyObject *dtype = dtype_object(tensor.dtype());
    const bool writable = mutable_data != nullptr;

    if (tensor.size() == 0) {
      // frombuffer rejects empty buffers; hand back a correctly shaped empty.
      PyRef shape = dims_tuple(tensor.dims());
      PyRef args(PyTuple_Pack(1, shape.get()));
      PyRef kwargs(PyDict_New());
      PyDict_SetItemString(kwargs.get(), "dtype", dtype);
      PyRef arr(PyObject_Call(np_empty.get(), args.get(), kwargs.get()));
      if (!arr) {
        py_throw("allocating an empty array for '" + tensor.name() + "'");
      }
      return arr;
    }

    // memoryview over our own storage, then frombuffer + reshape.  Both are
    // views: nothing is copied, and writes land in E3SM memory.  The cast
    // away from const is required by the C API even for PyBUF_READ, which is
    // what actually makes the resulting array read-only.
    char *base = writable ? static_cast<char *>(mutable_data)
                          : const_cast<char *>(
                                static_cast<const char *>(tensor.cdata()));
    PyRef view(PyMemoryView_FromMemory(base,
                                       static_cast<Py_ssize_t>(tensor.nbytes()),
                                       writable ? PyBUF_WRITE : PyBUF_READ));
    if (!view) {
      py_throw("creating a memoryview for '" + tensor.name() + "'");
    }

    PyRef args(PyTuple_Pack(1, view.get()));
    PyRef kwargs(PyDict_New());
    PyDict_SetItemString(kwargs.get(), "dtype", dtype);
    PyRef flat(PyObject_Call(np_frombuffer.get(), args.get(), kwargs.get()));
    if (!flat) {
      py_throw("wrapping '" + tensor.name() + "' as a numpy array");
    }

    if (tensor.rank() <= 1) {
      return flat;
    }
    PyRef shape = dims_tuple(tensor.dims());
    PyRef reshaped(
        PyObject_CallMethod(flat.get(), "reshape", "O", shape.get()));
    if (!reshaped) {
      py_throw("reshaping '" + tensor.name() + "'");
    }
    return reshaped;
  }

  /// Dict of name -> read-only numpy array for a whole TensorMap.
  PyRef make_input_dict(const TensorMap &tensors) {
    PyRef dict(PyDict_New());
    EMULATOR_INFER_REQUIRE(static_cast<bool>(dict),
                           "Out of memory building the tensor dict.");
    for (const auto &tensor : tensors) {
      PyRef arr = make_array(tensor, nullptr);
      if (PyDict_SetItemString(dict.get(), tensor.name().c_str(), arr.get()) !=
          0) {
        py_throw("adding '" + tensor.name() + "' to the tensor dict");
      }
    }
    return dict;
  }

  /// Dict of name -> writable numpy array for a whole TensorMap.
  PyRef make_output_dict(TensorMap &tensors) {
    PyRef dict(PyDict_New());
    EMULATOR_INFER_REQUIRE(static_cast<bool>(dict),
                           "Out of memory building the tensor dict.");
    for (auto &tensor : tensors) {
      PyRef arr = make_array(tensor, tensor.data());
      if (PyDict_SetItemString(dict.get(), tensor.name().c_str(), arr.get()) !=
          0) {
        py_throw("adding '" + tensor.name() + "' to the tensor dict");
      }
    }
    return dict;
  }

  /// Copy a Python array-like into an existing tensor, converting dtype.
  void copy_into(PyObject *value, Tensor &tensor) {
    PyRef args(PyTuple_Pack(1, value));
    PyRef kwargs(PyDict_New());
    PyDict_SetItemString(kwargs.get(), "dtype", dtype_object(tensor.dtype()));
    PyRef contiguous(
        PyObject_Call(np_ascontiguous.get(), args.get(), kwargs.get()));
    if (!contiguous) {
      py_throw("converting the value returned for '" + tensor.name() +
               "' to a contiguous " + dtype_name(tensor.dtype()) + " array");
    }

    Py_buffer buffer;
    if (PyObject_GetBuffer(contiguous.get(), &buffer, PyBUF_SIMPLE) != 0) {
      py_throw("accessing the buffer of the value returned for '" +
               tensor.name() + "'");
    }
    const std::size_t nbytes = static_cast<std::size_t>(buffer.len);
    if (nbytes != tensor.nbytes()) {
      PyBuffer_Release(&buffer);
      EMULATOR_INFER_REQUIRE(false,
                             "Python returned " << nbytes << " bytes for '"
                                                << tensor.name()
                                                << "' but the tensor holds "
                                                << tensor.nbytes()
                                                << " (expected shape "
                                                << tensor.to_string() << ").");
    }
    std::memcpy(tensor.data(), buffer.buf, nbytes);
    PyBuffer_Release(&buffer);
  }

  /// Element type and shape of a numpy array, for adopting returned values.
  void describe_array(PyObject *value, std::vector<std::int64_t> &dims,
                      DType &dtype) {
    PyRef as_array(PyObject_CallFunctionObjArgs(np_ascontiguous.get(), value,
                                                nullptr));
    if (!as_array) {
      py_throw("inspecting a value returned by the Python emulator");
    }

    PyRef shape = as_array.attr("shape");
    dims.clear();
    if (shape && PyTuple_Check(shape.get())) {
      const Py_ssize_t n = PyTuple_Size(shape.get());
      for (Py_ssize_t i = 0; i < n; ++i) {
        dims.push_back(PyLong_AsLongLong(PyTuple_GetItem(shape.get(), i)));
      }
    }

    PyRef dt = as_array.attr("dtype");
    PyRef dt_name = dt ? dt.attr("name") : PyRef();
    const std::string name = py_to_string(dt_name.get());
    EMULATOR_INFER_REQUIRE(!name.empty(),
                           "Could not determine the element type of a value "
                           "returned by the Python emulator.");
    dtype = dtype_from_string(name);
  }
};

// ===========================================================================
// PythonBackend
// ===========================================================================

PythonBackend::PythonBackend(const InferenceConfig &config)
    : InferenceBackend(config), m_impl(new Impl()) {
  m_module_name = config.get("python_module");
  m_impl->infer_method = config.get("python_infer_method", "infer");
  m_impl->finalize_method = config.get("python_finalize_method", "finalize");

  const std::string style = config.get("python_call_style", "auto");
  if (style == "auto") {
    m_impl->style = Impl::CallStyle::AUTO;
  } else if (style == "inout" || style == "in_out" || style == "inplace") {
    m_impl->style = Impl::CallStyle::INOUT;
  } else if (style == "return" || style == "returns") {
    m_impl->style = Impl::CallStyle::RETURN;
  } else {
    EMULATOR_INFER_REQUIRE(false, "Unknown python_call_style '"
                                      << style
                                      << "'. Valid values: auto, inout, "
                                         "return.");
  }
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

bool PythonBackend::uses_inout_style() const {
  return m_impl->style == Impl::CallStyle::INOUT;
}

void PythonBackend::init_impl() {
  EMULATOR_INFER_REQUIRE(!m_module_name.empty(),
                         "The python backend needs a module: set the "
                         "'python_module' option to the name of an importable "
                         "module providing infer().");

  PyInterpreter::instance().initialize();
  m_impl->interpreter_started = true;

  PyGilGuard gil;

  // --- sys.path ----------------------------------------------------------
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
    const std::string details = py_take_error();
    throw InferenceError(
        "The python inference backend needs numpy in the interpreter that "
        "E3SM is linked against, but importing it failed:\n" +
        details);
  }
  m_impl->np_frombuffer = m_impl->numpy.attr("frombuffer");
  m_impl->np_ascontiguous = m_impl->numpy.attr("ascontiguousarray");
  m_impl->np_empty = m_impl->numpy.attr("empty");
  EMULATOR_INFER_REQUIRE(static_cast<bool>(m_impl->np_frombuffer) &&
                             static_cast<bool>(m_impl->np_ascontiguous) &&
                             static_cast<bool>(m_impl->np_empty),
                         "The imported 'numpy' module does not look like "
                         "numpy (frombuffer/ascontiguousarray/empty are "
                         "missing).");

  PyRef np_dtype = m_impl->numpy.attr("dtype");
  for (DType dtype : {DType::FLOAT32, DType::FLOAT64, DType::INT32,
                      DType::INT64}) {
    PyRef name = py_string(dtype_name(dtype));
    PyRef obj(
        PyObject_CallFunctionObjArgs(np_dtype.get(), name.get(), nullptr));
    if (!obj) {
      py_throw(std::string("building the numpy dtype for ") +
               dtype_name(dtype));
    }
    m_impl->np_dtypes.emplace(dtype, obj);
  }

  // --- user module -------------------------------------------------------
  {
    FpeGuard fpe_guard; // the module may import torch/jax/...
    m_impl->module = PyRef(PyImport_ImportModule(m_module_name.c_str()));
  }
  if (!m_impl->module) {
    const std::string details = py_take_error();
    throw InferenceError("Could not import the python emulator module '" +
                         m_module_name +
                         "'. Is its directory on sys.path (option "
                         "'python_path')?\n" +
                         details);
  }

  // --- the emulator object ----------------------------------------------
  // Explicit factory, else the create_emulator() convention, else a
  // module-level infer().
  std::string factory_name = m_config.get("python_factory");
  const bool factory_requested = !factory_name.empty();
  if (!factory_requested && m_impl->module.attr("create_emulator")) {
    factory_name = "create_emulator";
  }

  if (!factory_name.empty()) {
    PyRef factory = m_impl->module.attr(factory_name.c_str());
    EMULATOR_INFER_REQUIRE(static_cast<bool>(factory),
                           "Module '" << m_module_name
                                      << "' has no attribute '" << factory_name
                                      << "' (option 'python_factory').");
    EMULATOR_INFER_REQUIRE(PyCallable_Check(factory.get()) != 0,
                           "'" << factory_name << "' in module '"
                               << m_module_name << "' is not callable.");
    PyRef cfg = [this] {
      // The config dict the factory receives.
      PyRef dict(PyDict_New());
      const auto set_str = [&dict](const char *key, const std::string &value) {
        PyRef v = py_string(value);
        PyDict_SetItemString(dict.get(), key, v.get());
      };
      const auto set_int = [&dict](const char *key, long long value) {
        PyRef v(PyLong_FromLongLong(value));
        PyDict_SetItemString(dict.get(), key, v.get());
      };
      const auto specs_list = [](const std::vector<TensorSpec> &specs) {
        PyRef list(PyList_New(0));
        for (const auto &spec : specs) {
          PyRef entry(PyDict_New());
          PyRef nm = py_string(spec.name);
          PyDict_SetItemString(entry.get(), "name", nm.get());
          PyRef dt = py_string(dtype_name(spec.dtype));
          PyDict_SetItemString(entry.get(), "dtype", dt.get());
          PyRef dims = dims_tuple(spec.dims);
          PyDict_SetItemString(entry.get(), "dims", dims.get());
          PyList_Append(list.get(), entry.get());
        }
        return list;
      };

      set_str("backend", m_config.backend);
      set_str("model_path", m_config.model_path);
      set_int("input_channels", m_config.input_channels);
      set_int("output_channels", m_config.output_channels);
      PyDict_SetItemString(dict.get(), "verbose",
                           m_config.verbose ? Py_True : Py_False);
      PyRef in_list = specs_list(m_config.inputs);
      PyDict_SetItemString(dict.get(), "inputs", in_list.get());
      PyRef out_list = specs_list(m_config.outputs);
      PyDict_SetItemString(dict.get(), "outputs", out_list.get());
      for (const auto &kv : m_config.options) {
        set_str(kv.first.c_str(), kv.second);
      }
      return dict;
    }();

    FpeGuard fpe_guard; // the factory typically loads the model
    m_impl->emulator =
        PyRef(PyObject_CallFunctionObjArgs(factory.get(), cfg.get(), nullptr));
    if (!m_impl->emulator) {
      py_throw("calling '" + factory_name + "' in module '" + m_module_name +
               "'");
    }
  } else {
    // Module-level infer(): the module itself acts as the emulator.
    m_impl->emulator = m_impl->module;
  }

  // --- the infer callable -----------------------------------------------
  m_impl->infer_callable = m_impl->emulator.attr(m_impl->infer_method.c_str());
  EMULATOR_INFER_REQUIRE(
      static_cast<bool>(m_impl->infer_callable),
      "The python emulator from module '"
          << m_module_name << "' has no '" << m_impl->infer_method
          << "' attribute. Provide infer(inputs, outputs) or infer(inputs), "
             "or name the method with 'python_infer_method'.");
  EMULATOR_INFER_REQUIRE(PyCallable_Check(m_impl->infer_callable.get()) != 0,
                         "'" << m_impl->infer_method
                             << "' on the python emulator from module '"
                             << m_module_name << "' is not callable.");

  // --- call style --------------------------------------------------------
  if (m_impl->style == Impl::CallStyle::AUTO) {
    // Two or more positional parameters means the emulator wants to write
    // into our output views; one parameter means it returns its results.
    m_impl->style = Impl::CallStyle::INOUT;
    PyRef inspect(PyImport_ImportModule("inspect"));
    if (inspect) {
      PyRef signature(PyObject_CallMethod(inspect.get(), "signature", "O",
                                          m_impl->infer_callable.get()));
      if (signature) {
        PyRef params = signature.attr("parameters");
        if (params) {
          const Py_ssize_t n = PyObject_Length(params.get());
          if (n == 1) {
            m_impl->style = Impl::CallStyle::RETURN;
          }
        }
      } else {
        PyErr_Clear(); // builtins and C callables have no signature
      }
    } else {
      PyErr_Clear();
    }
  }

  if (m_config.verbose) {
    std::cout << "[python inference] module=" << m_module_name
              << " method=" << m_impl->infer_method << " style="
              << (uses_inout_style() ? "inout" : "return") << "\n";
  }
}

bool PythonBackend::infer_impl(const TensorMap &inputs, TensorMap &outputs) {
  PyGilGuard gil;

  PyRef in_dict = m_impl->make_input_dict(inputs);

  if (uses_inout_style()) {
    // Outputs must exist up front so Python can write into them.
    if (outputs.empty()) {
      const auto specs = output_specs();
      EMULATOR_INFER_REQUIRE(
          !specs.empty(),
          "The python emulator writes into pre-allocated outputs, but no "
          "output tensors were provided and none are declared. Pass output "
          "tensors, declare `output:` specs, or use a Python infer(inputs) "
          "that returns its results.");
      const std::int64_t batch =
          (inputs.size() > 0 && inputs[0].rank() > 0) ? inputs[0].dim(0) : 1;
      for (const auto &spec : specs) {
        outputs.add(spec.make_tensor(batch));
      }
    }

    PyRef out_dict = m_impl->make_output_dict(outputs);
    PyRef result(PyObject_CallFunctionObjArgs(m_impl->infer_callable.get(),
                                              in_dict.get(), out_dict.get(),
                                              nullptr));
    if (!result) {
      py_throw("calling " + m_module_name + "." + m_impl->infer_method +
               "(inputs, outputs)");
    }
    return true;
  }

  // Return style: Python hands back a dict (or a single array).
  PyRef result(PyObject_CallFunctionObjArgs(m_impl->infer_callable.get(),
                                            in_dict.get(), nullptr));
  if (!result) {
    py_throw("calling " + m_module_name + "." + m_impl->infer_method +
             "(inputs)");
  }

  if (PyDict_Check(result.get())) {
    if (outputs.empty()) {
      // Build destinations for whatever Python produced.  Declared specs win
      // on order (positional consumers stay deterministic) and on element
      // type (a component asking for float32 gets float32); the shape comes
      // from the returned array, which is what knows the batch size.
      std::vector<TensorSpec> specs = output_specs();
      const bool declared = !specs.empty();
      if (specs.empty()) {
        PyObject *key = nullptr;
        PyObject *value = nullptr;
        Py_ssize_t pos = 0;
        while (PyDict_Next(result.get(), &pos, &key, &value)) {
          specs.push_back(TensorSpec(py_to_string(key), {}, DType::FLOAT64));
        }
      }
      for (const auto &spec : specs) {
        PyObject *value = PyDict_GetItemString(result.get(), spec.name.c_str());
        EMULATOR_INFER_REQUIRE(value != nullptr,
                               "The python emulator did not return '"
                                   << spec.name << "'.");
        std::vector<std::int64_t> dims;
        DType dtype = DType::FLOAT64;
        m_impl->describe_array(value, dims, dtype);
        outputs.add(Tensor(spec.name, dims, declared ? spec.dtype : dtype));
      }
    }

    for (auto &tensor : outputs) {
      PyObject *value = PyDict_GetItemString(result.get(), tensor.name().c_str());
      EMULATOR_INFER_REQUIRE(value != nullptr,
                             "The python emulator returned no value for '"
                                 << tensor.name() << "'.");
      m_impl->copy_into(value, tensor);
    }
    return true;
  }

  // A bare array is accepted when there is exactly one output.
  if (outputs.empty()) {
    const auto specs = output_specs();
    EMULATOR_INFER_REQUIRE(
        specs.size() == 1,
        "The python emulator returned a single value, but "
            << specs.size()
            << " outputs are declared. Return a dict keyed by output name.");
    std::vector<std::int64_t> dims;
    DType dtype = DType::FLOAT64;
    m_impl->describe_array(result.get(), dims, dtype);
    outputs.add(Tensor(specs[0].name, dims, dtype));
  }
  EMULATOR_INFER_REQUIRE(outputs.size() == 1,
                         "The python emulator returned a single value, but "
                             << outputs.size()
                             << " output tensors were provided. Return a dict "
                                "keyed by output name.");
  m_impl->copy_into(result.get(), outputs[0]);
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
          // Report, but do not throw: teardown must not break a model run.
          std::cerr << "[emulator::inference] warning: "
                    << m_module_name << "." << m_impl->finalize_method
                    << "() failed:\n"
                    << py_take_error() << "\n";
        }
      }
    }

    // Drop our Python references while the interpreter is still alive.
    m_impl->infer_callable = PyRef();
    m_impl->emulator = PyRef();
    m_impl->module = PyRef();
    m_impl->np_frombuffer = PyRef();
    m_impl->np_ascontiguous = PyRef();
    m_impl->np_empty = PyRef();
    m_impl->np_dtypes.clear();
    m_impl->numpy = PyRef();
  }

  PyInterpreter::instance().finalize();
  m_impl->interpreter_started = false;
}

} // namespace inference
} // namespace emulator
