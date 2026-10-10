/**
 * @file python_inference_backend.cpp
 * @brief Python inference backend implementation.
 */

// Python.h must come before any standard header
#include <Python.h>

#include "python_inference_backend.hpp"

#include "fpe_guard.hpp"
#include "inference_error.hpp"

#include <cstdint>
#include <sstream>
#include <vector>

namespace emulator {
namespace inference {

namespace {

/// Holds the GIL for the lifetime of the object; safe to nest.
class GilGuard {
public:
  GilGuard() : m_state(PyGILState_Ensure()) {}
  ~GilGuard() { PyGILState_Release(m_state); }
  GilGuard(const GilGuard &) = delete;
  GilGuard &operator=(const GilGuard &) = delete;

private:
  PyGILState_STATE m_state;
};

/// Owning, move-only handle to a PyObject; takes over a new reference.
class PyRef {
public:
  PyRef() = default;
  explicit PyRef(PyObject *obj) : m_obj(obj) {}
  ~PyRef() { reset(); }

  PyRef(const PyRef &) = delete;
  PyRef &operator=(const PyRef &) = delete;
  PyRef(PyRef &&other) noexcept : m_obj(other.m_obj) { other.m_obj = nullptr; }
  PyRef &operator=(PyRef &&other) noexcept {
    if (this != &other) {
      reset();
      m_obj = other.m_obj;
      other.m_obj = nullptr;
    }
    return *this;
  }

  PyObject *get() const { return m_obj; }
  explicit operator bool() const { return m_obj != nullptr; }

private:
  void reset() {
    if (m_obj != nullptr && Py_IsInitialized() != 0) {
      GilGuard gil;
      Py_DECREF(m_obj);
    }
    m_obj = nullptr;
  }

  PyObject *m_obj = nullptr;
};

std::string to_string(PyObject *obj) {
  PyRef str(obj != nullptr ? PyObject_Str(obj) : nullptr);
  const char *utf8 = str ? PyUnicode_AsUTF8(str.get()) : nullptr;
  if (utf8 == nullptr) {
    PyErr_Clear();
    return "";
  }
  return utf8;
}

/// Clear the pending Python exception and return it with its traceback.
std::string take_error() {
#if PY_VERSION_HEX >= 0x030C0000
  PyRef value(PyErr_GetRaisedException());
#else
  PyObject *t = nullptr, *v = nullptr, *tb = nullptr;
  PyErr_Fetch(&t, &v, &tb);
  PyErr_NormalizeException(&t, &v, &tb);
  if (v != nullptr && tb != nullptr) {
    PyException_SetTraceback(v, tb);
  }
  PyRef type(t), value(v), trace(tb);
#endif
  if (!value) {
    return "";
  }

  // traceback.format_exception(type, value, tb), joined
  PyRef module(PyImport_ImportModule("traceback"));
  PyRef trace_obj(PyException_GetTraceback(value.get()));
  PyRef lines(module ? PyObject_CallMethod(
                           module.get(), "format_exception", "OOO",
                           reinterpret_cast<PyObject *>(Py_TYPE(value.get())),
                           value.get(), trace_obj ? trace_obj.get() : Py_None)
                     : nullptr);
  PyRef empty(PyUnicode_FromString(""));
  PyRef joined(lines && empty ? PyUnicode_Join(empty.get(), lines.get())
                              : nullptr);
  const std::string message = to_string(joined.get());
  PyErr_Clear();
  return message.empty() ? to_string(value.get()) : message;
}

/// Own a new reference, or throw the pending Python exception.
PyRef checked(PyObject *obj, const std::string &doing) {
  if (obj == nullptr) {
    throw InferenceError("Python error while " + doing + ":\n" + take_error());
  }
  return PyRef(obj);
}

void set_item(const PyRef &dict, const std::string &key, const PyRef &value) {
  if (PyDict_SetItemString(dict.get(), key.c_str(), value.get()) != 0) {
    throw InferenceError("Python error while setting '" + key + "':\n" +
                         take_error());
  }
}

PyRef py_string(const std::string &s) {
  return checked(PyUnicode_FromString(s.c_str()), "converting a string");
}

PyRef py_int(long long value) {
  return checked(PyLong_FromLongLong(value), "converting an integer");
}

/// Prepend colon-separated directories to sys.path; the first one wins.
void prepend_sys_path(const std::string &paths) {
  std::vector<std::string> entries;
  std::istringstream stream(paths);
  for (std::string entry; std::getline(stream, entry, ':');) {
    if (!entry.empty()) {
      entries.push_back(entry);
    }
  }
  PyObject *sys_path = PySys_GetObject("path"); // borrowed
  EMULATOR_INFER_REQUIRE(sys_path != nullptr, "Python has no sys.path.");
  for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
    PyRef entry = py_string(*it);
    if (PySequence_Contains(sys_path, entry.get()) == 0) {
      PyList_Insert(sys_path, 0, entry.get());
    }
  }
}

/// The dict handed to the factory.
PyRef config_dict(const InferenceConfig &config) {
  PyRef dict = checked(PyDict_New(), "building the config");
  for (const auto &option : config.options) {
    set_item(dict, option.first, py_string(option.second));
  }
  set_item(dict, "model_path", py_string(config.model_path));
  set_item(dict, "input_channels", py_int(config.input_channels));
  set_item(dict, "output_channels", py_int(config.output_channels));
  set_item(dict, "verbose", PyRef(PyBool_FromLong(config.verbose ? 1 : 0)));
  return dict;
}

/// A python tuple of integers; scale multiplies each entry.
PyRef py_tuple(const std::vector<std::int64_t> &v, const std::string &doing,
               std::int64_t scale = 1) {
  PyRef t = checked(PyTuple_New(static_cast<Py_ssize_t>(v.size())), doing);
  for (std::size_t i = 0; i < v.size(); ++i) {
    // PyTuple_SetItem takes over the reference
    PyTuple_SetItem(t.get(), static_cast<Py_ssize_t>(i),
                    PyLong_FromLongLong(v[i] * scale));
  }
  return t;
}

/**
 * An object exposing a tensor's device memory through the CUDA array
 * interface, which cupy, torch, numba, ... wrap without a copy
 * (e.g. cupy.asarray(x), torch.as_tensor(x, device='cuda')).
 */
PyRef as_device_array(const PyRef &namespace_type, const Tensor &tensor,
                      const double *data, bool writable) {
  const std::string doing = "wrapping device tensor '" + tensor.name() + "'";
  PyRef cai = checked(PyDict_New(), doing);
  set_item(cai, "shape", py_tuple(tensor.dims(), doing));
  set_item(cai, "strides",
           py_tuple(tensor.strides(), doing,
                    static_cast<std::int64_t>(sizeof(double))));
  set_item(cai, "typestr", py_string("<f8"));
  PyRef ptr_and_ro = checked(PyTuple_New(2), doing);
  PyTuple_SetItem(ptr_and_ro.get(), 0,
                  PyLong_FromUnsignedLongLong(
                      reinterpret_cast<std::uintptr_t>(data)));
  PyTuple_SetItem(ptr_and_ro.get(), 1, PyBool_FromLong(writable ? 0 : 1));
  set_item(cai, "data", ptr_and_ro);
  set_item(cai, "version", py_int(3));
  // The caller synchronizes its own work before calling infer()
  Py_INCREF(Py_None);
  set_item(cai, "stream", PyRef(Py_None));

  PyRef args = checked(PyTuple_New(0), doing);
  PyRef kwargs = checked(PyDict_New(), doing);
  set_item(kwargs, "__cuda_array_interface__", cai);
  return checked(PyObject_Call(namespace_type.get(), args.get(), kwargs.get()),
                 doing);
}

/// A numpy array that views (does not copy) a tensor's host memory.
PyRef as_numpy(const PyRef &numpy, const Tensor &tensor, const double *data,
               bool writable) {
  const std::string doing = "wrapping tensor '" + tensor.name() + "'";
  PyRef shape = py_tuple(tensor.dims(), doing);

  if (tensor.size() == 0) {
    // There is no memory to view
    PyRef array = checked(
        PyObject_CallMethod(numpy.get(), "empty", "(O)", shape.get()), doing);
    if (!writable) {
      PyRef flags =
          checked(PyObject_GetAttrString(array.get(), "flags"), doing);
      PyObject_SetAttrString(flags.get(), "writeable", Py_False);
    }
    return array;
  }

  // PyBUF_READ is what makes the array read-only, despite the cast
  PyRef memory = checked(
      PyMemoryView_FromMemory(
          const_cast<char *>(reinterpret_cast<const char *>(data)),
          static_cast<Py_ssize_t>(tensor.span() * sizeof(double)),
          writable ? PyBUF_WRITE : PyBUF_READ),
      doing);
  PyRef flat = checked(PyObject_CallMethod(numpy.get(), "frombuffer", "Os",
                                           memory.get(), "float64"),
                       doing);
  if (tensor.contiguous()) {
    return checked(
        PyObject_CallMethod(flat.get(), "reshape", "(O)", shape.get()), doing);
  }

  // Strided memory (e.g. a padded field): view it with its strides
  PyRef ndarray =
      checked(PyObject_GetAttrString(numpy.get(), "ndarray"), doing);
  PyRef kwargs = checked(PyDict_New(), doing);
  set_item(kwargs, "shape", shape);
  set_item(kwargs, "dtype", py_string("float64"));
  set_item(kwargs, "buffer", flat);
  set_item(kwargs, "strides",
           py_tuple(tensor.strides(), doing,
                    static_cast<std::int64_t>(sizeof(double))));
  PyRef args = checked(PyTuple_New(0), doing);
  return checked(PyObject_Call(ndarray.get(), args.get(), kwargs.get()),
                 doing);
}

} // namespace

struct PythonBackend::Impl {
  PyRef numpy;
  PyRef simple_namespace; ///< types.SimpleNamespace, to wrap device memory
  PyRef model;            ///< What the factory returned
  bool device_arrays = false;
};

PythonBackend::PythonBackend(const InferenceConfig &config)
    : InferenceBackend(config), m_impl(new Impl()) {
  const std::string module_name = m_config.get("python_module");
  const std::string factory_name =
      m_config.get("python_factory", "create_emulator");
  EMULATOR_INFER_REQUIRE(!module_name.empty(),
                         "The Python backend needs the python_module option.");

  if (Py_IsInitialized() == 0) {
    // 0: leave signal handling to the host model
    Py_InitializeEx(0);
  }
  GilGuard gil;
  FpeGuard no_fpe;

  prepend_sys_path(m_config.get("python_path"));
  m_impl->numpy = checked(PyImport_ImportModule("numpy"), "importing numpy");
  PyRef types = checked(PyImport_ImportModule("types"), "importing types");
  m_impl->simple_namespace =
      checked(PyObject_GetAttrString(types.get(), "SimpleNamespace"),
              "looking up types.SimpleNamespace");
  m_impl->device_arrays = m_config.get_bool("device_arrays", false);

  PyRef module = checked(PyImport_ImportModule(module_name.c_str()),
                         "importing '" + module_name +
                             "' (is its directory in python_path?)");
  PyRef factory =
      checked(PyObject_GetAttrString(module.get(), factory_name.c_str()),
              "looking up " + module_name + "." + factory_name);
  PyRef settings = config_dict(m_config);
  m_impl->model = checked(
      PyObject_CallFunctionObjArgs(factory.get(), settings.get(), nullptr),
      "calling " + module_name + "." + factory_name + "(config)");
}

PythonBackend::~PythonBackend() = default;

bool PythonBackend::infer(const TensorMap &inputs, TensorMap &outputs) {
  EMULATOR_INFER_REQUIRE(m_impl, "The Python backend was finalized.");

  GilGuard gil;
  FpeGuard no_fpe;

  auto wrap = [&](const Tensor &tensor, const double *data, bool writable) {
    EMULATOR_INFER_REQUIRE(accepts(tensor.memory().space),
                           "Tensor " << tensor.to_string()
                                     << " is in device memory: set the "
                                        "device_arrays option to accept it.");
    return tensor.on_device()
               ? as_device_array(m_impl->simple_namespace, tensor, data,
                                 writable)
               : as_numpy(m_impl->numpy, tensor, data, writable);
  };
  PyRef in = checked(PyDict_New(), "building the inputs");
  for (const auto &tensor : inputs) {
    set_item(in, tensor.name(), wrap(tensor, tensor.cdata(), false));
  }
  PyRef out = checked(PyDict_New(), "building the outputs");
  for (auto &tensor : outputs) {
    set_item(out, tensor.name(), wrap(tensor, tensor.data(), true));
  }
  checked(PyObject_CallMethod(m_impl->model.get(), "infer", "OO", in.get(),
                              out.get()),
          "calling infer(inputs, outputs)");
  return true;
}

bool PythonBackend::accepts(MemorySpace space) const {
  return space == MemorySpace::HOST || (m_impl && m_impl->device_arrays);
}

void PythonBackend::finalize() {
  if (!m_impl) {
    return;
  }
  // Dropping the model is what frees it, whether or not finalize() succeeds
  const std::unique_ptr<Impl> impl = std::move(m_impl);

  GilGuard gil;
  if (PyObject_HasAttrString(impl->model.get(), "finalize") != 0) {
    checked(PyObject_CallMethod(impl->model.get(), "finalize", nullptr),
            "calling finalize()");
  }
}

} // namespace inference
} // namespace emulator
