/**
 * @file python_interpreter.cpp
 * @brief Embedded CPython session management.
 */

#include "python_interpreter.hpp"

#include <sstream>

#if defined(__GLIBC__) || defined(__GNU_LIBRARY__)
#include <cfenv>
#define EMULATOR_HAS_FEENABLEEXCEPT 1
#endif

namespace emulator {
namespace inference {

// ===========================================================================
// PyInterpreter
// ===========================================================================

PyInterpreter &PyInterpreter::instance() {
  static PyInterpreter interpreter;
  return interpreter;
}

void PyInterpreter::initialize() {
  if (m_customers == 0 && !Py_IsInitialized()) {
    // Importing numpy (and most ML stacks) trips benign FPEs.
    FpeGuard fpe_guard;

    // 0: do not install Python's signal handlers.  MPI and the E3SM driver
    // own signal handling; a Ctrl-C must not be swallowed by the interpreter.
    Py_InitializeEx(0);
    EMULATOR_INFER_REQUIRE(Py_IsInitialized(),
                           "Failed to start the embedded Python interpreter.");
    m_owns = true;

    // Py_InitializeEx leaves the GIL held by this thread.  Release it so all
    // later access can go through PyGILState_Ensure uniformly.
    m_saved_state = PyEval_SaveThread();
  }
  ++m_customers;
}

void PyInterpreter::finalize() {
  EMULATOR_INFER_REQUIRE(m_customers > 0,
                         "PyInterpreter::finalize() called without a matching "
                         "initialize().");
  --m_customers;
  // The interpreter intentionally stays up; see the header for why.
}

bool PyInterpreter::shutdown() {
  if (m_customers > 0 || !m_owns) {
    return false; // still in use, or we never owned it
  }

  if (m_saved_state != nullptr) {
    PyEval_RestoreThread(m_saved_state);
    m_saved_state = nullptr;
  }
  // Ignore the return code: a non-zero status here means a module could not
  // clean itself up, which must not turn into an error during model shutdown.
  Py_FinalizeEx();
  m_owns = false;
  return true;
}

void PyInterpreter::add_sys_path(const std::string &path) {
  EMULATOR_INFER_REQUIRE(Py_IsInitialized(),
                         "Cannot modify sys.path before the Python "
                         "interpreter is initialized.");
  if (path.empty()) {
    return;
  }

  PyRef sys(PyImport_ImportModule("sys"));
  if (!sys) {
    py_throw("importing the 'sys' module");
  }
  PyRef sys_path = sys.attr("path");
  EMULATOR_INFER_REQUIRE(static_cast<bool>(sys_path) &&
                             PyList_Check(sys_path.get()),
                         "sys.path is missing or is not a list.");

  PyRef entry = py_string(path);
  const Py_ssize_t n = PyList_Size(sys_path.get());
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject *item = PyList_GetItem(sys_path.get(), i); // borrowed
    if (item != nullptr && py_to_string(item) == path) {
      return; // already present
    }
  }
  if (PyList_Insert(sys_path.get(), 0, entry.get()) != 0) {
    py_throw("inserting '" + path + "' into sys.path");
  }
}

// ===========================================================================
// FpeGuard
// ===========================================================================

FpeGuard::FpeGuard() {
#ifdef EMULATOR_HAS_FEENABLEEXCEPT
  m_saved_excepts = fegetexcept();
  if (m_saved_excepts > 0) {
    fedisableexcept(m_saved_excepts);
  }
#endif
}

FpeGuard::~FpeGuard() {
#ifdef EMULATOR_HAS_FEENABLEEXCEPT
  if (m_saved_excepts > 0) {
    feclearexcept(m_saved_excepts);
    feenableexcept(m_saved_excepts);
  }
#endif
}

// ===========================================================================
// Error handling and string helpers
// ===========================================================================

std::string py_take_error() {
  if (PyErr_Occurred() == nullptr) {
    return std::string();
  }

  PyObject *type = nullptr;
  PyObject *value = nullptr;
  PyObject *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  std::string message;

  // Prefer traceback.format_exception, which gives the same text a user would
  // see from the interpreter (including the Python-side stack).
  PyRef tb_module(PyImport_ImportModule("traceback"));
  if (tb_module) {
    PyRef formatted(PyObject_CallMethod(
        tb_module.get(), "format_exception", "OOO", type ? type : Py_None,
        value ? value : Py_None, traceback ? traceback : Py_None));
    if (formatted && PyList_Check(formatted.get())) {
      std::ostringstream oss;
      const Py_ssize_t n = PyList_Size(formatted.get());
      for (Py_ssize_t i = 0; i < n; ++i) {
        oss << py_to_string(PyList_GetItem(formatted.get(), i));
      }
      message = oss.str();
    }
  }

  if (message.empty()) {
    // Fall back to str(value) if formatting the traceback itself failed.
    if (value != nullptr) {
      PyRef as_str(PyObject_Str(value));
      message = py_to_string(as_str.get());
    }
    if (message.empty()) {
      message = "<unavailable python error>";
    }
  }

  PyErr_Clear();
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);

  return message;
}

void py_throw(const std::string &context) {
  const std::string details = py_take_error();
  throw InferenceError("Python error while " + context + ":\n" + details);
}

PyRef py_string(const std::string &s) {
  return PyRef(PyUnicode_FromStringAndSize(s.data(),
                                           static_cast<Py_ssize_t>(s.size())));
}

std::string py_to_string(PyObject *obj) {
  if (obj == nullptr) {
    return std::string();
  }
  if (PyUnicode_Check(obj)) {
    Py_ssize_t size = 0;
    const char *utf8 = PyUnicode_AsUTF8AndSize(obj, &size);
    if (utf8 != nullptr) {
      return std::string(utf8, static_cast<std::size_t>(size));
    }
    PyErr_Clear();
    return std::string();
  }
  if (PyBytes_Check(obj)) {
    return std::string(PyBytes_AsString(obj),
                       static_cast<std::size_t>(PyBytes_Size(obj)));
  }
  return std::string();
}

} // namespace inference
} // namespace emulator
