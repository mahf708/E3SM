/**
 * @file python_interpreter.hpp
 * @brief Embedded CPython session management and small RAII helpers.
 *
 * Only compiled when EMULATOR_ENABLE_PYTHON is on.  Everything here uses the
 * plain CPython C API so that embedding needs nothing beyond the Python
 * development headers (no pybind11 build dependency).  EAMxx solves the same
 * problem with pybind11 in `share/core/eamxx_pysession.hpp`; the semantics
 * here are deliberately the same (reference-counted session, never finalize
 * an interpreter we did not start) so the two can coexist in one process.
 */

#ifndef E3SM_EMULATOR_PYTHON_INTERPRETER_HPP
#define E3SM_EMULATOR_PYTHON_INTERPRETER_HPP

// Python.h must come before the standard headers it configures.
#include <Python.h>

#include <string>
#include <utility>

#include "inference_error.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Reference-counted embedded Python interpreter.
 *
 * Several emulators (and other E3SM code) may want Python in one process, and
 * CPython cannot be re-initialized reliably after Py_Finalize, so the
 * interpreter is a process-wide resource with a customer count.
 *
 * If the interpreter is already running when the first customer arrives — the
 * process is a Python process, or EAMxx's PySession got there first — this
 * class attaches to it and never finalizes it.
 */
class PyInterpreter {
public:
  static PyInterpreter &instance();

  PyInterpreter(const PyInterpreter &) = delete;
  PyInterpreter &operator=(const PyInterpreter &) = delete;

  /// @brief Start (or attach to) the interpreter and add one customer.
  void initialize();

  /**
   * @brief Drop one customer.
   *
   * This deliberately does *not* stop the interpreter, even at zero
   * customers.  CPython cannot be re-initialized reliably once finalized:
   * numpy 2.x refuses with "cannot load module more than once per process",
   * and PyTorch behaves the same way, so a second emulator created later in
   * the same run would fail to import anything.  Since a Python emulator is
   * useless without those extensions, the interpreter stays up for the
   * lifetime of the process.  What actually matters — the model, its weights
   * and any cached arrays — is released when the backend drops its Python
   * references.
   *
   * Call shutdown() explicitly if a process really must stop the interpreter.
   */
  void finalize();

  /**
   * @brief Stop the interpreter if we started it and nobody is using it.
   *
   * Provided for completeness (leak checkers, embedding tests).  Anything
   * that imports a C extension module cannot be brought back up afterwards.
   *
   * @return true if the interpreter was stopped by this call
   */
  bool shutdown();

  /// @brief Number of live customers.
  int num_customers() const { return m_customers; }

  /// @brief True if the interpreter is running.
  bool is_running() const { return Py_IsInitialized() != 0; }

  /// @brief True if this class called Py_InitializeEx.
  bool owns_interpreter() const { return m_owns; }

  /**
   * @brief Prepend a directory to sys.path.  Requires the GIL.
   *
   * Prepending (rather than appending) means a run directory can shadow an
   * installed module, which is what a user editing an emulator in place
   * expects.  Duplicate entries are not added.
   */
  void add_sys_path(const std::string &path);

private:
  PyInterpreter() = default;

  int m_customers = 0;
  bool m_owns = false;
  PyThreadState *m_saved_state = nullptr;
};

/**
 * @brief RAII GIL acquisition for the current thread.
 *
 * Every call into Python must be wrapped in one of these.  It is safe when
 * the calling thread already holds the GIL.
 */
class PyGilGuard {
public:
  PyGilGuard() : m_state(PyGILState_Ensure()) {}
  ~PyGilGuard() { PyGILState_Release(m_state); }
  PyGilGuard(const PyGilGuard &) = delete;
  PyGilGuard &operator=(const PyGilGuard &) = delete;

private:
  PyGILState_STATE m_state;
};

/**
 * @brief RAII disable of floating-point exception traps.
 *
 * Importing numpy raises benign FPEs; with trapping enabled (as E3SM debug
 * builds do) the process dies inside the import.  EAMxx's PySession does the
 * same thing around its imports.
 */
class FpeGuard {
public:
  FpeGuard();
  ~FpeGuard();
  FpeGuard(const FpeGuard &) = delete;
  FpeGuard &operator=(const FpeGuard &) = delete;

private:
  int m_saved_excepts = 0;
};

/**
 * @brief Owning handle for a PyObject*, to keep reference counting honest.
 *
 * Construction *steals* a reference (the common case for C-API returns);
 * PyRef::borrow() increments first for borrowed references.
 */
class PyRef {
public:
  PyRef() = default;

  /// @brief Take ownership of a new reference.
  explicit PyRef(PyObject *obj) : m_obj(obj) {}

  /// @brief Add a reference to a borrowed object and own that one.
  static PyRef borrow(PyObject *obj) {
    Py_XINCREF(obj);
    return PyRef(obj);
  }

  ~PyRef() { Py_XDECREF(m_obj); }

  PyRef(const PyRef &other) : m_obj(other.m_obj) { Py_XINCREF(m_obj); }
  PyRef &operator=(const PyRef &other) {
    if (this != &other) {
      Py_XINCREF(other.m_obj);
      Py_XDECREF(m_obj);
      m_obj = other.m_obj;
    }
    return *this;
  }
  PyRef(PyRef &&other) noexcept : m_obj(other.m_obj) { other.m_obj = nullptr; }
  PyRef &operator=(PyRef &&other) noexcept {
    if (this != &other) {
      Py_XDECREF(m_obj);
      m_obj = other.m_obj;
      other.m_obj = nullptr;
    }
    return *this;
  }

  PyObject *get() const { return m_obj; }
  explicit operator bool() const { return m_obj != nullptr; }

  /// @brief Relinquish ownership to a caller that wants a new reference.
  PyObject *release() {
    PyObject *obj = m_obj;
    m_obj = nullptr;
    return obj;
  }

  /// @brief Attribute lookup returning an owning handle (null if absent).
  PyRef attr(const char *name) const {
    if (m_obj == nullptr) {
      return PyRef();
    }
    if (PyObject_HasAttrString(m_obj, name) == 0) {
      return PyRef();
    }
    return PyRef(PyObject_GetAttrString(m_obj, name));
  }

private:
  PyObject *m_obj = nullptr;
};

/**
 * @brief Consume the pending Python exception and format it as a string.
 *
 * Includes the traceback when one is available.  Returns an empty string if
 * no exception is set.  Requires the GIL.
 */
std::string py_take_error();

/**
 * @brief Throw an InferenceError describing the pending Python exception.
 *
 * @param context What we were attempting, e.g. "importing module 'foo'"
 */
[[noreturn]] void py_throw(const std::string &context);

/// @brief Convert a C++ string to a Python str (owning handle).
PyRef py_string(const std::string &s);

/// @brief Read a Python str/bytes as a C++ string (empty if not a string).
std::string py_to_string(PyObject *obj);

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_PYTHON_INTERPRETER_HPP
