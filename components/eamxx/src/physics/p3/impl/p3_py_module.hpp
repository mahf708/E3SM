#ifndef P3_PY_MODULE_HPP
#define P3_PY_MODULE_HPP

#ifdef EAMXX_HAS_PYTHON

#include <pybind11/pybind11.h>

namespace scream {
namespace p3 {

/*
 * P3-local handle to the py::module imported by the AtmosphereProcess
 * base class from the py_module_name yaml entry. Handled locally inside
 * of P3 instead of PySession and atm_proc base. Lifetime is basically
 * like the rest of P3: initialize_impl to finalize_impl
 */
inline const pybind11::module* g_p3_py_module = nullptr;

} // namespace p3
} // namespace scream

#endif // EAMXX_HAS_PYTHON
#endif // P3_PY_MODULE_HPP
